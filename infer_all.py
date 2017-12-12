#!/usr/bin/env python

import tensorflow as tf, os, json, subprocess, cv2, argparse, pdb, rtree, numpy as np, psycopg2, md5, math
from scipy.misc import imread, imresize
from scipy import misc
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes, get_rectangles
from shapely.geometry import MultiPolygon, box, shape
from multiprocessing import Queue, Process
from Dataset import InferenceGenerator
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from geopy.distance import VincentyDistance
load_dotenv(find_dotenv())

def gsd(lat, zoom):
    """
    Computes the Ground Sample Distance (GSD).  More details can be found
    here: https://msdn.microsoft.com/en-us/library/bb259689.aspx

    Args:
        lat : float - latitude of the GSD of interest
        zoom : int - zoom level (WMTS)
    """
    return (math.cos(lat * math.pi / 180) * 2 * math.pi * 6378137) / (256 * 2**zoom)

def raster_to_proj(x, y, img_geom, ref_point):
    (lon,), (lat,) = img_geom.centroid.xy
    return(
        VincentyDistance(meters=gsd(lat, 18) * x).destination(point=ref_point, bearing=90).longitude,
        VincentyDistance(meters=gsd(lat, 18) * y).destination(point=ref_point, bearing=180).latitude
    )

def process_results(queue, H, args, db_args, data_dir, ts):
    conn = psycopg2.connect(**db_args)
    cur = conn.cursor()
    while True:
        item = queue.get()
        if item is None:
            return

        (np_pred_boxes, np_pred_confidences), meta, VERSION = item
        pred_anno = al.Annotation()
        rects = get_rectangles(
            H, 
            np_pred_confidences, 
            np_pred_boxes,
            use_stitching=True, 
            rnn_len=H['rnn_len'], 
            min_conf=args.min_conf, 
            tau=args.tau, 
        )

        (roff, coff, filename, valid_geom, done, height, width, img_geom) = meta
        img_geom = shape(img_geom)

        pred_anno.rects = rects
        pred_anno.imagePath = os.path.abspath(data_dir)
        pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, height, width)

        bounds = img_geom.bounds
        ref_point = (bounds[3], bounds[0]) # top left corner


        for r in rects:
            minx, miny = raster_to_proj(r.x1 + coff, r.y1 + roff, img_geom, ref_point)
            maxx, maxy = raster_to_proj(r.x2 + coff, r.y2 + roff, img_geom, ref_point)
            building = box(minx, miny, maxx, maxy)

            cur.execute("""
                INSERT INTO buildings.buildings (filename, minx, miny, maxx, maxy, roff, coff, score, project, ts, version, geom)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, ST_GeomFromText(%s, 4326))
            """, (filename, int(r.x1), int(r.y1), int(r.x2), int(r.y2), roff, coff, r.score, args.country, ts, VERSION, building.wkt))
        
        if done:
            cur.execute("UPDATE buildings.images SET last_tested=%s WHERE project=%s AND filename=%s", (ts, args.country, filename))
            conn.commit()
            print('Committed image: %s' % filename)

def infer_all(args, H, db_args):
    VERSION = md5.new(args.weights).hexdigest()
    
    conn = psycopg2.connect(**db_args)
    tf.reset_default_graph()
    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    queue = Queue()
    ts = datetime.now().isoformat()

    ps = []
    for _ in range(4):
        processor = Process(target=process_results, args=(queue, H, args, db_args, '../data', ts))
        processor.start()
        ps.append(processor)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)
        area_to_cover = None
        if args.boundary:
            area_to_cover = shape(json.load(open(args.boundary)))
        generator = InferenceGenerator(conn, args.country, area_to_cover=area_to_cover, data_dir='../data', threads=8)
        for orig_img, meta in generator:
            img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}
            result = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            queue.put((result, meta, VERSION))

    for p in ps:
        queue.put(None)
        p.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--country', required=True)
    parser.add_argument('--boundary', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    db_args = {
        'dbname' : 'aigh',
        'host' : os.environ.get('DB_HOST', 'localhost'),
        'user' : os.environ.get('DB_USER', ''),
        'password' : os.environ.get('DB_PASSWORD', '')
    }

    infer_all(args, H, db_args)
 
if __name__ == '__main__':
    main()