#!/usr/bin/env python

import tensorflow as tf, os, json, subprocess, cv2, argparse, pdb, numpy as np, psycopg2
import cStringIO, requests
from sklearn.cluster import DBSCAN
from scipy.misc import imread, imresize
from scipy import misc
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from Dataset import RandomSampler
from shapely.geometry import MultiPolygon, mapping, box
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def process_file(filename, boxes, img_data, country, img_geom):
    if len(boxes) > 0:
        labels = DBSCAN(eps=100, min_samples=1).fit(boxes[:, :2]).labels_
        if (labels >= 0).any():
            # Find the largest cluster
            values, counts = np.unique(labels[labels >= 0], return_counts=True)
            cluster_id = values[np.argmax(counts)]
            mp = MultiPolygon([box(*boxes[i]) for i, _ in enumerate(filter(lambda x: x == cluster_id, labels))])
            (cx,), (cy,) = mp.centroid.xy
            cx, cy = int(cx), int(cy)
        else:
            # No cluster was found, just center around the most confident box
            best_box = np.argmax(boxes[:, -1])
            cx = int(round(boxes[best_box, (0,2)].mean()))
            cy = int(round(boxes[best_box, (1,3)].mean()))

        xmin, xmax = cx - 250, cx + 250
        ymin, ymax = cy - 250, cy + 250

        if xmin < 0:
            dx = -xmin
            xmin, xmax = xmin + dx, xmax + dx
        if xmax > img_data.shape[1]:
            dx = xmax - img_data.shape[1]
            xmin, xmax = xmin - dx, xmax - dx
        if ymin < 0:
            dy = -ymin
            ymin, ymax = ymin + dy, ymax + dy
        if ymax > img_data.shape[0]:
            dy = ymax - img_data.shape[0]
            ymin, ymax = ymin - dy, ymax - dy

        boxes[:, (0, 2)] -= xmin
        boxes[:, (1, 3)] -= ymin

        boxes = np.clip(boxes, a_min=0, a_max=500)
        mask = (boxes[:,2] - boxes[:,0] >= 3) & (boxes[:,3] - boxes[:,1] >= 3)
        boxes = boxes[mask]

        features = [{'geometry' : mapping(box(*b)), 'type' : 'Feature', 'properties' : {}} for b in boxes]
        vdata = {'type' : 'FeatureCollection', 'features' : features}

        binary = cStringIO.StringIO()
        binary.write(cv2.imencode('.jpg', img_data[ymin:ymax, xmin:xmax, :])[1].tostring())
        binary.reset()

        data = {
            'vectordata' : json.dumps(vdata),
            'geom' : json.dumps(mapping(img_geom))
        }

        files = {
            'file' : binary
        }
        res = requests.post('https://aighmapper.ml/sample/%s' % country.replace('-overlap', ''), data=data, files=files)
        if res.status_code == 200:
            print('Successfully uploaded sample to server!')
            return True
        else:
            print(res.text)
            raise Exception(res.text)
    return False

def sample(args, H, conn):
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
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)

        img_iter = RandomSampler(conn, args.country)

        img, orig, (roff, coff, filename, whole_img, img_geom) = next(img_iter)
        current = {'filename' : filename, 'whole_img' : whole_img, 'img_geom' : img_geom, 'boxes' : []}
        cur = conn.cursor()

        upload_samples = 0
        while upload_samples < args.max_samples:
            if current['filename'] and current['filename'] != filename:
                res = process_file(current['filename'], np.array(current['boxes']), current['whole_img'], args.country, current['img_geom'])
                cur.execute("UPDATE buildings.images SET done=true WHERE project=%s AND filename=%s", (args.country, current['filename']))
                conn.commit()
                print('Done with %s' % current['filename'])
                current = {'filename' : filename, 'whole_img' : whole_img, 'img_geom' : img_geom, 'boxes' : []}
                upload_samples += 1 if res else 0

            img = imresize(img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            pred_anno = al.Annotation()
            new_img, rects, _ = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
            pred_anno.rects = rects
            pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig.shape[0], orig.shape[1])
            for r in rects:
                current['boxes'].append(map(int, [r.x1+coff, r.y1+roff, r.x2+coff, r.y2+roff]))

            img, orig, (roff, coff, filename, whole_img, img_geom) = next(img_iter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=False, type=bool)
    parser.add_argument('--country', required=True)
    parser.add_argument('--max_samples', default=20)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''

    conn = psycopg2.connect(
        dbname='aigh',
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', ''),
        password=os.environ.get('DB_PASSWORD', '')
    )

    sample(args, H, conn)

if __name__ == '__main__':
    main()