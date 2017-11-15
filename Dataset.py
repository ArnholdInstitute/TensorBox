
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2, boto3, psycopg2
import glob, pdb, os, re, json, random, numpy as np
from shapely.geometry import shape, box
from datetime import datetime
from skimage import io
from multiprocessing import Process, Queue

SIZE = 500

def RandomSampler(conn, country):
    ts = datetime.now().isoformat()
    s3 = boto3.client('s3')
    read_cur = conn.cursor()
    write_cur = conn.cursor()

    read_cur.execute("""
        SELECT filename, ST_AsGeoJSON(shifted)::json FROM buildings.images
        WHERE project=%s AND (done IS NULL OR done=false)
        ORDER BY random()
    """, (country,))

    for filename, geom in read_cur:
        params = {'Bucket' : 'dg-images', 'Key' : filename}
        url = s3.generate_presigned_url(ClientMethod='get_object', Params=params)
        # Convert from RGB -> BGR and also strip off the bottom logo

        img = io.imread(url)[:-25,:,(2,1,0)]

        for i in range(0, img.shape[0], SIZE):
            for j in range(0, img.shape[1], SIZE):
                x, y = i, j
                if i+SIZE > img.shape[0]:
                    x = img.shape[0] - SIZE
                if j + SIZE > img.shape[1]:
                    y = img.shape[1] - SIZE
                orig = img[x:x+SIZE, y:y+SIZE, :]
                yield (
                    orig.copy(),
                    orig,
                    (x, y, filename, img, shape(geom))
                )
        write_cur.execute("UPDATE buildings.images SET done=true WHERE project=%s AND filename=%s", (country, filename))
        conn.commit()

def helper(rows, thread_id, queue, cache_images, data_dir):
    s3 = boto3.client('s3')
    N = len(rows)
    for img_num, (filename,) in enumerate(rows):
        if not os.path.exists(os.path.join(data_dir, filename)):
            attempts = 0
            while attempts < 3:
                try:
                    params = {'Bucket' : 'dg-images', 'Key' : filename}
                    url = s3.generate_presigned_url(ClientMethod='get_object', Params=params)
                    # Convert from RGB -> BGR and also strip off the bottom logo
                    img = io.imread(url)
                    break
                except Exception as e:
                    attempts += 1
            if cache_images:
                io.imsave(os.path.join(data_dir, filename), img)
            img = img[:-25,:,:] # Strip out logo
            
        else:
            img = io.imread(os.path.join(data_dir, filename))[:-25, :, :]

        for i in range(0, img.shape[0], SIZE):
            for j in range(0, img.shape[1], SIZE):
                x, y = i, j
                if i+SIZE > img.shape[0]:
                    x = img.shape[0] - SIZE
                if j + SIZE > img.shape[1]:
                    y = img.shape[1] - SIZE
                orig = img[x:x+SIZE, y:y+SIZE, :]
                valid_geom = box(j, i, y+SIZE, x+SIZE)
                done = i + SIZE >= img.shape[0] and j+SIZE >= img.shape[1]
                queue.put((
                    orig,
                    (x, y, filename, valid_geom, done, SIZE, SIZE)
                ))
        print('Thread %d: done with %d/%d' % (thread_id, img_num, N))
    queue.put(None)

def InferenceGenerator(conn, country, area_to_cover = None, transform=lambda x: x, cache=True, data_dir = './', threads=1):
    s3 = boto3.client('s3')

    if cache and not os.path.exists(os.path.join(data_dir, 'images/%s' % country)):
        os.makedirs(os.path.join(data_dir, 'images/%s' % country))

    condition = ' AND last_tested IS NULL'
    if area_to_cover:
        condition += " AND ST_Intersects(ST_GeomFromText('%s', 4326), shifted)" % area_to_cover.wkt

    with conn.cursor() as cur:
        cur.execute("""
            SELECT filename FROM buildings.images
            WHERE project=%%s %s
        """ % condition, (country,))

        rows = cur.fetchall()

        print('Processing %d files' % len(rows))

        queue = Queue(maxsize=10)
        procs = []
        chunk_size = len(rows) / threads
        for i in range(threads):
            chunk = rows[i*chunk_size:min((i+1)*chunk_size, len(rows))]
            p = Process(target=helper, args=(chunk, i, queue, cache, data_dir))
            p.start()
            procs.append(p)

        done_count = 0
        while done_count < threads:
            item = queue.get()
            if item is None:
                done_count += 1
                continue
            yield item

        for p in procs:
            p.join()





