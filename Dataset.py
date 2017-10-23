
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import  cv2, boto3
import glob, pdb, os, re, json, random, numpy as np
from shapely.geometry import shape
from datetime import datetime
from skimage import io

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

def InferenceGenerator(conn, country, area_to_cover = None, transform=lambda x: x):
    ts = datetime.now().isoformat()
    s3 = boto3.client('s3')

    condition = ''
    if area_to_cover:
        condition = " AND ST_Contains(ST_GeomFromText('%s', 4326), shifted)" % area_to_cover.wkt

    with conn.cursor() as cur:
        cur.execute("""
            SELECT filename FROM buildings.images
            WHERE project=%%s %s
        """ % condition, (country,))

        for filename, in cur:
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
                        torch.from_numpy(transform(orig.copy().astype(float)).transpose((2,0,1))[(2,1,0),:,:]).float(),
                        orig,
                        (x, y, filename)
                    )
