import tensorflow as tf, os, cv2, pdb, numpy as np, time, json, pandas, glob, md5
from scipy.misc import imread, imresize
from scipy import misc
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from shapely.geometry import MultiPolygon, box
from subprocess import check_output
from zipfile import ZipFile

pandas.options.mode.chained_assignment = None

def get_image_dir(weights, test_boxes):
    weights_iteration = int(weights.split('-')[-1])
    return '%s/images_%s_%d' % (os.path.dirname(weights), os.path.basename(test_boxes)[:-5], weights_iteration)

class TensorBox:
    name = 'TensorBox'

    @classmethod
    def mk_hash(cls, path):
        '''
        Create an MD5 hash from a models weight file.
        Arguments:
            path : str - path to TensorBox checkpoint
        '''
        dirs = path.split('/')
        if 'TensorBox' in dirs:
            dirs = dirs[dirs.index('TensorBox'):]
            path = '/'.join(dirs)
        else:
            path = os.path.join('TensorBox', path)

        return md5.new('/'.join(path[-4:])).hexdigest()

    @classmethod
    def zip_weights(cls, path, base_dir='./'):
        if not os.path.exists(path + '.meta'):
            raise ValueError('Invalid TensorBox checkpoint...')

        dirs = path.split('/')

        res = {
            'name' : 'TensorBox',
            'instance' : '_'.join(dirs[-2:]),
            'id' : cls.mk_hash(path)
        }

        zipfile = os.path.join(base_dir, res['id'] + '.zip')

        if os.path.exists(zipfile):
            os.remove(zipfile)

        weight_dir = os.path.dirname(path)

        with ZipFile(zipfile, 'w') as z:
            for file in glob.glob(path + '*'):
                z.write(file, os.path.join(res['id'], os.path.basename(file)))
            z.write(os.path.join(weight_dir, 'hypes.json'), os.path.join(res['id'], 'hypes.json'))

        return zipfile

    def __init__(self, weights = None):
        if weights is None:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            download_url = 'https://github.com/ArnholdInstitute/ColdSpots/releases/download/1.0/tensorbox.zip'
            if not os.path.exists('weights/tensorbox'):
                print('Downloading weights for tensorbox')
                if not os.path.exists(os.path.join('weights/tensorbox.zip')):
                    check_output(['wget', download_url, '-O', 'weights/tensorbox.zip'])
                print('Unzipping...')
                check_output(['unzip', 'weights/tensorbox.zip', '-d', 'weights'])
            description = json.load(open('weights/tensorbox/description.json'))
            weights = os.path.join('weights/tensorbox', description['weights'])
            print('Building model...')
            
        self.weights = weights
        hypes_file = '%s/hypes.json' % os.path.dirname(weights)
        with open(hypes_file, 'r') as f:
            self.H = json.load(f)

        tf.reset_default_graph()
        self.H["grid_width"] = self.H["image_width"] / self.H["region_size"]
        self.H["grid_height"] = self.H["image_height"] / self.H["region_size"]

        self.x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(self.H, tf.expand_dims(self.x_in, 0), 'test', reuse=None)
        grid_area = self.H['grid_height'] * self.H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * self.H['rnn_len'], 2])), [grid_area, self.H['rnn_len'], 2])
        if self.H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas

        self.pred_boxes = pred_boxes
        self.pred_confidences = pred_confidences
        saver = tf.train.Saver()

        self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())
        saver.restore(self.session, weights)

    def close_session(self):
        self.session.close()

    def predict_image(self, image, threshold, eval_mode = False):
        """
        Infer buildings for a single image.
        Inputs:
            image :: n x m x 3 ndarray - Should be in RGB format
        """

        orig_img = image.copy()[:,:,:3]
        img = imresize(orig_img, (self.H["image_height"], self.H["image_width"]), interp='cubic')
        feed = {self.x_in: img}

        t0 = time.time()
        (np_pred_boxes, np_pred_confidences) = self.session.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)
        total_time = time.time() - t0

        new_img, rects, all_rects = add_rectangles(
            self.H, 
            [img], 
            np_pred_confidences, 
            np_pred_boxes,
            use_stitching=True, 
            rnn_len=self.H['rnn_len'], 
            min_conf=threshold, 
            tau=0.25, 
            show_suppressed=False
        )

        pred_anno = al.Annotation()
        pred_anno.rects = all_rects
        pred_anno = rescale_boxes((self.H["image_height"], self.H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])

        pred_rects = pandas.DataFrame([[r.x1, r.y1, r.x2, r.y2, r.score] for r in all_rects], columns=['x1', 'y1', 'x2', 'y2', 'score'])

        if eval_mode:
            return pred_rects[pred_rects['score'] > threshold], pred_rects, total_time
        else:
            return pred_rects[pred_rects['score'] > threshold]


    def predict_all(self, test_boxes_file, threshold, data_dir = None):
        test_boxes = json.load(open(test_boxes_file))
        true_annolist = al.parse(test_boxes_file)
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(test_boxes_file))
        
        total_time = 0.0

        for i in range(len(true_annolist)):
            true_anno = true_annolist[i]

            orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]

            pred, all_rects, time = self.predict_image(orig_img, threshold, eval_mode = True)

            pred['image_id'] = i
            all_rects['image_id'] = i

            yield pred, all_rects, test_boxes[i]






