#!/usr/bin/env python

import tensorflow as tf, os, json, subprocess, cv2, argparse, pdb, rtree, numpy as np
from scipy.misc import imread, imresize
from scipy import misc
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from shapely.geometry import MultiPolygon, box

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.test_boxes)[:-5], weights_iteration, expname)
    return image_dir

def get_metrics(gt_boxes, pred_boxes):
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    total_overlap = 0.0

    # Create the RTree out of the ground truth boxes
    idx = rtree.index.Index()
    for j, rect in enumerate(gt_boxes):
        idx.insert(j, tuple(rect))

    gt_mp = MultiPolygon([box(*b) for b in gt_boxes])
    pred_mp = MultiPolygon([box(*b) for b in pred_boxes])

    for rect in pred_boxes:
        best_jaccard = 0.0
        best_idx = None
        best_overlap = 0.0
        for gt_idx in idx.intersection(rect):
            gt = gt_boxes[gt_idx]
            intersection = (min(rect[2], gt[2]) - max(rect[0], gt[0])) * (min(rect[3], gt[3]) - max(rect[1], gt[1]))
            rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            union = rect_area + gt_area - intersection
            jaccard = float(intersection) / float(union)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_idx = gt_idx
            if intersection > best_overlap:
                best_overlap = intersection
        if best_idx is None or best_jaccard <= 0.00000000000001:
            false_positives += 1
        else:
            idx.delete(best_idx, gt_boxes[best_idx])
            true_positives += 1
        total_overlap = best_overlap
    total_jaccard = total_overlap / (gt_mp.area + pred_mp.area - total_overlap) if len(gt_boxes) > 0 else None
    false_negatives = len(gt_boxes) - true_positives
    return false_positives, false_negatives, true_positives, total_jaccard


def get_results(args, H):
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

        pred_annolist = al.AnnoList()

        true_annolist = al.parse(args.test_boxes)
        data_dir = os.path.join(os.path.dirname(args.test_boxes))

        false_positives, false_negatives, true_positives = 0,0,0

        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        for i in range(len(true_annolist)): 
            true_anno = true_annolist[i]

            orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]
            img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}

            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            pred_anno = al.Annotation()
            pred_anno.imageName = true_anno.imageName
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
            pred_anno.rects = rects
            pred_anno.imagePath = os.path.abspath(data_dir)
            pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
            pred_annolist.append(pred_anno)

            prediction = np.array([[r.x1, r.y1, r.x2, r.y2] for r in rects])
            targets = np.array([[r.x1, r.y1, r.x2, r.y2] for r in true_anno.rects])

            fp, fn, tp, jaccard = get_metrics(targets, prediction)
            false_positives += fp
            false_negatives += fn
            true_positives += tp

            precision = np.float64(true_positives)/(true_positives + false_positives)
            recall = np.float64(true_positives)/(true_positives + false_negatives)

            print('[%d/%d]: False positives: %d, False negatives: %d, True positives: %d, Precision: %f, Recall: %f' % 
                (i, len(true_annolist), false_positives, false_negatives, true_positives, precision, recall))

            if true_positives < (false_positives + false_negatives):
                actual = orig_img.copy()
                pred = orig_img.copy()

                for rect in rects:
                    cv2.rectangle(pred, (int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)), (0,0,255))

                for rect in true_anno.rects:
                    cv2.rectangle(actual, (int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)), (0,255,0))

                data = np.concatenate([pred, actual], axis=1)
                cv2.imwrite('test.jpg', data)

            imname = '%s/%s' % (image_dir, os.path.basename(true_anno.imageName))
            misc.imsave(imname, new_img)
            if i % 25 == 0:
                print(i)
    return pred_annolist, true_annolist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--test_boxes', required=True)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=False, type=bool)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''
    pred_boxes = '%s.%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))
    true_boxes = '%s.gt_%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))

    pred_annolist, true_annolist = get_results(args, H)

if __name__ == '__main__':
    main()