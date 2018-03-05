import os
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from _datetime import datetime
import time
import numpy as np
import urllib
from gevent.wsgi import WSGIServer
from skimage import io
import logging
import yaml
from model_manager import Load_img_single_model, Load_img_multi_model
from gevent import monkey
monkey.patch_socket()

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

with open('config.yaml') as f:
    config = yaml.load(f)

img_s_config = config['image_recogniton_single']
img_m_config = config['image_recogniton_multi']
warm_up_config = config['warm_up']

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['BUNDLE_ERRORS'] = True
api = Api(app)


def setup_logger(name, log_file, level=logging.DEBUG):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    return logger


def load_image_url_numpy(image_url):
    return io.imread(image_url)


def load_image_url_string(image_url):
    resp = urllib.request.urlopen(image_url)
    img = resp.read()
    return img


def time_wrapper(logger):
    def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            logger.info('%r (%r, %r) %2.2f sec' %
                        (method.__name__, args, kw, te - ts))
            return result
        return timed
    return timeit


# load image_recognition_single model
logger_img_s = setup_logger(img_s_config['model_name'],
                            img_s_config['log_file'])
logger_img_s.info('Start loadding {} model'.format(img_s_config['model_name']))
st_date_s = datetime.now()

image_recognition_single = Load_img_single_model(img_s_config['model_pb_file'],
                                                 img_s_config['label_file'],
                                                 img_s_config['gpu_num'],
                                                 img_s_config['input_tensor'],
                                                 img_s_config['output_tensor'])

img_s_label = image_recognition_single.load_label()
img_s_sess, img_s_img_tensor, img_s_softmax = image_recognition_single.load_tensor_to_run()
duration = datetime.now() - st_date_s
logger_img_s.info(
    'Successfully loaded {} model. Duration: {}'.format(img_s_config['model_name'], duration))

# get warn-up
n = 1
while n <= warm_up_config['times']:
    st_date_s = datetime.now()
    warm_up_image_np = load_image_url_string(warm_up_config['image'])
    probabilities = img_s_sess.run(
        img_s_softmax, feed_dict={img_s_img_tensor: warm_up_image_np})
    duration = datetime.now() - st_date_s
    logger_img_s.info(
        'Successfully warm up {} model : {}, Duration: {}'.format(img_s_config['model_name'], n, duration))
    n += 1

# load image_recognition_multi model
logger_img_m = setup_logger(img_m_config['model_name'],
                            img_m_config['log_file'])
logger_img_m.info('Start loadding {} model'.format(img_m_config['model_name']))
st_date_m = datetime.now()

image_recognition_multi = Load_img_multi_model(img_m_config['model_pb_file'],
                                               img_m_config['label_file'],
                                               img_m_config['gpu_num'])

img_m_label = image_recognition_multi.load_label()
img_m_class_lst = list(map(int, img_m_label.keys()))
img_m_sess, img_m_img_tensor, img_m_detection_boxes, img_m_detection_scores, img_m_detection_classes, img_m_num_detections = image_recognition_multi.load_tensor_to_run()
duration = datetime.now() - st_date_m
logger_img_m.info(
    'Successfully loaded {} model. Duration: {}'.format(img_m_config['model_name'], duration))

# get warm up
n = 1
while n <= warm_up_config['times']:
    st_date_m = datetime.now()
    warm_up_image_np = load_image_url_numpy(warm_up_config['image'])
    warm_up_image_np = np.expand_dims(warm_up_image_np, axis=0)
    (boxes, scores, classes, num) = img_m_sess.run(
        [img_m_detection_boxes, img_m_detection_scores,
         img_m_detection_classes, img_m_num_detections],
        feed_dict={img_m_img_tensor: warm_up_image_np})
    duration = datetime.now() - st_date_m
    logger_img_m.info(
        'Successfully warm up {} model : {}, Duration: {}'.format(img_m_config['model_name'], n, duration))
    n += 1


class ImageRecogSingle(Resource):

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('imageUrl', type=str,
                                   help='Invalid image url: {error_msg}')
        # self.reqparse.add_argument('top', type=int,
        #                            help='Invalid top number, must be int: {error_msg}')
        super(ImageRecogSingle, self).__init__()

    @time_wrapper(logger_img_s)
    def get(self):
        args = self.reqparse.parse_args(strict=True)
        image_path = args['imageUrl']
        # top_class = args['top']
        top_class = 1
        logger_img_s.info('Receive image url: {}'.format(image_path))

        try:
            image_string = load_image_url_string(image_path)
            logger_img_s.info('Successfully open image')
        except Exception:
            logger_img_s.error('Failed to open image', exc_info=True)
            return {}
        sess_start = time.time()
        probabilities = img_s_sess.run(
            img_s_softmax, feed_dict={img_s_img_tensor: image_string})
        sess_end = time.time()
        logger_img_s.info('Successfully run graph in {} secs'.format(
            sess_end - sess_start))
        probabilities = probabilities[0, 0:]  # or
        sorted_inds = [i[0] for i in sorted(
            enumerate(-probabilities), key=lambda x: x[1])]  # or
        # probabilities = np.squeeze(probabilities)
        # sorted_inds = probabilities.argsort()[-5:][::-1]
        top_result = []
        tmp_list = []
        for i in range(top_class):
            index = sorted_inds[i]
            tmp_list.append(img_s_label[index])
        top_result.append(tmp_list)

        return jsonify(top_result)


class ImageRecogMulti(Resource):

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('imageUrl', type=str,
                                   help='Invalid image url: {error_msg}')
        super(ImageRecogMulti, self).__init__()

    @time_wrapper(logger_img_m)
    def get(self):
        args = self.reqparse.parse_args(strict=True)
        image_path = args['imageUrl']
        logger_img_m.info('Receive image url: {}'.format(image_path))

        try:
            image_np = load_image_url_numpy(image_path)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            logger_img_m.info('Successfully open image')
        except Exception:
            logger_img_m.error('Failed to open image', exc_info=True)
            return {}
        sess_start = time.time()
        (boxes, scores, classes, num) = img_m_sess.run(
            [img_m_detection_boxes, img_m_detection_scores,
                img_m_detection_classes, img_m_num_detections],
            feed_dict={img_m_img_tensor: image_np_expanded})
        # options=options, run_metadata=run_metadata)
        sess_end = time.time()
        logger_img_m.info('Successfully run graph in {} secs'.format(
            sess_end - sess_start))
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        final_result = {}
        tmp_all_box_list = []
        for i in range(0, img_m_config['detection_num']):
            if scores[i] > 0.5:
                tmp_dict = {}
                if classes[i] in img_m_class_lst:
                    tmp_dict['foodName'] = img_m_label[str(
                        classes[i])]['name']
                    tmp_dict['confidence'] = str(scores[i])
                    tmp_all_box_list.append(tmp_dict)
        final_result['recognitionList'] = tmp_all_box_list

        return jsonify(final_result)


api.add_resource(ImageRecogSingle, '/recognition/single')
api.add_resource(ImageRecogMulti, '/recognition/multi')

if __name__ == '__main__':
    http_server = WSGIServer(('your ip', 8888), app)


# test it : http://your ip:8888/recognition/single?imageUrl=url----image3.jpg
# test it : http://your ip:8888/recognition/multi?imageUrl=url----image3.jpg
