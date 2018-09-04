import os
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from gevent.wsgi import WSGIServer
from img_recog_api.single_model_creator import ImageModelSingle
from img_recog_api.utils import log, load_data
import logging
from gevent import monkey

monkey.patch_socket()

# select a gpu if you get more than one
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

log.log_conf('/path/to/log.conf')

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['BUNDLE_ERRORS'] = True
api = Api(app)

config = 'config/config.yaml'


# create and load models
model = ImageModelSingle(config)
logger.info('load model finished!')


class ImageRecog(Resource):

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('imageUrl', type=str,
                                   help='Invalid image url: {error_msg}')
        self.reqparse.add_argument('top', type=int, default=1,
                                   help='Invalid top number, must be int: {error_msg}')
        super(ImageRecog, self).__init__()

    def post(self):
        args = self.reqparse.parse_args(strict=True)
        image_path = args['imageUrl']
        top_class = args['top']

        # precess image
        image = load_data.load_image_url_base64_bytes(image_path, 'url')

        # predict image
        results = model.predict(image, top=top_class)
        return jsonify(results)

api.add_resource(ImageRecog, '/image/recognition')

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8888), app)
