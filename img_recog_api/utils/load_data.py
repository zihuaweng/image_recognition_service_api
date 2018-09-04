import urllib.request
import numpy as np
import cv2
import base64
import yaml
from collections import ChainMap
from PIL import Image


def load_image_url_bytes(image_url):
    resp = urllib.request.urlopen(image_url)
    img_bytes = resp.read()
    return img_bytes


def load_image_base64_bytes(image_base64):
    img_bytes = base64.b64decode(image_base64)
    return img_bytes


def PIL_rescale(img, max_length):
    img.thumbnail((max_length, max_length), Image.ANTIALIAS)

    return img


def load_label_list_tag(list_file):
    """
    load label file from format:
    apple 1
    banana 1
    pen 0
    """
    with open(list_file) as f:
        labels = [line.strip().split() for line in f.readlines()]
    return labels


class DeepChainMap(ChainMap):
    def __init__(self, *maps):
        super(DeepChainMap, self).__init__(*filter(None, maps))

    def __getattr__(self, k):
        k_maps = list()
        for m in self.maps:
            v = m.get(k)
            if isinstance(v, dict):
                k_maps.append(v)
            elif v is not None:
                return v
        return DeepChainMap(*k_maps)


def load_config(config_yaml):
    """Load model config"""
    with open(config_yaml) as f:
        conf = yaml.load(f)

    # conf = DeepChainMap(conf)

    return conf


def generate_test_image(dtype, size=None):
    """Generate test image for model warm up.

    # Arguments
        dtype: str, test image dtype. Only support 'array', 'string'
        size: 2-4 dimensional tuple or list, only eg: (1, 50, 50,3) or (50, 50, 3) or gray image (50, 50)
        eg:
        generate_test_image(dtype='array', size=(1, 299, 299, 3))

    # Returns
        array or string of image
    """
    if dtype == 'array':
        if len(size) == 4:
            img = np.random.randint(255, size=size[1:])
            img_np = np.expand_dims(img, axis=0)
            return img_np.astype(np.uint8)
        elif len(size) == 3 or len(size) == 2:
            img_np = np.random.randint(255, size=size)
            return img_np.astype(np.uint8)
        else:
            raise ValueError(
                'Invalid value for size. Must be shape of image. '
                'eg: (1, 50, 50,3) or (50, 50, 3) or gray image (50, 50)')

    elif dtype == 'bytes':
        img = np.random.randint(255, size=(299, 299, 3))
        img_str = cv2.imencode('.jpg', img)[1].tostring()
        return img_str

    else:
        raise ValueError('Unsupported dtype. Please Choose from "array" or "string".')


def load_image_url_base64_bytes(image, dtype_in='url'):
    if dtype_in == 'url':
        try:
            img = load_image_url_bytes(image)
            return img
        except ValueError:
            print('Invalid image url.')
    elif dtype_in == 'base64':
        try:
            img = load_image_base64_bytes(image)
            return img
        except ValueError:
            print('Invalid image base64 string.')
    else:
        raise KeyError('Unknown dtype_in value: {}'.format(dtype_in))
