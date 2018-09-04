import tensorflow as tf
import os
import unittest
from .sub_models import SingleTf
from .utils import load_data
from .utils.helpers import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_mock_graph_path():
    return os.path.join('.', 'mock_graph.pb')


def create_mock_graph(graph_file):
    g = tf.Graph()
    with g.as_default():
        tf.placeholder(tf.string, name='Placeholder')
        tf.constant([[0, 0.8, 0.7, 0.6]], name='Predictions')
        # for multi results: eg: 2 examples
        # tf.constant([[0, 0.8, 0.7, 0.6],[0, 0.8, 0.7, 0.6]], name='Predictions')
        graph_def = g.as_graph_def()

    with tf.gfile.Open(graph_file, 'w') as fl:
        fl.write(graph_def.SerializeToString())


def get_mock_label_path():
    return os.path.join('.', 'labels.txt')


def create_mock_label_1(label_file):
    label_contents = 'apple 1\nbanana 1\npen 0\nbook 0'
    with open(label_file, 'w') as f:
        f.write(label_contents)


def create_mock_label_0(label_file):
    label_contents = 'apple 1\npen 0\nbanana 1\nbook 0'
    with open(label_file, 'w') as f:
        f.write(label_contents)


class SingleTfTest(unittest.TestCase):

    def setUp(self):
        self.graph_file = get_mock_graph_path()
        create_mock_graph(self.graph_file)

    def tearDown(self):
        self.model = None
        os.remove(self.graph_file)

    def test_predict_1(self):
        label_file = get_mock_label_path()
        create_mock_label_1(label_file)
        model = SingleTf(self.graph_file, label_file, "")
        model.load_model()
        model.load_label()
        test_img = load_data.generate_test_image(dtype='bytes')
        result = model.predict(test_img, top=1)
        expect_result = {
            'recognitionList': [
                {
                    "className": "banana",
                    "confidence": 0.8,
                }
            ]
        }
        self.assertIsInstance(result['recognitionList'][0]['confidence'], float)
        assertDeepAlmostEqual(self, result, expect_result)
        os.remove(label_file)

    def test_predict_0(self):
        label_file = get_mock_label_path()
        create_mock_label_0(label_file)
        model = SingleTf(self.graph_file, label_file, "")
        model.load_model()
        model.load_label()
        test_img = load_data.generate_test_image(dtype='bytes')
        result = model.predict(test_img, top=1)
        expect_result = {
            'recognitionList': []
        }
        self.assertDictEqual(result, expect_result)
        os.remove(label_file)


if __name__ == '__main__':
    unittest.main()
