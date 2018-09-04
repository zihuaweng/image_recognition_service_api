import tensorflow as tf
from .utils import load_data
from .models import Models
import logging


class SingleTf(Models):

    def __init__(self, model_file, label_file, gpu_num, input_tensor=None, output_tensor=None):
        super().__init__(model_file, label_file)
        self.gpu_num = gpu_num
        self.model = None
        self.labels = None
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def _load_graph(self, model_file):
        with tf.Graph().as_default() as graph:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def load_model(self):

        if isinstance(self.gpu_num, int):
            device = '/device:GPU:' + str(self.gpu_num)
        else:
            device = '/cpu:0'

        with tf.device(device):
            config = tf.ConfigProto()
            if 'GPU' in device:
                # config.gpu_options.per_process_gpu_memory_fraction = 0.5
                config.gpu_options.allow_growth = True
            detection_graph = self._load_graph(self.model_file)

            # get input and output node in graph:
            # just specify it in config
            # if no input_tensor and output_tensor in config, then find it with patten
            if self.input_tensor == None and self.output_tensor == None:
                for op in detection_graph.get_operations():
                    if 'Placeholder' in op.name:
                        self.input_tensor = op.name + ':0'
                    if 'Predictions' in op.name:
                        self.output_tensor = op.name + ':0'

            logging.info('SingleTf -- Loading model with input tensor: {}, output tensor: {}'.format(self.input_tensor,
                                                                                                     self.output_tensor))

            _session = tf.Session(config=config, graph=detection_graph)
            input_placeholder = detection_graph.get_tensor_by_name(self.input_tensor)
            softmax = detection_graph.get_tensor_by_name(self.output_tensor)

        logging.info('SingleTf -- Finish loading model file.')

        # (Optional) warm up loaded model
        # it would speed up the prediction as the first round would be too slow
        # just mimic a test data and predict it a few times
        test_image = load_data.generate_test_image(dtype='bytes')
        for i in range(2):
            _ = _session.run(
                softmax, feed_dict={input_placeholder: test_image})

        logging.info('SingleTf -- Finish model warm up.')

        self.model = (_session, input_placeholder, softmax)

    def load_label(self):
        self.labels = load_data.load_label_list_tag(self.label_file)
        logging.info('SingleTf -- Finish loading label file.')

    def predict(self, image, top=5):
        '''Prediction for single object

        Args:
            image: image file dytes.
            top: top results to return.

        Returns:
              A json format predictions with top number of predicted item classes and probabilities.

        '''

        if self.model and self.labels:
            sess, input_placeholder, softmax = self.model
            labels = self.labels
            top = int(top)
        else:
            raise Exception(
                'Model was not initialized. Please load model using Image_model_single.create_model() first.')

        probabilities = sess.run(softmax, feed_dict={input_placeholder: image})

        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(
            enumerate(-probabilities), key=lambda x: x[1])]

        top_result = {}
        recognition_list = []

        top_num = min(top, len(labels))
        for i in range(top_num):
            tmp_result = {}
            index = sorted_inds[i]
            tmp_result['className'] = labels[index][0]
            tmp_result['confidence'] = probabilities[index].item()
            recognition_list.append(tmp_result)

        top_result['recognitionList'] = recognition_list

        return top_result