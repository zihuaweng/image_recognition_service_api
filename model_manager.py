import tensorflow as tf
import json


class Load_models(object):

    def __init__(self, model_pb_file, label_file):
        self.model_pb_file = model_pb_file
        self.label_file = label_file

    def read_json_label(self):
        with open(self.label_file) as f:
            output_data = json.load(f)
            return output_data

    def load_graph(self):
        with tf.Graph().as_default() as graph:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_pb_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


class Load_img_single_model(Load_models):

    def __init__(self, model_pb_file,
                 label_file, gpu_num, input_tensor, output_tensor):
        super().__init__(model_pb_file, label_file)
        self.gpu_num = gpu_num
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def load_label(self):
        label_list = [line.strip() for line in open(self.label_file)]
        return label_list

    def load_tensor_to_run(self):
        with tf.device('/device:GPU:' + str(self.gpu_num)):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            detection_graph = self.load_graph()
            with tf.Session(config=config, graph=detection_graph) as sess:
                input_placeholder = detection_graph.get_tensor_by_name(
                    self.input_tensor)
                softmax = detection_graph.get_tensor_by_name(
                    self.output_tensor)
        return sess, input_placeholder, softmax


class Load_img_multi_model(Load_models):

    def __init__(self, model_pb_file, label_file, gpu_num):
        super().__init__(model_pb_file, label_file)
        self.gpu_num = gpu_num

    def load_label(self):
        with open(self.label_file) as f:
            output_data = json.load(f)
            return output_data

    def load_tensor_to_run(self):
        with tf.device('/device:GPU:' + str(self.gpu_num)):
            config = tf.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5
            config.gpu_options.allow_growth = True
            detection_graph = self.load_graph()
            with tf.Session(config=config, graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name(
                    'detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
        return sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections
