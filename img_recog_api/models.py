from abc import ABC, abstractmethod


class Models(ABC):
    """Abstract base class for image model prediction"""

    def __init__(self, model_file, label_file):
        """specify a model type."""
        self.model_file = model_file
        self.label_file = label_file
        self.model = None
        self.labels = None

    @abstractmethod
    def load_model(self):
        """(optional) load model file to graph file (tensorflow version).
        Keras or caffe should read into the corresponding model files.
        """

        def _warmup():
            """warm up model for speed
            This function used to generate test images and run model for required times to warmup.
            No need to get any useful results from model execution.
            eg: tensorflow version:
                test_img = load_data.generate_test_image(dtype='url', size=(50, 50, 3))
                _ = sess.run(softmax, feed={inpu_placeholder: test_img})
            """
            pass

        _warmup()

        model = None

        self.model = model

    @abstractmethod
    def load_label(self):
        """load label file to list
        """

        labels = None

        self.labels = labels

    @abstractmethod
    def predict(self, image, **kw):
        """Generates output predictions for the input samples.

        Args:
            image: image file dytes.

        Returns:
            prediction results in json format.
        """

        pass
