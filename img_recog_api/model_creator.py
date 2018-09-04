from .sub_models import *


class ImageModelSingle(object):
    """factory to selection which Model to create"""

    def __init__(self, config):
        """specify a model type."""
        self.model = None
        self.__initialize_load_model(config)

    def __initialize_load_model(self, config):
        conf = load_data.load_config(config)
        sub_model = eval(conf['model_type'])
        conf.pop("model_type")
        self.model = sub_model(**conf)
        self.model.load_model()
        self.model.load_label()

    def predict(self, image, **kw):
        """Generates output predictions for the input samples.

        Args:
            image: image file dytes
            top: top results to return

        Returns:
              A json format predictions with top number of predicted item classes and probabilities.
        """
        return self.model.predict(image, **kw)
