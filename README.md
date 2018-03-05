# Serving-multiple-tensorflow-models
Serve multiple tensorflow models with python using flask + gevent + gunicorn.

In this demo, I deployed InceptionResnetV2 model and the [goole object detection model API](https://github.com/tensorflow/models/tree/master/research/object_detection).

**Is easy to deploy multiple models in this way. Also, [tensorflow serving](https://www.tensorflow.org/serving/) maybe a good choose for long-term use.**

## Requirement:
- Ubuntu 16.04
- python3
- pip install -r requirements.txt

## How to use it:
- change config.yaml according to your models
- add config information in gunicorn_all.config. For detail information pleae view [gunicorn setting document](http://docs.gunicorn.org/en/latest/settings.html)
- change the Load_img_single_model(InceptionResnetV2) and Load_img_multi_model(goole object detection) classes in model_manager.py according to your models.
- specify how to run the your session in image_recognition_all.py.

### Run the server:
gunicorn -c gunicorn_all.config image_recognition_all.py:app -D

### Test it:
in terminal:
- first model:

  curl http://your ip:8888/recognition/single?imageUrl=url----image3.jpg
- second model:

  curl http://your ip:8888/recognition/multi?imageUrl=url----image3.jpg

## TODO
- remove the warm-up part (If anyone come up a good idea to speed up the first testing plase contact me :) )
