# Serving-multiple-tensorflow-models
Serving multiple tensorflow models with python: flask + gevent + gunicorn

**Is easy to deploy multiple models in this way. Also, [tensorflow serving](https://www.tensorflow.org/serving/) maybe a good choose for long-term use.**

## Requirement:
- Ubuntu 16.04
- python3
- pip install -r requirements.txt

## How to use it:
- change config.yaml according to your models
- add config information in gunicorn_all.config. For detail information pleae view [gunicorn setting document](http://docs.gunicorn.org/en/latest/settings.html)

### Run the server:
gunicorn -c gunicorn_all.config image_recognition_all.py:app -D

### Test it:
in terminal:
- first model:

  curl http://your ip:8888/recognition/single?imageUrl=url----image3.jpg
- second model:

  curl http://your ip:8888/recognition/multi?imageUrl=url----image3.jpg
