# Api for Image Recognition Service
Easy way to deploy multiple models with deep learning frameworks for your products.

深度学习模型多模型线上部署API. 可根据需要快速部署tensorflow, Keras模型(flask + gevent + gunicorn). 有更高级的需求可以使用[tensorflow serving](https://www.tensorflow.org/serving/)

## My test environment:
- Ubuntu 16.04
- python3


## Installation

    python setup.py build
    python setup.py install

## Usage

    from img_recog_api.model_creator import ImageModelSingle

    config = 'path/to/config'

    # image recognition
    # create and load models
    Model_s = ImageModelSingle(config)
    # load image to api support format
    image = your_load_image_func(image)
    # predict image
    results_s = Model_s.predict(image, top=4)

## configs
Contains all config files.
- config.yaml
```
model_type: SingleTf
model_file: /path/to/model.pb
label_file: /path/to/labels.txt
# gpu_num could be set as any int. for cpu using gpu_num: ""
gpu_num: 0
input_tensor: input tensor name of tensorflow model(eg: Placeholder:0), if None, find it with pattern
output_tensor: output tensor name of tensorflow model(eg: Prediction:0), if None, find it with pattern
```

## Outputs

ImageModelSingle()

model_type=SingleTf

    **result = model.predict(image, top)**

    {
      "recognitionList": [
        {
          "className": "label1",
          "confidence": 0.2440231442451477
        }
      ]
    }


## Add your own model
Just over write the sub_models.py and add new class for your model like the example SingleTf.

## Deploy your models

In this demo, I deployed a serving model with python using flask + gevent + gunicorn.

**Is easy to deploy multiple models in this way. But [tensorflow serving](https://www.tensorflow.org/serving/) maybe a good choose for long-term use.**

### How to use it:
- cd to example folder.
- add your prediction scripts in sub_model.py and install the api.
- change config.yaml according to your models.
- add config information in gunicorn.config. For detail information pleae view [gunicorn setting document](http://docs.gunicorn.org/en/latest/settings.html)
- change image_recognition_all.py.


### Run the server:
    sh ./example_run.sh

### Test it:
    python example/service_test.py

## TODO
- remove the warm-up part (If anyone comes up a good idea to speed up the first test, feel free to contact me :) )
