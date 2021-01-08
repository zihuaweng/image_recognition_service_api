#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = [
    'numpy',
    'opencv-python',
    'scikit-image',
   'tensorflow-gpu==2.4.0'
]

setup(
    # Metadata
    name='img_recog_api',
    version=1.0,
    author='Zihua Weng',
    author_email='wengzihua123@126.com',
    description='Api for image recognition and object detection.',
    license='MIT',
    # Package info
    packages=find_packages(exclude=('example', 'log')),
    zip_safe=True,
    install_requires=requirements,
)
