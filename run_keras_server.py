from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io

# initialise constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = 'float32'

# initialise constants used for server queuing
IMAGE_QUEUE = 'image_queue'
BATCH_SIZE = 32
SERVER_SLEEP = .25
CLIENT_SLEEP = .25

# initialise our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host='localhost', port=6379, db=0)
model = None


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode('utf-8')


def base64_decode_image(a, dtype, shape):
    if sys.version_info.major == 3:
        a = bytes(a, encoding='utf-8')
        
    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    
    return a


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    return image


def classify_process():
    """ load the pre-trained Keras model"""
    print('* Loading model...')
    model = ResNet50(weights='imagenet')
    print('* Model loaded')
