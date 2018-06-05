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
    
    # continually poll for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialise the image IDs and batch of images themselves
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None
        
        # loop over the queue
        for q in queue:
            # deserialise the object and obtain the input image
            q = json.loads(q.decode('utf-8'))
            image = base64_decode_image(q['image'], IMAGE_DTYPE,
                                        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
            
            # check to see if the batch list is None
            if batch is None:
                batch = image
            # otherwise, stack the data
            else:
                batch = np.vstack([batch, image])
                
            # update the list of image IDs
            imageIDs.append(q['id'])
            
        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            # classify the batch
            print('* Batch size: {}'.format(batch.shape))
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds)
            
            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(imageIDs, results):
                output = []
                
                for (imagenetID, label, prob) in resultSet:
                    r = {'label': label, 'probability': float(prob)}
                    output.append(r)
                
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))
                
            # remove the set of images from our queue
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
            
        time.sleep(SERVER_SLEEP)
        

@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}
    
    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialise it
            image = image.copy(order='C')
            
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            d = {'id': k, 'image': base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))
            
            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)
                
                # check to see if our model has classified the input
                # image
                if output is not None:
                    output = output.decode('utf-8')
                    data['predictions'] = json.loads(output)
                    
                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break
                
                time.sleep(CLIENT_SLEEP)
                
            data['success'] = True
            
        return flask.jsonify(data)
    
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == '__main__':
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    print('* Starting model service...')
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()
    
    # start the web server
    print('* Starting web service...')
    app.run()