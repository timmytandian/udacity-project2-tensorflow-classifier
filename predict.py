# Hide some warning messages from Tensorflow GPU if we run with CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#########################
# IMPORT LIBRARIES
#########################
# Import libraries
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import prediction_function as func
import argparse

#########################
# Read the command line arguments
#########################
parser = argparse.ArgumentParser()
parser.add_argument('image_path',
                    help='The relative path of image to read. Input example: ./test_images/cautleya_spicata.jpg',
                    type=str
)
parser.add_argument('model',
                    help='The folder name containing Tensorflow SavedModel model. Input example: gcolab_epoch100_1656146260',
                    type=str
)
parser.add_argument('-K', '--top_k',
                    dest='top_k',
                    help='The top K classes with highest prediction probability to be returned. Input example: 5',
                    type=int
)
parser.add_argument('-C', '--category_names',
                    #default='label_map.json',
                    dest='category_names',
                    help='A .json file containing the mapping of class into category name or label. Input example: label_map.json',
                    type=str
)
args = parser.parse_args()

#########################
# LOAD MODEL
#########################
# Load the tf SavedModel
model_filepath = args.model + '/'
model = tf.keras.models.load_model(model_filepath)
#print(model.summary())

#########################
# MAKE PREDICTION
#########################
# read image
#test_images_dict = {'0':'cautleya_spicata','1':'hard-leaved_pocket_orchid','2':'orange_dahlia','3':'wild_pansy'}
#image_path = './test_images/{0}.jpg'.format(test_images_dict['0'])
image_path = args.image_path
im = Image.open(image_path)
im_array = np.asarray(im)

# make prediction based on the top_k arguments
if args.top_k:
    top_k = args.top_k
    probs, classes = func.predict(image_path, model, top_k)
    print('Top {0} class prediction: {1}'.format(top_k,classes))
    print('Top {0} prediction probability: {1}'.format(top_k,probs))
else:
    top_k = 1
    probs, classes = func.predict(image_path, model, top_k)
    print('Class prediction: {0}'.format(classes[0]))
    print('Prediction probability: {0}'.format(probs[0]))

# map the class as label if a json file is defined in 'category_names' argument
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        labels = [class_names[str(i+1)] for i in classes.tolist()]
        if top_k > 1:
            print('Labels of top {0} prediction: {1}'.format(top_k,labels))
        else:
            print('Label: ', labels[0])