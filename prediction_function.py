# Import libraries
from PIL import Image
import numpy as np
import tensorflow as tf

# Define functions
# TODO: Create the process_image function
def process_image(image_array, image_size=224):
    tf_im = tf.convert_to_tensor(image_array)
    tf_im = tf.image.resize(tf_im, (image_size, image_size))
    tf_im = tf.cast(tf_im, tf.float32)
    tf_im /= 255
    return tf_im.numpy()

# TODO: Create the predict function
def predict(image_path, model, top_k):
    with Image.open(image_path) as im:
        # convert image to numpy array then preprocess it
        im_arr = np.asarray(im)
        im_arr = process_image(im_arr)

        # expand dimension to convert array from shape (224,224,3) to (1,224,224,3)
        im_arr = np.expand_dims(im_arr, axis=0)

        # make prediction
        probs = model.predict(im_arr).flatten()

        # sort the index based on probabilities
        sorted_probs_index = np.argsort(probs)[::-1]
        top_probs_index = sorted_probs_index[:top_k]
        top_probs_value = probs[sorted_probs_index][:top_k]

        return top_probs_value, top_probs_index