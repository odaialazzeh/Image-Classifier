import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("path_to_image", help="Path to the input image", type=str)
parser.add_argument("image_classifier_model", type=str)


parser.add_argument("--top_k", default=5, type=int)
parser.add_argument("--classes_names", default="./label_map.json", type=str)

args = parser.parse_args()   

path_to_image = args.path_to_image
image_classifier_model = args.image_classifier_model
top_k = args.top_k
classes_names = args.classes_names

print('Used model:', image_classifier_model)
print('Number of top k class:', top_k) 

# Load a JSON file that maps the class values to classes names
with open(classes_names, 'r') as f:
    class_names = json.load(f)

# Load the saved Keras model 
loaded_keras_model = tf.keras.models.load_model(image_classifier_model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

print(loaded_keras_model.summary())

image_size = 224

# Predict the top K flower classes along with associated probabilities
def predict(path_to_image, model, top_k):
    image = Image.open(path_to_image).resize((image_size, image_size))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)

    top_k_probs = np.sort(predictions[0])[-top_k:][::-1]
    top_k_classes = np.argsort(predictions[0])[-top_k:][::-1]

    return top_k_probs, top_k_classes

top_k_probs, top_k_classes = predict(path_to_image, loaded_keras_model, top_k) 
print('List of flower classes along with corresponding probabilities:', top_k_classes, top_k_probs)

for flower in range(len(top_k_classes)): 
    print('Flower Name:', class_names.get(str(top_k_classes[flower] + 1)))
    print('Class Probability:', top_k_probs[flower])
