import argparse
import json
import time
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Setup parser
parser = argparse.ArgumentParser(description='TensorFlow ImageNet Validation')
parser.add_argument('--data', type=str, default='imagenet2012', help='Dataset name')
parser.add_argument('--model', type=str, default='EfficientNetB0', help='Model name to use for validation')
parser.add_argument('--weights', type=str, default=None, help='Path to weights file')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for validation')
parser.add_argument('--img-size', type=int, default=224, help='Size of input images')
parser.add_argument('--results-file', type=str, default='', help='Output JSON file for validation results')

args = parser.parse_args()

# Setup TensorFlow Dataset
def preprocess(features):
    image = tf.image.resize(features['image'], (args.img_size, args.img_size))
    return preprocess_input(image), features['label']

ds = tfds.load(args.data, split='validation', as_supervised=False)
ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = ds.batch(args.batch_size)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Load model
if args.weights:
    model = load_model(args.weights)
else:
    model = EfficientNetB0(weights='imagenet')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Perform validation
results = model.evaluate(ds)
results_dict = {
    'loss': results[0],
    'accuracy': results[1]
}

# Print results
print("Validation results:", results_dict)

# Optionally save results to a JSON file
if args.results_file:
    with open(args.results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
