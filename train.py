import argparse
import os
import time
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import applications as tf_models
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Setting up parser
parser = argparse.ArgumentParser(description='TensorFlow ImageNet Training')
parser.add_argument('--model', default='ResNet50', type=str, help='Model name to train')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
parser.add_argument('--epochs', default=90, type=int, help='Number of epochs to train for')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--output', default='./output', type=str, help='Output path for checkpoints')
parser.add_argument('--data-dir', default=None, type=str, help='Directory of the dataset')
parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')

# Parse arguments
args = parser.parse_args()

# Configure automatic mixed precision
if args.use_amp:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

# Load model
model = getattr(tf_models, args.model)(weights=None, classes=1000)
optimizer = optimizers.SGD(lr=args.lr, momentum=0.9)
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Data loading
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

def load_dataset(split):
    dataset, info = tfds.load('imagenet2012', with_info=True, data_dir=args.data_dir, split=split)
    return dataset

train_dataset = load_dataset('train').map(preprocess_image).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset('validation').map(preprocess_image).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

# Callbacks
checkpoint_dir = os.path.join(args.output, "checkpoints")
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'ckpt-{epoch:02d}'),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

log_dir = os.path.join(args.output, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    elif epoch < 60:
        return lr * 0.1
    else:
        return lr * 0.01

lr_callback = LearningRateScheduler(scheduler)

# Training
model.fit(
    train_dataset,
    epochs=args.epochs,
    validation_data=val_dataset,
    callbacks=[checkpoint_callback, tensorboard_callback, lr_callback]
)
