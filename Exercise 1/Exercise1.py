# Set up Python environment
# !pip install -q tensorflow==2.0.0-alpha0

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

# Load the data

import random
import csv
from sklearn.model_selection import train_test_split
from collections import Counter
import itertools

# Define constants

# How many neighhbors to consider when predicting a node's label
# The fact that we fix this is quite interesting, more discussion
# of this later.
FIXED_NEIGHBOR_SIZE = 10

# The batch size we'll bucket our data into before running the machine
# learning functions
BATCH_SIZE = 32


def label_to_int(label):
    '''
    Translates binary tokens into int
    '''
    if label == 'left-leaning':
        return 0
    else:
        return 1


def label_to_vec(label):
    '''
    Translates binary tokens into one-hot vector
    '''
    if label == 'left-leaning':
        return [1,0]
    else:
        return [0,1]


def load_data():
    '''
    Transforms data into the format needed for our training
    '''

    edges = []

    with open('./data/out.moreno_blogs_blogs') as f:
        reader = csv.reader(f, delimiter=' ')
        for i in reader:
            if i[0] != "%":
                edges.append( (int(i[0])-1, int(i[1])-1) )

    labels = []
    with open('./data/ent.moreno_blogs_blogs.blog.orientation') as f:
        reader = csv.reader(f, delimiter=' ')
        for i in reader:
            labels.append(i[0])

    # Dataset for test and training
    # (where L is a left/right label)

    # X are inputs: [L, L, L, L] list of neighbor labels
    X = []

    # y are labels
    y = []

    for (node_id, label) in enumerate(labels):
        neighbors = set()
        for (v1, v2) in edges:
            if v1 == node_id:
                neighbors.add(v2)
            if v2 == node_id:
                neighbors.add(v1)

        try:
            neighbors.remove(node_id)

        except:
            # It's fine, we're just guarding against self-reference that would make this
            # exercise a bit easier
            pass

        neighbor_labels = [label_to_vec(labels[i]) for i in neighbors]
        random.shuffle(neighbor_labels)

        X.append(neighbor_labels)
        y.append(label_to_int(label))

    return X, y


X, y = load_data()

print("Data statistics")
print("Number of data-points", len(X))
print("Average number of neighbors", np.average([len(i) for i in X]))
print("Max number of neighbors", np.max([len(i) for i in X]))
print("Min number of neighbors", np.min([len(i) for i in X]))
print("Distribution of labels", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print()
print("Number of training examples", len(X_train))
print("Distribution of labels", Counter(y_train))
print("Example: ", X_train[90], " has label ", y_test[90])

print()
print("Number of test examples", len(X_test))
print("Distribution of labels", Counter(y_test))
print("Example: ", X_test[0], " has label ", y_test[0])


# Training Data

def fix_size_of_list(data, target_len=FIXED_NEIGHBOR_SIZE):
    '''
      This function highlights one of the central challenges of graph data:
      it is naturally variable sized and frameworks like TensorFlow want
      fixed sized tensor data.

      Our simplistic solution is to fix the size - we chop it down if too large, or
      zero pad it if too small.
    '''

    delta = len(data) - target_len

    if delta >= 0:
        return data[0:target_len]
    else:
        return np.pad(data, [(0, -delta), (0,0)], mode='constant', constant_values=0)

# Create Tensorflow dataset objects ready for training and evaluation

## Training data
X_train_fixed = [fix_size_of_list(i) for i in X_train]

dataset_train = tf.data.Dataset.from_tensor_slices((X_train_fixed, y_train))
dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_train = dataset_train.shuffle(BATCH_SIZE * 10)

# Test data
X_test_fixed = [fix_size_of_list(i) for i in X_test]
dataset_test = tf.data.Dataset.from_tensor_slices((X_test_fixed, y_test))
dataset_test = dataset_test.batch(BATCH_SIZE)


# Model

model = keras.Sequential([
    layers.Input(shape=[FIXED_NEIGHBOR_SIZE, 2]),
    layers.Flatten(),
    # Please add a Dense layer here of width 2
    layers.Dense(10, input_shape=[10,]),
    layers.Dense(10, input_shape=[10,]),
    layers.Softmax()
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training:")
model.fit(dataset_train, epochs=13, verbose=1)

print("\n\nFinal test accuracy:")

results = model.evaluate(dataset_test)

for l, v in zip(model.metrics_names, results):
    print(l, v)

# Experiment log
# Default accuracy 0.9728
# Increased FIXED_NEIGHBOR_SIZE to 27 (mean value), and accuracy pass to 0.9673 (no significantly change)
# Decreased FIXED_NEIGHBOR_SIZE to 5, and accuracy = 0.9402174 and loss = 0.26878709346055984 (increased loss)
# Add one more dense layer: decreased loss to 0.12