# interact with os, working with files & directories, executing sys command 
import os

# provides support for large, multidimensional arrays and matrics, along with may math functions
# NumPy is used in this code to efficiently store and manipulate image data, making it easier to perform tasks such as preprocessing and analysis
import numpy as np

# Computer vision library, providing functions for image and video processing, including image manipulation, feature detection, object detection
import cv2

# keras is a high neural n/w API, it is used for building, training, evaluating & deep learning model
# keras.preprocessing.image module, which provides various utilities for image preprocessing tasks
from keras.preprocessing.image import img_to_array

# to_categorical function from keras.utils is used for one-hot encoding categorical labels. 
# One-hot encoding is a technique used to represent categorical data where each category is represented as a binary vector.
from keras.utils import to_categorical

# sklearn - Machine learning library, provides simple and efficient tools for data mining and data analysis
# LabelEncoder from 'sklearn.preprocessing' is used for encoding categorical labels into numerical labels. It assigns a unique integer to each category in the dataset
from sklearn.preprocessing import LabelEncoder

# 'train_test_split' function from sklearn.model_selection is used for splitting datasets into two subsets: one for training a machine learning model and the other for testing its performance.
from sklearn.model_selection import train_test_split

# it serves as a linear stack of layers.
# It's a fundamental building block used to create neural networks, especially for feedforward architectures where data flows sequentially through layers from input to output.
from keras.models import Sequential

#Conv2D: This layer creates a convolutional kernel that is convolved with the layer input to produce a tensor of outputs.(eg. parameter include no. of filters/kernels)
#MaxPooling2D: This layer performs max pooling operation for spatial data. (eg. reduce dimensions)
#Flatten: This layer flattens the input tensor into a 1D array. It's typically used to transition from convolutional/pooling layers to fully connected layers.
#Dense: This layer is a regular densely-connected neural network layer. It implements the operation: output = activation(dot(input, kernel) + bias).
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths to your dataset
train_dir = "dataset1/train"
test_dir = "dataset1/test"

# Define image dimensions and number of classes
img_width, img_height = 48, 48
num_classes = 2  # happy and sad

# Function to load and preprocess images
def load_images(directory): 
    images = []
    labels = []
    # itrate through each category (subdirectory) in the specified directory
    for category in os.listdir(directory): 
        # Construct the full path of the current category
        category_path = os.path.join(directory, category)
        # print(category_path)
        
        # Iterate through each image file in the current category 
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            # print(img_path)
            
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_width, img_height))
            image = img_to_array(image) # 'img_to_array' function specifically converts images from the Mat format, which is used by OpenCV, to a NumPy array.
            images.append(image)
            labels.append(category)
    # Convert the lists of images and labels into NumPy arrays and return them        
    return np.array(images), np.array(labels)

# Load and preprocess train and test images
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)

# Encode labels into integers eg.Encoded labels: [1 2 0 1 2]
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert labels to one-hot encoding eg. [[0. 1. 0.] [0. 0. 1.] [1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
train_labels_onehot = to_categorical(train_labels_encoded, num_classes)
test_labels_onehot = to_categorical(test_labels_encoded, num_classes)

# Split train set into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels_onehot, test_size=0.2)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Convert test labels to integers
test_labels = label_encoder.transform(test_labels)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, to_categorical(test_labels, num_classes))
print("Test Accuracy:", test_accuracy)


# Function to predict emotion from an image
def predict_emotion(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
     # Resize the image to a specific width and height
    img = cv2.resize(img, (img_width, img_height))
    # Expand the dimensions of the image array
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    prediction = model.predict(img)
    
    # Determine the predicted emotion based on the model prediction
    #The 'np.argmax()' function returns the index of the maximum value in the prediction array
    emotion = "happy" if np.argmax(prediction) == 0 else "sad"
    return emotion

# Example usage:
image_path = "myTest/test4.jpg"
predicted_emotion = predict_emotion(image_path)
print("Predicted Emotion:", predicted_emotion)


