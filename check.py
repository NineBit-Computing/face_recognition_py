import os
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
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
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_width, img_height))
            image = img_to_array(image)
            images.append(image)
            labels.append(category)
    return np.array(images), np.array(labels)

# Load and preprocess train and test images
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)

# Encode labels into integers
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert labels to one-hot encoding
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
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    prediction = model.predict(img)
    emotion = "happy" if np.argmax(prediction) == 0 else "sad"
    return emotion

# Example usage:
image_path = "myTest/test4.jpg"
predicted_emotion = predict_emotion(image_path)
print("Predicted Emotion:", predicted_emotion)


