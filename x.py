import os
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define image dimensions and number of classes
img_width, img_height = 48, 48
num_classes = 2  # happy and sad

# Function to load and preprocess images
def load_images(directory, labels): 
    images = []
    encoded_labels = []
    # iterate through each category (subdirectory) in the specified directory
    for category in os.listdir(directory):
        # Check if the category is a directory
        if os.path.isdir(os.path.join(directory, category)):
            # Get the corresponding label for the category
            label = labels.get(category)
            if label is not None:
                # Construct the full path of the current category
                category_path = os.path.join(directory, category)

                # Iterate through each image file in the current category 
                for img in os.listdir(category_path):
                    img_path = os.path.join(category_path, img)

                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (img_width, img_height))
                    image = img_to_array(image)
                    images.append(image)
                    encoded_labels.append(label)
    return np.array(images), np.array(encoded_labels)

# Function to create and train the model
def train_model(dataset_train, dataset_test, labels):  # Pass labels dictionary as argument
    # Load and preprocess train and test images
    train_images, train_labels = load_images(dataset_train, labels)
    test_images, test_labels = load_images(dataset_test, labels)

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

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_onehot)
    print("Test Accuracy:", test_accuracy)

    return model

# Function to predict emotion from an image
def predict_emotion(model, image_path, emotions):  # Pass emotions dictionary as argument
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image to a specific width and height
    img = cv2.resize(img, (img_width, img_height))
    # Expand the dimensions of the image array
    img = np.expand_dims(img, axis=0)
    img = img / 255.0 
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    predicted_emotion = emotions[emotion_index]
    return predicted_emotion

# eg
def main():
    dataset_train = input("Enter the path to the training dataset: ")
    dataset_test = input("Enter the path to the test dataset: ")

    # Ask for label names and emotions
    labels = {}
    emotions = []
    for directory in ["d1", "d2"]:
        label_name = input(f"Please enter the label name for directory '{directory}': ")
        emotion_value = input(f"Please enter the emotion value for '{label_name}': ")
        labels[directory] = label_name
        emotions.append(emotion_value)

    model = train_model(dataset_train, dataset_test, labels)
    image_path = input("Enter the path to the Test Image: ")
    predicted_emotion = predict_emotion(model, image_path, emotions)
    print("Predicted Emotion:", predicted_emotion)

#This is the entry point of the script, where the execution begins
if __name__ == "__main__":
    main()
