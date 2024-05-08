import os
import numpy as np
import cv2
from keras.models import load_model

# Define image dimensions and number of classes
img_width, img_height = 48, 48

# Function to predict emotion from an image
def predict_emotion(test_model, image_path, emotions):  # Pass emotions dictionary as argument
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image to a specific width and height
    img = cv2.resize(img, (img_width, img_height))
    # Expand the dimensions of the image array
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = test_model.predict(img)
    print(f"Prediction is {prediction}")
    emotion_index = np.argmax(prediction)
    predicted_emotion = emotions[emotion_index]
    return predicted_emotion

if __name__ == "__main__":
    model_file = '/home/bharat/codebase/model_save/model.h5'
    if not os.path.exists(model_file):
        print("Model file not found. Please train the model first.")
    else:
        emotions = ["sad", "happy"]  # Define emotions
        test_model = load_model(model_file)
        while True:
            image_path = input("Enter the path to the Test Image (enter 'q' to quit): ")
            if image_path.lower() == 'q':
                break
            predicted_emotion = predict_emotion(test_model, image_path, emotions)
            print("Predicted Emotion:", predicted_emotion)






