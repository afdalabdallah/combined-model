import subprocess
import cv2
import json
from pathlib import Path
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def detect(image_path, image_name):
    detection_command = f"python ./yolov7/detect.py --weights ./yolov7/best.pt --conf 0.1 --source {image_path} --name {image_name}"

    result = subprocess.run(detection_command, shell=True, text=True, capture_output=True)

def predict_and_display(image_path, model):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    CATEGORIES = ['Black','Blue','Brown','Gray', 'Green','Orange','Pink','Purple','Red','White','Yellow']
    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(predictions)
   
    # Load and display the image
    # img = image.load_img(image_path, target_size=(32, 32))
    # plt.imshow(img)
    # plt.axis('off')

    # # Display the predicted class
    # plt.title(f'Predicted Class: {CATEGORIES[predicted_class]}')
    # plt.show()
    return CATEGORIES[predicted_class]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array


def main():
    # Load the JSON data
    image_path = 'datasets/IMG20231214210246.jpg'
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    detect(image_path,image_name)
    json_path = f'./runs/detect/{image_name}/result.json'

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # # Read the image
    image_path = data['path']
    image = cv2.imread(image_path)
    highest_confidence_per_label = {}
    for prediction in data['prediction']:
        bounding_box = prediction['bounding_box']
        label = prediction['label']
        confidence = prediction['confidence']

        # Convert bounding box coordinates to integers
        bounding_box = [int(coord) for coord in bounding_box]
        if label not in highest_confidence_per_label or confidence > highest_confidence_per_label[label]['confidence']:
            # Update the highest confidence prediction for the label
            highest_confidence_per_label[label] = {
                'bounding_box': bounding_box,
                'label': label,
                'confidence': confidence
            }
       
    for label, highest_confidence_prediction in highest_confidence_per_label.items():
        bounding_box = highest_confidence_prediction['bounding_box']
        label = highest_confidence_prediction['label']
        confidence = highest_confidence_prediction['confidence']

        # Crop the region of interest (ROI) using the bounding box
        cropped_roi = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        # Save the cropped ROI to a file
        os.makedirs(f'./cropped/{image_name}/', exist_ok=True)
        save_path = f'./cropped/{image_name}/cropped_{label}_{confidence:.2f}.jpg'
        cv2.imwrite(save_path, cropped_roi)

    cropped_folder = f"./cropped/{image_name}"
    color_model = load_model('./color_classification_cnn_model.h5')
    result_prediction = []
    # for filename in os.listdir(cropped_folder):
    filename = 'cropped_body2_0.48.jpg'
    label_start = filename.find('cropped_') + len('cropped_')
    label_end = filename.find('_', label_start)
    label = filename[label_start:label_end]
    file_path = os.path.join(cropped_folder, filename)
    categories = predict_and_display(file_path, color_model)
    pred = {
        "label": label,
        "color": categories,
    }
    result_prediction.append(pred)
    print(result_prediction)
    return result_prediction


main()