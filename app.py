# pip install opencv-python-headless torch torchvision ultralytics

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load a more powerful pre-trained gender classification model
gender_model = models.efficientnet_b0(pretrained=True)
num_ftrs = gender_model.classifier[1].in_features
gender_model.classifier[1] = nn.Linear(num_ftrs, 2)  # Assuming binary classification for gender
gender_model.eval()

# Load a pre-trained age estimation model
age_model = models.resnet18(pretrained=True)
num_ftrs = age_model.fc.in_features
age_model.fc = nn.Linear(num_ftrs, 101)  # Assuming age prediction in the range [0, 100]
age_model.eval()

# Transformation for models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Gender classes
gender_classes = ['Male', 'Female']

def detect_and_predict_gender_and_age(frame):
    # Perform object detection
    results = model(frame)

    # Extract bounding boxes for person class
    bboxes = results.xyxy[0].cpu().numpy()  # xyxy format: (x1, y1, x2, y2, confidence, class)
    bboxes = [bbox for bbox in bboxes if bbox[5] == 0]  # Class 0 is for person

    predictions = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Crop the detected person
        person = frame[y1:y2, x1:x2]

        # Convert person crop to PIL image
        person_pil = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))

        # Apply data augmentation during inference
        augmented_imgs = [transform(person_pil).unsqueeze(0)]
        for _ in range(4):
            augmented_img = transform(torchvision.transforms.functional.hflip(person_pil)).unsqueeze(0)
            augmented_imgs.append(augmented_img)
        
        # Predict gender and age with augmented images
        all_gender_preds = []
        all_age_preds = []
        with torch.no_grad():
            for img in augmented_imgs:
                gender_pred = gender_model(img)
                age_pred = age_model(img)
                all_gender_preds.append(gender_pred)
                all_age_preds.append(age_pred)
        
        # Average the predictions
        avg_gender_pred = torch.mean(torch.stack(all_gender_preds), dim=0)
        avg_age_pred = torch.mean(torch.stack(all_age_preds), dim=0)
        
        # Gender prediction
        gender_confidence = torch.nn.functional.softmax(avg_gender_pred, dim=1)[0]
        gender_idx = torch.argmax(avg_gender_pred, dim=1).item()
        gender = gender_classes[gender_idx]
        gender_confidence_val = gender_confidence[gender_idx].item()

        # Age prediction
        age = torch.argmax(avg_age_pred, dim=1).item()
        age_confidence_val = torch.nn.functional.softmax(avg_age_pred, dim=1)[0][age].item()
        
        predictions.append((x1, y1, x2, y2, gender, gender_confidence_val, age, age_confidence_val))

    return predictions

def process_video(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons and predict genders and ages
        predictions = detect_and_predict_gender_and_age(frame)

        # Draw bounding boxes and labels on the frame
        for (x1, y1, x2, y2, gender, gender_confidence, age, age_confidence) in predictions:
            label = f"{gender}: {gender_confidence:.2f}, Age: {age} ({age_confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the frame
        cv2.imshow('Gender and Age Estimation', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    # Load image
    frame = cv2.imread(image_path)

    # Detect persons and predict genders and ages
    predictions = detect_and_predict_gender_and_age(frame)

    # Draw bounding boxes and labels on the frame
    for (x1, y1, x2, y2, gender, gender_confidence, age, age_confidence) in predictions:
        label = f"{gender}: {gender_confidence:.2f}, Age: {age} ({age_confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Get the dimensions of the screen
    screen_res = 1280, 720
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)

    # Resize image
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)

    # Center the window on the screen
    cv2.namedWindow('Gender and Age Estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gender and Age Estimation', window_width, window_height)
    screen_width, screen_height = screen_res
    cv2.moveWindow('Gender and Age Estimation', (screen_width - window_width) // 2, (screen_height - window_height) // 2)

    # Display the frame
    cv2.imshow('Gender and Age Estimation', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    choice = input("Enter 'video' to process a video or 'image' to process an image: ").strip().lower()
    file_path = input("Enter the file path: ").strip()

    if choice == 'video':
        process_video(file_path)
    elif choice == 'image':
        process_image(file_path)
    else:
        print("Invalid choice. Please enter 'video' or 'image'.")

if __name__ == "__main__":
    main()
