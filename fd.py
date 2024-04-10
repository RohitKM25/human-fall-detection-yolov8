import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from ultralytics import YOLO
import cv2
import os
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = YOLO("fall_det_1.pt")


def detect_fall_image(image_file_path):
    frame = cv2.imread(image_file_path)
    results = model.track(frame, conf=0.6)
    return results[0].plot()


def track_fall_camera():
    cap = cv2.VideoCapture("videoplayback.mp4")

    while cap.isOpened():
        s, frame = cap.read()

        if not s:
            break
        results = model.track(frame, persist=True, conf=0.6)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# Step 1: Input video to frame conversion


def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Step 2: Skeleton sequence generation (using OpenPose or similar)

# Step 3: Feature extraction


def extract_features(frames):
    features = []
    for frame in frames:
        # Example: Extract features using color histogram
        hist = cv2.calcHist([frame], [0, 1, 2], None, [
                            8, 8, 8], [0, 256, 0, 256, 0, 256])
        features.append(hist.flatten())
    return np.array(features)

# Step 4: Classification


def classify(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test

# Step 5: Graph generation


def generate_graphs(y_pred, y_test):
    # Example: Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(y_test, y_pred)

    # Example: Generate accuracy graph
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Example: Generate ROC-AUC curve
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", roc_auc)

    # Example: Generate comparison graph with other techniques
    # ...

# Main function


labels = []
frames = []
for i in os.listdir("test/images"):
    img = f'test/images/{i}'
    frames.append(detect_fall_image(img))
    labels.append(re.search("-(\\d)[.]", i).group(1))
# Step 3: Feature extraction
features = extract_features(frames)
print(labels)
# # Dummy labels (replace with actual labels)
# labels = np.random.randint(0, 2, size=len(frames))
# Step 4: Classification
y_pred, y_test = classify(features, labels)
# Step 5: Graph generation
generate_graphs(y_pred, y_test)
