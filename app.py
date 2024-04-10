from ultralytics import YOLO
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = YOLO("fall_det_1.pt")

cap = cv2.VideoCapture("videoplayback.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture frame")
        break
    results = model.track(frame, persist=True, conf=0.5)
    annotated_frame = results[0].plot()
    cv2.imshow("Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
