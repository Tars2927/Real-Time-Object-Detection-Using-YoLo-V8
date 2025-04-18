from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    """Detect objects in a given frame using YOLOv8."""
    results = model(frame)
    return results[0].plot()

def start_webcam():
    """Start real-time object detection using webcam."""
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame = detect_objects(frame)
        cv2.imshow("Real-Time Object Detection", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam()
