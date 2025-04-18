from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Load image and run object detection
            img = cv2.imread(img_path)
            results = model(img)
            output_img = results[0].plot()

            # Save detected image
            result_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(result_path, output_img)

            return render_template("index.html", uploaded=True, img_path=result_path)

    return render_template("index.html", uploaded=False)

def generate_frames():
    """Capture webcam frames and perform object detection."""
    cap = cv2.VideoCapture(0)  # Change index if needed

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect objects
        results = model(frame)
        detected_frame = results[0].plot()

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    """Stream webcam video to the webpage."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
