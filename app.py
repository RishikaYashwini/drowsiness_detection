import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from flask import Flask, render_template, Response, jsonify
import winsound

app = Flask(__name__)

# Global variable to track drowsiness status for the frontend
drowsy_state = False

# MediaPipe FaceLandmarker setup (new tasks API)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Eye landmark indices from MediaPipe's 478-point face mesh
# These 6 points per eye map to the same EAR calculation as dlib's 68-point model
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20


def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_coords(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status')
def status():
    """Endpoint for the browser to check drowsiness status"""
    global drowsy_state
    return jsonify({"drowsy": drowsy_state})

def generate_frames():
    global drowsy_state
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    frame_counter = 0
    timestamp_ms = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms += 33  # ~30fps
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]

                left_eye = get_eye_coords(landmarks, LEFT_EYE, w, h)
                right_eye = get_eye_coords(landmarks, RIGHT_EYE, w, h)

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        drowsy_state = True
                        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        drowsy_state = False
                else:
                    frame_counter = 0
                    drowsy_state = False
            else:
                drowsy_state = False
 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # Get port from environment variable, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
