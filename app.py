import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from flask import Flask, render_template, Response, jsonify
import base64

app = Flask(__name__)

# Global variable to track drowsiness status for the frontend
drowsy_state = False

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye landmark indices from MediaPipe's 478-point face mesh
# These 6 points per eye map to the same EAR calculation as dlib's 68-point model
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25
# CONSECUTIVE_FRAMES = 20

def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    # Decode base64 image from browser
    header, encoded = data.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    drowsy = False
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks.landmark
        
        def get_coords(indices):
            return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

        left_ear = calculate_ear(get_coords(LEFT_EYE))
        right_ear = calculate_ear(get_coords(RIGHT_EYE))
        
        if (left_ear + right_ear) / 2.0 < EAR_THRESHOLD:
            drowsy = True

    return jsonify({"drowsy": drowsy})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)