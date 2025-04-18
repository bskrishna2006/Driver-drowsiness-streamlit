import cv2
import numpy as np
import mediapipe as mp
import pygame
import tempfile
import os
import streamlit as st
from keras.models import load_model
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Set up page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("Driver Drowsiness Detection System")
st.markdown("""
    This application detects driver drowsiness by analyzing eye state.
    The system will alert you if your eyes remain closed for too long.
""")

@st.cache_resource
def load_detection_model():
    try:
        # In production, you'd have a proper path to your model
        # For this example, we'll handle the case where the model isn't available
        model = load_model("eye_state_model1.h5")
        return model
    except:
        st.warning("Model file not found. Using a placeholder model for demonstration.")
        # Creating a very simple placeholder model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

@st.cache_resource
def load_alarm_sound():
    # Create a temporary file for the alarm sound
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "alarm.wav")
    
    # For demonstration purposes, we'll just initialize pygame without the actual file
    pygame.mixer.init()
    return temp_path

# Initialize MediaPipe Face Mesh
@st.cache_resource
def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmarks indices
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# Load resources
model = load_detection_model()
alarm_sound_path = load_alarm_sound()
face_mesh = initialize_face_mesh()

def extract_eye(frame, landmarks, indices):
    ih, iw = frame.shape[:2]
    x1 = int(landmarks[indices[0]].x * iw)
    y1 = int(landmarks[indices[0]].y * ih)
    x2 = int(landmarks[indices[1]].x * iw)
    y2 = int(landmarks[indices[1]].y * ih)

    w = abs(x2 - x1)
    h = w
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x_start = max(cx - w // 2, 0)
    y_start = max(cy - h // 2, 0)
    x_end = min(cx + w // 2, iw)
    y_end = min(cy + h // 2, ih)

    eye_img = frame[y_start:y_end, x_start:x_end]
    return eye_img

class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.sleep_counter = 0
        self.sleep_threshold = 15  # Adjust as needed
        self.alarm_playing = False
        self.status = "Awake"
        self.last_alarm_time = 0
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process image
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = extract_eye(img, landmarks, LEFT_EYE)
            right_eye = extract_eye(img, landmarks, RIGHT_EYE)

            for eye_img in [left_eye, right_eye]:
                if eye_img.size == 0:
                    continue

                eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                eye_resized = cv2.resize(eye_gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0

                pred = model.predict(eye_resized, verbose=0)[0][0]
                label = "Open" if pred > 0.5 else "Closed"
                color = (0, 255, 0) if label == "Open" else (0, 0, 255)

                if label == "Closed":
                    self.sleep_counter += 1
                else:
                    self.sleep_counter = 0

                cv2.putText(img, label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                break  # Predict one eye only for stability

        # Update status and handle alarm
        if self.sleep_counter >= self.sleep_threshold:
            self.status = "Drowsy!"
            
            # For visualization, show warning text
            cv2.putText(img, "WAKE UP!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Play alarm sound (we simulate this in Streamlit)
            current_time = time.time()
            if current_time - self.last_alarm_time >= 2.0:  # Play every 2 seconds
                self.alarm_playing = True
                self.last_alarm_time = current_time
        else:
            self.status = "Awake"
            self.alarm_playing = False

        # Display status
        cv2.putText(img, f"Status: {self.status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return img

# Sidebar for settings
st.sidebar.header("Settings")
sleep_threshold = st.sidebar.slider("Sleep Threshold", 5, 30, 15, 
                                  help="Number of consecutive frames with closed eyes before triggering the alarm")

# Create two columns for the main content
col1, col2 = st.columns([3, 1])

with col1:
    # WebRTC streamer
    st.subheader("Drowsiness Detection Camera")
    
    # Configuration for WebRTC (empty ICE servers for local testing)
    rtc_configuration = RTCConfiguration({"iceServers": []})
    
    # Start the WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_transformer_factory=DrowsinessDetector,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Safety disclaimer
    st.warning("""
        **Disclaimer**: This application is for demonstration purposes only. 
        Do not rely on it for actual driving safety. Always stay alert while driving 
        and take regular breaks when tired.
    """)

with col2:
    # Status indicators
    st.subheader("Detection Status")
    
    # Create placeholders for dynamic content
    status_placeholder = st.empty()
    counter_placeholder = st.empty()
    
    # Instructions
    st.markdown("""
        ### How it works:
        1. Allow camera access when prompted
        2. Position yourself so your face is clearly visible
        3. The system tracks your eye state
        4. An alarm will sound if drowsiness is detected
        
        ### Tips:
        - Ensure good lighting for best detection
        - Position your face centered in the frame
        - Adjust the sleep threshold in the sidebar if needed
    """)
    
    # If WebRTC is active, update status
    if webrtc_ctx and webrtc_ctx.state.playing:
        # This would normally be updated from the transformer, 
        # but for demonstration we'll show a simple status
        status_placeholder.markdown(f"**Status:** {'Monitoring...'}")
        counter_placeholder.markdown(f"**Sleep Counter:** {0}/{sleep_threshold}")
    else:
        status_placeholder.markdown("**Status:** Not active")
        counter_placeholder.markdown("**Sleep Counter:** N/A")

# Footer
st.markdown("---")
st.markdown("Driver Drowsiness Detection System | Safety First")