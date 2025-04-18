import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pygame
from keras.models import load_model
import time
from PIL import Image
import io
import os

# Page config
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        border-radius: 0.5rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert {
        background-color: #ffebee;
        color: #c62828;
    }
    .safe {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("""
    This application uses computer vision and machine learning to detect driver drowsiness in real-time.
    It monitors your eye state and alerts you when signs of drowsiness are detected.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    sleep_threshold = st.slider("Drowsiness Threshold (frames)", 5, 30, 15)
    alarm_volume = st.slider("Alarm Volume", 0.0, 1.0, 0.7)
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Click 'Start Detection' to begin
    2. Position your face in front of the camera
    3. The system will monitor your eye state
    4. An alarm will sound if drowsiness is detected
    5. Click 'Stop' to end the session
    """)

# Initialize audio system
def init_audio():
    try:
        pygame.mixer.init()
        pygame.mixer.music.set_volume(alarm_volume)
        if os.path.exists("mixkit-facility-alarm-sound-999.wav"):
            pygame.mixer.music.load("mixkit-facility-alarm-sound-999.wav")
            return True
        else:
            st.error("Alarm sound file not found!")
            return False
    except Exception as e:
        st.error(f"Failed to initialize audio: {str(e)}")
        return False

# Play alarm with error handling
def play_alarm():
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    except Exception as e:
        st.error(f"Failed to play alarm: {str(e)}")

# Stop alarm with error handling
def stop_alarm():
    try:
        pygame.mixer.music.stop()
    except Exception as e:
        st.error(f"Failed to stop alarm: {str(e)}")

# Initialize components
@st.cache_resource
def load_components():
    model = load_model("eye_state_model1.h5")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
    return model, face_mesh

# Eye landmarks
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

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

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Feed")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.header("Statistics")
    stats_placeholder = st.empty()
    alert_placeholder = st.empty()

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'sleep_counter' not in st.session_state:
    st.session_state.sleep_counter = 0
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'drowsy_frames' not in st.session_state:
    st.session_state.drowsy_frames = 0
if 'audio_initialized' not in st.session_state:
    st.session_state.audio_initialized = False

# Control buttons
col3, col4 = st.columns(2)
with col3:
    if st.button("Start Detection", key="start"):
        st.session_state.running = True
        if not st.session_state.audio_initialized:
            st.session_state.audio_initialized = init_audio()
with col4:
    if st.button("Stop", key="stop"):
        st.session_state.running = False
        stop_alarm()

# Main detection loop
if st.session_state.running and st.session_state.audio_initialized:
    model, face_mesh = load_components()
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break

        status = "Awake"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = extract_eye(frame, landmarks, LEFT_EYE)
            right_eye = extract_eye(frame, landmarks, RIGHT_EYE)

            for eye_img in [left_eye, right_eye]:
                if eye_img.size == 0:
                    continue

                eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                eye_resized = cv2.resize(eye_gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0

                pred = model.predict(eye_resized, verbose=0)[0][0]
                label = "Open" if pred > 0.5 else "Closed"
                color = (0, 255, 0) if label == "Open" else (0, 0, 255)

                if label == "Closed":
                    st.session_state.sleep_counter += 1
                    st.session_state.drowsy_frames += 1
                else:
                    st.session_state.sleep_counter = 0

                cv2.putText(frame, label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                break

        st.session_state.total_frames += 1

        if st.session_state.sleep_counter >= sleep_threshold:
            status = "Drowsy!"
            play_alarm()
            cv2.putText(frame, "WAKE UP!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            stop_alarm()

        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update status
        status_class = "alert" if status == "Drowsy!" else "safe"
        status_placeholder.markdown(f"""
            <div class="status-box {status_class}">
                <h3>Current Status: {status}</h3>
                <p>Consecutive drowsy frames: {st.session_state.sleep_counter}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Update statistics
        drowsiness_percentage = (st.session_state.drowsy_frames / st.session_state.total_frames * 100) if st.session_state.total_frames > 0 else 0
        stats_placeholder.markdown(f"""
            <div class="status-box safe">
                <h3>Statistics</h3>
                <p>Total Frames: {st.session_state.total_frames}</p>
                <p>Drowsy Frames: {st.session_state.drowsy_frames}</p>
                <p>Drowsiness Percentage: {drowsiness_percentage:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.1)  # Add small delay to prevent overwhelming the system

    cap.release()
    pygame.mixer.quit() 