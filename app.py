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
import keyboard
from datetime import datetime
import shutil

# Create a directory for storing songs if it doesn't exist
SONGS_DIR = "saved_songs"
if not os.path.exists(SONGS_DIR):
    os.makedirs(SONGS_DIR)

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
    .emergency {
        background-color: #ffebee;
        color: #c62828;
        animation: blink 1s infinite;
    }
    .drowsiness-level {
        height: 20px;
        background: linear-gradient(to right, #4CAF50, #FFC107, #F44336);
        border-radius: 10px;
        margin: 10px 0;
        position: relative;
    }
    .drowsiness-indicator {
        width: 20px;
        height: 20px;
        background-color: white;
        border: 2px solid #333;
        border-radius: 50%;
        position: absolute;
        top: -5px;
        transform: translateX(-50%);
        transition: left 0.3s ease;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("""
    This application uses computer vision and machine learning to detect driver drowsiness in real-time.
    It monitors your eye state and alerts you when signs of drowsiness are detected.
""")

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
if 'emergency_triggered' not in st.session_state:
    st.session_state.emergency_triggered = False
if 'song_playing' not in st.session_state:
    st.session_state.song_playing = False
if 'drowsiness_level' not in st.session_state:
    st.session_state.drowsiness_level = 0
if 'last_eye_state' not in st.session_state:
    st.session_state.last_eye_state = "Open"
if 'first_sleep_detected' not in st.session_state:
    st.session_state.first_sleep_detected = False
if 'initial_detection' not in st.session_state:
    st.session_state.initial_detection = True
if 'closed_eye_frames' not in st.session_state:
    st.session_state.closed_eye_frames = 0
if 'blink_frames' not in st.session_state:
    st.session_state.blink_frames = 0

# Sidebar
with st.sidebar:
    st.header("Settings")
    sleep_threshold = st.slider("Drowsiness Threshold (frames)", 10, 30, 20)  # Increased minimum
    song_threshold = st.slider("Song Alert Threshold (frames)", 8, 15, 10)  # Increased for blink tolerance
    blink_threshold = st.slider("Blink Threshold (frames)", 1, 5, 3)  # New threshold for blinks
    alarm_volume = st.slider("Alarm Volume", 0.0, 1.0, 0.7)
    song_volume = st.slider("Song Volume", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    st.markdown("### Upload Favorite Song")
    uploaded_file = st.file_uploader("Choose a song file (MP3/WAV)", type=['mp3', 'wav'])
    
    # Save uploaded song permanently
    if uploaded_file is not None:
        song_path = os.path.join(SONGS_DIR, uploaded_file.name)
        if not os.path.exists(song_path):
            with open(song_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Song saved successfully: {uploaded_file.name}")
    
    # Display saved songs
    st.markdown("### Saved Songs")
    saved_songs = [f for f in os.listdir(SONGS_DIR) if f.endswith(('.mp3', '.wav'))]
    if saved_songs:
        selected_song = st.selectbox("Select a saved song", saved_songs)
        if selected_song:
            st.session_state.selected_song = os.path.join(SONGS_DIR, selected_song)
    else:
        st.info("No saved songs found. Please upload a song.")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload your favorite song (MP3/WAV)
    2. Click 'Start Detection' to begin
    3. Position your face in front of the camera
    4. The system will monitor your eye state
    5. Press 'E' key for emergency simulation
    6. Click 'Stop' to end the session
    """)

# Initialize audio system
def init_audio():
    try:
        # Check if running on Streamlit Cloud
        is_streamlit_cloud = os.environ.get('IS_STREAMLIT_CLOUD', False)
        
        if is_streamlit_cloud:
            st.warning("⚠️ Audio alerts are disabled in cloud deployment. For full functionality including audio alerts, please run the application locally.")
            return True
        
        pygame.mixer.init()
        pygame.mixer.music.set_volume(alarm_volume)
        if os.path.exists("mixkit-facility-alarm-sound-999.wav"):
            pygame.mixer.music.load("mixkit-facility-alarm-sound-999.wav")
            return True
        else:
            st.error("Alarm sound file not found!")
            return False
    except Exception as e:
        st.warning(f"⚠️ Audio system initialization failed: {str(e)}\nAudio alerts will be disabled.")
        return True  # Return True to allow the app to run without audio

# Play alarm with error handling
def play_alarm():
    try:
        if os.environ.get('IS_STREAMLIT_CLOUD', False):
            return
        
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.music.set_volume(alarm_volume)
            pygame.mixer.music.load("mixkit-facility-alarm-sound-999.wav")
            pygame.mixer.music.play(-1)
    except Exception as e:
        st.warning("⚠️ Unable to play alarm sound")

# Play favorite song
def play_favorite_song():
    try:
        if os.environ.get('IS_STREAMLIT_CLOUD', False):
            return
            
        if 'selected_song' in st.session_state and os.path.exists(st.session_state.selected_song):
            pygame.mixer.music.stop()
            pygame.mixer.music.set_volume(song_volume)
            pygame.mixer.music.load(st.session_state.selected_song)
            pygame.mixer.music.play(-1)
            return True
    except Exception as e:
        st.warning("⚠️ Unable to play song")
    return False

# Stop alarm with error handling
def stop_alarm():
    try:
        if os.environ.get('IS_STREAMLIT_CLOUD', False):
            return
            
        pygame.mixer.music.stop()
    except Exception as e:
        pass

# Calculate drowsiness level (0-100)
def calculate_drowsiness_level(sleep_counter, total_frames):
    if total_frames == 0:
        return 0
    base_level = (sleep_counter / sleep_threshold) * 100
    return min(100, max(0, base_level))

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
    drowsiness_level_placeholder = st.empty()

with col2:
    st.header("Statistics")
    stats_placeholder = st.empty()
    alert_placeholder = st.empty()
    emergency_placeholder = st.empty()

col3, col4 = st.columns(2)
with col3:
    if st.button("Start Detection", key="start"):
        st.session_state.running = True
        st.session_state.initial_detection = True
        st.session_state.drowsiness_level = 0
        st.session_state.closed_eye_frames = 0
        st.session_state.sleep_counter = 0
        st.session_state.first_sleep_detected = False
        if not st.session_state.audio_initialized:
            st.session_state.audio_initialized = init_audio()
with col4:
    if st.button("Stop", key="stop"):
        st.session_state.running = False
        stop_alarm()

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

                # Update eye state counters with blink detection
                if label == "Closed":
                    st.session_state.closed_eye_frames += 1
                    st.session_state.blink_frames += 1
                    if st.session_state.closed_eye_frames > blink_threshold:  # Only count as drowsy after blink threshold
                        st.session_state.drowsy_frames += 1
                else:
                    # Reset counters if eyes were closed for less than blink threshold
                    if st.session_state.blink_frames <= blink_threshold:
                        st.session_state.drowsy_frames = max(0, st.session_state.drowsy_frames - st.session_state.blink_frames)
                    st.session_state.closed_eye_frames = 0
                    st.session_state.blink_frames = 0

                # Update drowsiness level based on continuous closed eye frames
                if st.session_state.closed_eye_frames >= song_threshold and not st.session_state.first_sleep_detected:
                    st.session_state.drowsiness_level = min(2, st.session_state.drowsiness_level + 1)
                    st.session_state.first_sleep_detected = True
                    st.session_state.initial_detection = False
                elif label == "Open":
                    st.session_state.first_sleep_detected = False
                
                st.session_state.last_eye_state = label
                
                # Update display with more detailed information
                cv2.putText(frame, f"Eye State: {label}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"Closed Frames: {st.session_state.closed_eye_frames}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                if st.session_state.blink_frames > 0 and st.session_state.blink_frames <= blink_threshold:
                    cv2.putText(frame, "Blink Detected", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 2)
                break

        st.session_state.total_frames += 1

        if keyboard.is_pressed('e') and not st.session_state.emergency_triggered:
            st.session_state.emergency_triggered = True
            st.session_state.drowsiness_level = 3
            stop_alarm()

        # Three-level safety response based on drowsiness level
        if st.session_state.emergency_triggered:
            # Level 3: Emergency (Accident Simulation)
            status = "EMERGENCY!"
            cv2.putText(frame, "ACCIDENT DETECTED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif st.session_state.drowsiness_level >= 2:
            # Level 2: Alarm Warning
            status = "Drowsy!"
            if not pygame.mixer.music.get_busy() or st.session_state.song_playing:
                st.session_state.song_playing = False
                play_alarm()
            cv2.putText(frame, "WAKE UP!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif st.session_state.drowsiness_level >= 1 and not st.session_state.initial_detection:
            # Level 1: Song Engagement (only after initial drowsiness detection)
            status = "Slight Drowsiness"
            if 'selected_song' in st.session_state and os.path.exists(st.session_state.selected_song):
                if not pygame.mixer.music.get_busy():
                    st.session_state.song_playing = True
                    play_favorite_song()
                elif st.session_state.closed_eye_frames >= song_threshold and not st.session_state.first_sleep_detected:
                    # Only increment to Level 2 if sustained eye closure
                    st.session_state.drowsiness_level = 2
                    st.session_state.song_playing = False
                    play_alarm()
                cv2.putText(frame, "Stay Alert!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        else:
            # Normal State
            status = "Awake"
            stop_alarm()
            st.session_state.song_playing = False

        # Display current level and status
        level_text = {
            0: "Normal",
            1: "Level 1: Song Alert",
            2: "Level 2: Alarm Warning",
            3: "Level 3: Emergency"
        }
        cv2.putText(frame, f"Level: {level_text[st.session_state.drowsiness_level]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update status with level information
        status_class = "alert" if status in ["Drowsy!", "EMERGENCY!"] else "safe"
        status_placeholder.markdown(f"""
            <div class="status-box {status_class}">
                <h3>Current Level: {level_text[st.session_state.drowsiness_level]}</h3>
                <p>Status: {status}</p>
                <p>Drowsiness Counter: {st.session_state.drowsiness_level}</p>
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

        # Emergency alert
        if st.session_state.emergency_triggered:
            emergency_placeholder.markdown(f"""
                <div class="status-box emergency">
                    <h3>🚨 Emergency Detected!</h3>
                    <p>Location shared with nearby emergency services</p>
                    <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """, unsafe_allow_html=True)
        
        time.sleep(0.1)

    cap.release()
    pygame.mixer.quit()

# Add a notice about audio limitations in cloud deployment
st.markdown("""
<div style='background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
    <h4 style='color: #856404; margin: 0;'>⚠️ Cloud Deployment Notice</h4>
    <p style='color: #856404; margin: 0.5rem 0 0 0;'>
        This is the cloud-deployed version of the application. Audio alerts are disabled in this environment.
        For full functionality including audio alerts, please run the application locally following the instructions in the
        <a href='https://github.com/bskrishna2006/Driver-drowsiness-streamlit' target='_blank'>GitHub repository</a>.
    </p>
</div>
""", unsafe_allow_html=True) 