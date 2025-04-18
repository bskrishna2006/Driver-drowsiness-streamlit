# Driver Drowsiness Detection System

A web-based application that detects driver drowsiness using computer vision and deep learning.

## Features

- Real-time drowsiness detection using webcam or video file
- Eye state classification (Open/Closed)
- Audio and visual alerts for drowsiness
- Adjustable sleep threshold
- Support for both webcam and video file input

## Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Choose your input source:
   - Webcam: For real-time detection using your camera
   - Video File: Upload a video file for analysis

4. Adjust the sleep threshold using the sidebar slider if needed

5. The application will display:
   - Live video feed with eye detection
   - Current status (Awake/Drowsy)
   - Audio alerts when drowsiness is detected

## Controls

- Sleep Threshold: Adjust the sensitivity of drowsiness detection (5-30)
- Input Source: Switch between webcam and video file
- Status Display: Shows current drowsiness state

## Notes

- Ensure good lighting conditions for better detection
- Position yourself properly in front of the camera
- The alarm sound will play when drowsiness is detected
- Press 'q' to quit the application

## License

This project is licensed under the MIT License - see the LICENSE file for details. 