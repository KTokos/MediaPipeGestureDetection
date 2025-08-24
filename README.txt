============================================================
README.txt - Pinch Detection and Gesture Drawing with MediaPipe
============================================================

📌 DESCRIPTION
--------------
This Python script uses MediaPipe and OpenCV to detect hand gestures in real time via webcam. It identifies pinch gestures and raised index fingers, enabling:
- Detection of single and double hand pinches
- Measurement of distance between pinch points in centimeters
- Freehand drawing using the index fingertip when raised

⚙️ REQUIREMENTS
---------------
- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- Webcam (default device index 0)

Install dependencies using:
    pip install opencv-python mediapipe

🚀 HOW TO RUN
-------------
1. Save the script as `gesture_draw.py`
2. Open a terminal and run:
    python gesture_draw.py
3. Press 'q' to quit the application
4. Press 'c' to clear the drawing canvas

🧠 FEATURES
-----------
- Real time hand tracking for up to 2 hands
- Pinch gesture detection using thumb and index fingertips
- Measurement of pinch distance between hands (in cm)
- Freehand drawing when only the index finger is raised
- Visual feedback overlays for gestures and measurements

📎 LOGIC OVERVIEW
-----------------
- Uses MediaPipe Hands to detect landmarks
- Converts normalized coordinates to pixel positions
- Detects pinch by measuring normalized thumb–index distance
- Estimates hand scale using wrist to middle fingertip length
- Draws lines between pinch points and displays length
- Tracks index fingertip for drawing when raised alone

💡 TIPS
-------
- Ensure good lighting and clear background for better detection
- Keep hands within the camera frame and avoid occlusion
- Adjust `min_detection_confidence` or pinch threshold if needed

🧹 CLEAN EXIT
-------------
- Webcam is released and OpenCV windows are closed automatically

👨‍💻 AUTHOR
-----------
Kristian Tokos, Computer Science Major, (Albany, NY)
