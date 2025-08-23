import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,                    # Track up to 2 hands
    min_detection_confidence=0.7        # Minimum confidence threshold for detection
)
mp_draw = mp.solutions.drawing_utils   # Utility for drawing hand landmarks

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

# Helper function to calculate Euclidean distance between two pixel coordinates
def euclidean_distance_pixels(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Helper function to check if only the index finger is raised
def is_only_index_raised(hand_landmarks):
    # Define landmark indices for fingertips and PIP joints
    tips = [mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]

    pips = [mp_hands.HandLandmark.THUMB_IP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]

    # Check if index finger is raised (tip above PIP in y-axis)
    index_raised = hand_landmarks.landmark[tips[1]].y < hand_landmarks.landmark[pips[1]].y

    # Check if other fingers are not raised (tip below PIP)
    others_down = all(
        hand_landmarks.landmark[tips[i]].y > hand_landmarks.landmark[pips[i]].y
        for i in [0, 2, 3, 4]
    )

    return index_raised and others_down

draw_points = []  # Stores drawing points when pointer is raised

# Main loop to process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Exit loop if frame not captured

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR image (OpenCV default) to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)  # Process the frame to detect hands

    h, w, _ = frame.shape  # Get frame dimensions
    pinch_points = {}      # Dictionary to store pinch midpoints for each hand
    hand_scale_cm_per_pixel = None  # Will be estimated based on hand size

    # If hands are detected and handedness info is available
    if results.multi_hand_landmarks and results.multi_handedness:
        # Loop through each detected hand and its handedness label
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'

            # Extract key landmarks for pinch detection and hand size estimation
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized coordinates (0–1) to pixel coordinates
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

            # Draw hand landmarks and connections on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Estimate hand length in pixels (wrist to middle fingertip)
            hand_length_px = euclidean_distance_pixels(wrist_pos, middle_pos)

            # Use average adult hand length (~18 cm) to estimate scale
            if hand_length_px > 0:
                hand_scale_cm_per_pixel = 18.0 / hand_length_px

            # Detect pinch gesture by measuring distance between thumb and index fingertips
            pinch_distance_norm = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
            )

            # If pinch distance is below threshold, consider it a pinch
            if pinch_distance_norm < 0.05:
                # Calculate midpoint between thumb and index fingertips
                midpoint = (
                    (thumb_pos[0] + index_pos[0]) // 2,
                    (thumb_pos[1] + index_pos[1]) // 2
                )
                # Store midpoint in dictionary with hand label
                pinch_points[label] = midpoint

            # Check if only the pointer finger is raised
            pointer_raised = is_only_index_raised(hand_landmarks)

            if pointer_raised:
                index_pos = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                             int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))

                # Display label
                cv2.putText(frame, f"{label} Pointer Raised", (index_pos[0], index_pos[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Append current index fingertip position to draw list
                draw_points.append(index_pos)

    # Display logic based on how many hands are pinching
    if len(pinch_points) == 1:
        # Only one hand is pinching
        label = list(pinch_points.keys())[0]
        midpoint = pinch_points[label]
        cv2.putText(frame, f"{label} Pinch", (midpoint[0], midpoint[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    elif len(pinch_points) == 2 and hand_scale_cm_per_pixel:
        # Both hands are pinching, draw line and show length in cm
        pt1 = pinch_points["Left"]
        pt2 = pinch_points["Right"]

        # Draw line between pinch points
        cv2.line(frame, pt1, pt2, (0, 255, 0), 4)

        # Calculate pixel distance and convert to centimeters
        line_length_px = euclidean_distance_pixels(pt1, pt2)
        line_length_cm = line_length_px * hand_scale_cm_per_pixel

        # Display length at midpoint of the line
        cv2.putText(frame, f"Length: {line_length_cm:.1f} cm",
                    ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display "Double Pinch!" message
        cv2.putText(frame, "Double Pinch!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Draw the freehand path from stored points
    for i in range(1, len(draw_points)):
        cv2.line(frame, draw_points[i - 1], draw_points[i], (0, 0, 255), 3)

    # Show the processed frame in a window
    cv2.imshow("Pinch Detection Refined", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Press 'c' to clear drawing
    if cv2.waitKey(1) & 0xFF == ord('c'):
        draw_points.clear()

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()