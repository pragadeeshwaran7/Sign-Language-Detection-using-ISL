import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('sign_language_model1.h5')

# Define class labels (adjust based on your actual training labels)
class_labels = [
    'ABOVE', 'ACROSS', 'ADVANCE', 'AFRAID', 'ALL', 'ALONE', 'ARISE', 'BAG',
    'BELOW', 'BRING', 'YES', 'ABOARD', 'ANGER', 'ASCEND', 'BESIDE', 'DRINK',
    'FLAG', 'HANG', 'MARRY', 'MIDDLE', 'MOON', 'PRISONER', 'ALL GONE'
]

st.title("🤟 Real-Time Sign Language Detection (Dual Hand Support)")
st.write("Detects gestures from both hands using MediaPipe, bounding boxes, and MLP/CNN-based prediction.")

# Webcam control
if "run" not in st.session_state:
    st.session_state.run = False

start = st.button("Start Webcam" if not st.session_state.run else "Stop Webcam")
if start:
    st.session_state.run = not st.session_state.run

FRAME_WINDOW = st.image([])

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Get model input shape
input_shape = model.input_shape[1:]  # Ignore batch dimension

cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            img_h, img_w, _ = frame.shape
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            xmin = int(min(x_coords) * img_w)
            ymin = int(min(y_coords) * img_h)
            xmax = int(max(x_coords) * img_w)
            ymax = int(max(y_coords) * img_h)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

            # Crop and preprocess hand
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size > 0:
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                normalized = resized / 255.0

                # Reshape based on model input
                if input_shape == (64, 64, 1):
                    input_array = normalized.reshape(1, 64, 64, 1)
                elif input_shape == (4096,):
                    input_array = normalized.reshape(1, 64 * 64)
                else:
                    st.error(f"Unexpected model input shape: {input_shape}")
                    continue

                # Prediction
                prediction = model.predict(input_array, verbose=0)
                pred_idx = np.argmax(prediction)
                pred_label = class_labels[pred_idx]
                confidence = prediction[0][pred_idx]

                label = f'Hand {idx+1}: {pred_label} ({confidence*100:.2f}%)'
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
hands.close()
