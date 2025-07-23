import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import pygame  # Use pygame for stable sound

# Load the trained model
model = tf.keras.models.load_model('gesture_help_model.keras')
print("✅ Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)
print("📷 Starting webcam feed...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Labels from training
class_names = ['Help', 'Normal']

# Initialize Pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.wav')

# Alert flag
alert_playing = False

def play_alert():
    global alert_playing
    alert_playing = True
    alert_sound.play()
    pygame.time.wait(int(alert_sound.get_length() * 1000))  # Wait for sound to finish
    alert_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # Default state
    label = "Normal"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_array = np.expand_dims(normalized_frame, axis=0)

        predictions = model.predict(input_array)
        class_index = np.argmax(predictions[0])
        label = class_names[class_index]

        if label == "Help" and not alert_playing:
            threading.Thread(target=play_alert).start()

    cv2.putText(frame, f'Prediction: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label == 'Help' else (0, 255, 0), 2)

    cv2.imshow("Gesture Help Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
