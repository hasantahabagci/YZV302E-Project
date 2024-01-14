import cv2
import os
import mediapipe as mp

def detect_hand(image):
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


    # Convert the image color from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)
    inc_coeff = 1.15
    subs_coeff = 0.85
    # Check if a hand is detected and get the bounding box
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box
            x_max = 0
            y_max = 0
            x_min = image.shape[1]
            y_min = image.shape[0]

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y

            # Draw the bounding box
            #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        x_min, x_max = int(x_min*subs_coeff) if int(x_min*subs_coeff) > 0 else 0 , int(x_max*inc_coeff) if int(x_max*inc_coeff) < image.shape[1] else image.shape[1]
        y_min, y_max = int(y_min*subs_coeff) if int(y_min*subs_coeff) > 0 else 0 , int(y_max*inc_coeff) if int(y_max*inc_coeff) < image.shape[0] else image.shape[0]

        return x_min, y_min, x_max, y_max
    
    else:
        return 0, 0, image.shape[1], image.shape[0]



