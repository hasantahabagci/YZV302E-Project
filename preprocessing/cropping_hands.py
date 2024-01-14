import cv2
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize global variables for cropping
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

# Function for manual cropping
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False

# Main function to process video stream
def process_image_stream(frame, image_path):
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = frame.shape[1]
            y_min = frame.shape[0]
            inc_coeff = 1.15
            subs_coeff = 0.85
            for lm in handLMs.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_max, x_min = max(x, x_max), min(x, x_min)
                y_max, y_min = max(y, y_max), min(y, y_min)

            x_min, x_max = int(x_min*subs_coeff) if int(x_min*subs_coeff) > 0 else 0 , int(x_max*inc_coeff) if int(x_max*inc_coeff) < frame.shape[1] else frame.shape[1]
            y_min, y_max = int(y_min*subs_coeff) if int(y_min*subs_coeff) > 0 else 0 , int(y_max*inc_coeff) if int(y_max*inc_coeff) < frame.shape[0] else frame.shape[0]

            if x_max - x_min < y_max - y_min:
                diff = y_max - y_min - (x_max - x_min)
                x_min, x_max = x_min - int(diff/2), x_max + int(diff/2)
            elif x_max - x_min > y_max - y_min:
                diff = x_max - x_min - (y_max - y_min)
                y_min, y_max = y_min - int(diff/2), y_max + int(diff/2)

            cropped_image = frame[y_min:y_max, x_min:x_max]

            if cropped_image.shape[0]*cropped_image.shape[1] < frame.shape[0]*frame.shape[1]/5:
                break
            save_cropped_image(cropped_image, image_path)
            break

    else:
        print("Could not detect hand landmarks for image: ", image_path)

        # if hand_landmarks:
        #     for handLMs in hand_landmarks:
        #         mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)


    cv2.destroyAllWindows()

# Function to save cropped image
def save_cropped_image(image, image_path):
    # cv2.imshow("Cropped Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    try:
        cv2.imwrite(image_path, image)
    except Exception as e:
        print("Could not save image: ", image_path)
        print(e)
        pass

if __name__ == "__main__":
    dataset_path = "../dataset"
    classes = os.listdir(dataset_path)
    try:
        classes.remove(".DS_Store")
    except ValueError:
        pass
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = os.listdir(class_path)
        try:
            image_files.remove(".DS_Store")
        except ValueError:
            pass
        for image_name in image_files:
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            process_image_stream(image, image_path)
            



