import torch
import cv2

from utils.original_model import OriginalModel
from utils.vgg16_model import VGG16Model
from utils.resnet_model import ResNet50Model
from utils.get_hand import detect_hand

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    dataset_path = "dataset"
    train_size = 0.85
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    num_epochs = 25

    original = OriginalModel(dataset_path, train_size, batch_size, device, num_epochs)
    resnet = ResNet50Model(dataset_path, train_size, batch_size, device, num_epochs)
    vgg16 = VGG16Model(dataset_path, train_size, batch_size, device, num_epochs)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        if not ret:
            break

        x_min, y_min, x_max, y_max = detect_hand(frame)
        new_img = frame[y_min:y_max, x_min:x_max]

        result_original = original.predict(new_img) # list of tuples like [('A', 0.9999999), ('B', 0.0000001)]
        result_resnet = resnet.predict(new_img) 
        result_vgg16 = vgg16.predict(new_img) 

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.putText(frame, f"Original: {result_original[0][0]} - {round(float(result_original[0][1]), 2)}", (x_min, y_max+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
        cv2.putText(frame, f"ResNet50: {result_resnet[0][0]} - {round(float(result_resnet[0][1]), 2)}", (x_min, y_max+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
        cv2.putText(frame, f"VGG16: {result_vgg16[0][0]} - {round(float(result_vgg16[0][1]), 2)}", (x_min, y_max+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)

        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





        

