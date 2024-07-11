import os
from ultralytics import YOLO
import cv2

model_path = "/Users/alinaandreeva/Downloads/request/best_saved_model/best_float32.tflite"
test_images_dir = "/Users/alinaandreeva/Downloads/request/datasets/test/images"
results_dir = "/Users/alinaandreeva/Downloads/request/test_results_tflite"

os.makedirs(results_dir, exist_ok=True)

class_names = ['complete trolley with wheels', 'complete trolley without wheels', 'side trolley with wheels', 'side trolley without wheels']

test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        continue

    input_size = (640, 640)
    img_resized = cv2.resize(img, input_size)

    try:
        results = model.predict(img_resized, verbose=False)
    except ValueError as e:
        print(f"Error during model prediction: {e}")
        continue

    for i, result in enumerate(results):
        # result.show()
        result.save(filename=os.path.join(results_dir, f"result_{i}.jpg"))

        detections = []
        for detection in result.boxes:
            class_id = detection.cls.item()
            class_name = class_names[int(class_id)]
            confidence = detection.conf.item()
            bbox = detection.xyxy.tolist()[0]

            detections.append((class_name, confidence, bbox))

        print(f"Image: {os.path.basename(img_path)}, Detections: {tuple(detections)}")