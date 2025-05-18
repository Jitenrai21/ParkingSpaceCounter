import cv2
import pickle
import numpy as np
from ultralytics import YOLO

# Load YOLOv8n model 
model = YOLO('yolov8n.pt') 

# Load pre-saved parking positions (list of (x, y) tuples)
with open('carpark_positions', 'rb') as f:
    park_positions = pickle.load(f)

# Parking spot dimensions (must match the original saved setup)
SPOT_WIDTH, SPOT_HEIGHT = 107, 45
FULL_PIXEL_COUNT = SPOT_WIDTH * SPOT_HEIGHT
THRESHOLD_RATIO = 0.15  # Adjustable

# Class ID for car in COCO dataset used by YOLO
CAR_CLASS_ID = 2  # 2 corresponds to 'car' in COCO

# Initialize video
cap = cv2.VideoCapture('data/carPark.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)

    # Run YOLO inference
    results = model(frame, conf=0.3, iou=0.4, verbose=False)[0]

    # Filters out only car detections
    car_boxes = [
        box.xyxy[0].cpu().numpy().astype(int)
        for box in results.boxes
        if int(box.cls[0]) == CAR_CLASS_ID
    ]
    free_count = 0
    for (x, y) in park_positions:
        x2, y2 = x + SPOT_WIDTH, y + SPOT_HEIGHT
        parking_crop = thresh[y:y2, x:x2]

        pixel_count = cv2.countNonZero(parking_crop)
        ratio = pixel_count / FULL_PIXEL_COUNT

        # Check for overlap with YOLO-detected car boxes
        occupied_by_detection = False
        for box in car_boxes:
            bx1, by1, bx2, by2 = box
            if bx2 > x and bx1 < x2 and by2 > y and by1 < y2:
                occupied_by_detection = True
                break

        # Combine both logic options (you can choose to use only one)
        is_occupied = (ratio > THRESHOLD_RATIO) or occupied_by_detection
        
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        if not is_occupied:
            free_count += 1

        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(overlay, f"{ratio:.2f}", (x + 2, y + SPOT_HEIGHT - 5),
                    font, 0.6, (255, 255, 255), 1)

    # Show count
    cv2.rectangle(overlay, (0, 0), (250, 60), (128, 0, 255), -1)
    cv2.putText(overlay, f"Available: {free_count}/{len(park_positions)} spots",
                (10, 40), font, 1.2, (255, 255, 255), 2)

    cv2.namedWindow('YOLOv8 + Threshold Parking Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 + Threshold Parking Detection', 1280, 720)
    cv2.imshow('YOLOv8 + Threshold Parking Detection', overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
