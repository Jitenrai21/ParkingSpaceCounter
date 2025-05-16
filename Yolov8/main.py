import cv2
import pickle
import logging
from ultralytics import YOLO
from utils import check_overlap  # Assumes this returns True if boxes overlap
from configs import *

logging.getLogger().setLevel(logging.ERROR)

with open(PARK_POSITIONS_FILE, "rb") as f:
    park_positions = pickle.load(f)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Failed to open video file: {VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference
    results = model(frame, imgsz=480, verbose=False)[0]

    # Extract car boxes only
    car_boxes = [
        box.xyxy[0].tolist()
        for box in results.boxes
        if int(box.cls[0]) == CAR_CLASS_ID
    ]

    # Initialize counter for free spaces
    free_count = 0

    # Check each parking spot
    for x, y in park_positions:
        x2, y2 = x + SPOT_WIDTH, y + SPOT_HEIGHT
        park_rect = (x, y, x2, y2)

        occupied = any(check_overlap(park_rect, car_box) for car_box in car_boxes)
        color = (0, 0, 255) if occupied else (0, 255, 0)

        if not occupied:
            free_count += 1

        # Draw rectangle on the frame
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    # Overlay counter
    cv2.putText(
        frame, f"{free_count}/{len(park_positions)} Free",
        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )

    # Show frame
    cv2.namedWindow("YOLO Parking Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Parking Detection", frame_width, frame_height)
    cv2.imshow("YOLO Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
