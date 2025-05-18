import cv2
import numpy as np
import pickle

# Video and saved polygon spot positions
cap = cv2.VideoCapture('data/parking.mp4')
with open('polygon_spots.pkl', 'rb') as f:
    polygon_spots = pickle.load(f)

# Threshold ratio: < means empty
OCCUPIED_THRESHOLD = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX

def count_white_pixels_inside_polygon(thresh_img, polygon):
    # Create mask
    mask = np.zeros(thresh_img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)

    # Apply mask
    masked_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)

    white_pixels = cv2.countNonZero(masked_img)
    area = cv2.countNonZero(mask)  # Total number of pixels in polygon

    ratio = white_pixels / area if area > 0 else 0
    return ratio

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    overlay = frame.copy()

    # Image processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)

    free_count = 0
    for idx, polygon in enumerate(polygon_spots):
        ratio = count_white_pixels_inside_polygon(thresh, polygon)

        if ratio < OCCUPIED_THRESHOLD:
            color = (0, 255, 0)  # Free
            free_count += 1
        else:
            color = (0, 0, 255)  # Occupied

        pts = np.array(polygon, np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(overlay, f"{ratio:.2f}", tuple(pts[0]), font, 0.6, color, 2)

    # Display count
    cv2.rectangle(overlay, (0, 0), (250, 40), (50, 50, 50), -1)
    cv2.putText(overlay, f"{free_count}/{len(polygon_spots)} Free", (10, 30),
                font, 0.8, (255, 255, 255), 2)

    cv2.imshow("Parking Spot Detection", overlay)
    if cv2.waitKey(90) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# """YOLO Implementation"""
# import cv2
# import numpy as np
# import pickle
# from ultralytics import YOLO

# # Load YOLOv8n model
# model = YOLO('yolov8n.pt')  # or your custom model like 'best.pt'

# # Load video and polygon spots
# cap = cv2.VideoCapture('data/parking.mp4')
# with open('polygon_spots.pkl', 'rb') as f:
#     polygon_spots = pickle.load(f)

# font = cv2.FONT_HERSHEY_SIMPLEX

# def is_polygon_occupied(polygon, car_boxes, iou_threshold=0.15):
#     poly_mask = np.zeros((600, 800), dtype=np.uint8)
#     cv2.fillPoly(poly_mask, [np.array(polygon, dtype=np.int32)], 255)
#     poly_area = cv2.countNonZero(poly_mask)

#     for box in car_boxes:
#         x1, y1, x2, y2 = map(int, box[:4])
#         car_mask = np.zeros_like(poly_mask)
#         cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)
#         intersection = cv2.bitwise_and(poly_mask, car_mask)
#         inter_area = cv2.countNonZero(intersection)

#         if poly_area > 0 and inter_area / poly_area > iou_threshold:
#             return True
#     return False

# while True:
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (800, 600))
#     overlay = frame.copy()

#     # Run YOLO detection
#     results = model(frame, verbose=False)[0]
#     car_boxes = [box.xyxy[0].cpu().numpy() for box in results.boxes if int(box.cls) in [2, 3, 5, 7]]  # car, motorcycle, bus, truck

#     free_count = 0
#     for polygon in polygon_spots:
#         occupied = is_polygon_occupied(polygon, car_boxes)

#         color = (0, 0, 255) if occupied else (0, 255, 0)
#         if not occupied:
#             free_count += 1

#         pts = np.array(polygon, dtype=np.int32)
#         cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

#     # Display count
#     cv2.rectangle(overlay, (0, 0), (250, 40), (50, 50, 50), -1)
#     cv2.putText(overlay, f"{free_count}/{len(polygon_spots)} Free", (10, 30),
#                 font, 0.8, (255, 255, 255), 2)

#     cv2.imshow("YOLOv8 Parking Detection", overlay)
#     if cv2.waitKey(60) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
