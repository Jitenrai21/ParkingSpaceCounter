import cv2
import pickle
import numpy as np

# Load image
image = cv2.imread('data/parking.png')
if image is None:
    print("Error: Image not found.")
    exit()

# Resize image for screen fit
display_width = 800
display_height = 600
image_resized = cv2.resize(image, (display_width, display_height))

spots = []
current_polygon = []

def mouse_callback(event, x, y, flags, param):
    global current_polygon, spots

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        if len(current_polygon) == 4:
            spots.append(current_polygon.copy())
            current_polygon.clear()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_polygon:
            # Remove last point in current polygon
            current_polygon.pop()
        elif spots:
            # Remove last complete polygon
            spots.pop()

# Setup window
cv2.namedWindow("Select Spots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Spots", display_width, display_height)
cv2.setMouseCallback("Select Spots", mouse_callback)

while True:
    display = image_resized.copy()

    # Draw completed polygons
    for spot in spots:
        cv2.polylines(display, [np.array(spot, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw current polygon in progress
    for point in current_polygon:
        cv2.circle(display, point, 5, (255, 0, 0), -1)

    cv2.imshow("Select Spots", display)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break
    elif key == ord('s'):  # Press 's' to save spots
        with open("polygon_spots.pkl", "wb") as f:
            pickle.dump(spots, f)
        print("Spots saved to polygon_spots.pkl")

cv2.destroyAllWindows()
