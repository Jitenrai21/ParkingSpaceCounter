import cv2
import numpy as np
import pickle

# Load video and positions
cap = cv2.VideoCapture('data/parking.mp4')
with open('park_positions', 'rb') as f:
    park_positions = pickle.load(f)

# Parameters
width, height = 134, 60
diff_threshold = 0.1  # You can tune this

# State holders
previous_frame = None
spots_status = [None for _ in park_positions]
counter = 0

def calc_diff(crop1, crop2):
    # Convert to grayscale for consistency
    crop1_gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    crop2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
    return np.abs(np.mean(crop1_gray) - np.mean(crop2_gray))

def check_empty(img_crop):
    # Apply thresholding for occupancy detection
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)
    count = cv2.countNonZero(img_thresh)
    return count < (0.1 * width * height)  # Adjust threshold

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    counter = 0

    if previous_frame is not None:
        for idx, (x, y) in enumerate(park_positions):
            crop_current = frame[y:y + height, x:x + width]
            crop_previous = previous_frame[y:y + height, x:x + width]

            diff = calc_diff(crop_current, crop_previous)

            if diff > diff_threshold * 255:  # Normalize threshold to pixel range
                is_empty = check_empty(crop_current)
                spots_status[idx] = is_empty
            # Else: keep previous status (do not re-check)

    else:
        # First frame: check all
        for idx, (x, y) in enumerate(park_positions):
            crop_current = frame[y:y + height, x:x + width]
            is_empty = check_empty(crop_current)
            spots_status[idx] = is_empty

    # Drawing
    for idx, (x, y) in enumerate(park_positions):
        color = (0, 255, 0) if spots_status[idx] else (0, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        if spots_status[idx]:
            counter += 1

    frame_out = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.rectangle(frame_out, (10, 10), (400, 60), (0, 0, 0), -1)
    cv2.putText(frame_out, f"Available: {counter} / {len(park_positions)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Display
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame_out)

    if cv2.waitKey(10) & 0xFF == 27:
        break

    previous_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()
