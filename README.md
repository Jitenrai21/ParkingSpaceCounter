
# Parking Space Counter with Flexible Polygon-Based Detection

This project implements a parking space detection system using OpenCV. It supports flexible spot definitions using 4-point polygons and optimized frame processing logic to improve efficiency.

## Features

- Support for manually defined 4-point polygons as flexible parking spot boundaries.
- Efficient processing by skipping every alternate frame for performance boost.
- Adaptive thresholding technique used for better contrast and robust parking detection.
- Visual overlay of free and occupied spots with live count.

## Flexible Polygon-Based Spot Detection

The system reads parking spot boundaries from `polygon_spots.pkl`, which stores a list of 4-point polygons. Each polygon corresponds to a unique parking spot and is drawn manually using a mouse event GUI tool.

Each spot is analyzed using binary masking and non-zero pixel count within the polygon mask to determine its occupancy status.

## Frame Processing Optimization

To reduce the processing load, the system includes a frame rate optimization mechanism where only 1 out of every 2 frames is processed.

This reduces CPU usage by approximately 50% while maintaining reliable detection performance.

## Adaptive Thresholding and Occupancy Logic

After converting the frame to grayscale and applying Gaussian blur, adaptive thresholding is performed:

img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)

Each cropped spot (rectangular or polygonal) is processed to count the number of non-zero pixels (white pixels). If the white pixel ratio exceeds 85%, the spot is marked as occupied.

## Requirements

- Python 3.7+
- pip (Python package manager)

## Dependencies

Install required packages using:

```

pip install -r requirements.txt

```

`requirements.txt` should include:
opencv-python
scikit-image
scikit-learn
matplotlib

## How to Run

1. Clone the repository:

```

git clone https://github.com/Jitenrai21/ParkingSpaceCounter

```
## Output

- Red rectangles for occupied spots
- Green rectangles for available spots
- Display showing number of available out of total spots
- Transparent overlays for clean visual feedback

## Future Work

- Detection of parking duration and violations
- License plate recognition and vehicle tracking


