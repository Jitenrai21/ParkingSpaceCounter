import cv2

ref_point = []
drawing = False

def click_event(event, x, y, flags, param):
    global ref_point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.append((x, y))
        print(f"Point {len(ref_point)}: {x}, {y}")
        if len(ref_point) == 2:
            width = abs(ref_point[1][0] - ref_point[0][0])
            height = abs(ref_point[1][1] - ref_point[0][1])
            print(f"\nEstimated Dimensions:\nWidth = {width} px, Height = {height} px\n")

img = cv2.imread('carPark.png')  # Make sure this matches your current reference image
cv2.imshow("Click Two Opposite Corners of a Space", img)
cv2.setMouseCallback("Click Two Opposite Corners of a Space", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
