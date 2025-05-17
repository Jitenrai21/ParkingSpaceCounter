import cv2

# cap = cv2.VideoCapture('parking.mp4')
cap = cv2.VideoCapture('carPark.mp4')

ret, frame = cap.read()
if ret:
    # cv2.imwrite('parking.png', frame)
    cv2.imwrite('carPark.png', frame)
    print('Image extracted.')
cap.release()