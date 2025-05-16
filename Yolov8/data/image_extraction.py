import cv2

cap = cv2.VideoCapture('parking.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('parking.png', frame)
    print('Image extracted.')
cap.release()
