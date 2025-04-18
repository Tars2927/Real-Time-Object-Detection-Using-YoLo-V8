import cv2

cap = cv2.VideoCapture(0)  # Try 1, 2, etc., if 0 doesn't work

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()