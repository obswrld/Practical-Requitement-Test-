import cv2

video_path = "data/overlay_output_final26.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening this video")
    exit()

ret, frame = cap.read()
if ret:
    print("Frame shape: ", frame.shape)
    cv2.imshow("First frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not read the first frame")

cap.release()
