import cv2
import os

video_path = "data/overlay_output_final26.avi"

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame")
else:
    output_path = "output/first_frame.jpg"
    cv2.imwrite(output_path, frame)
    print(f"First frame saved to: {output_path}")
    print("Frame shape: ", frame.shape)
