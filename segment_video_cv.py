import cv2
import numpy as np
import os

INPUT_VIDEO = "data/overlay_output_final26.avi"
OUTPUT_FOLDER = "output"
OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, "output_segmented.avi")

K = 3

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print(f"Error: Could not open input video at '{INPUT_VIDEO}'")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (width, height))

print("Processing video...")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pixel_values = frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)

    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(frame.shape)

    edges = cv2.Canny(segmented, 80, 150)
    outline = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    result = cv2.addWeighted(frame, 0.7, outline, 0.8, 0)

    out.write(result)

    frame_id += 1
    if frame_id % 10 == 0:
        print(f"Processed frame {frame_id}")

cap.release()
out.release()

print(f"\nDone! Segmented video saved as: {OUTPUT_VIDEO}")
