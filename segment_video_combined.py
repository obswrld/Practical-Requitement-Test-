import cv2
import numpy as np
import os
from cellpose import models
from cellpose.utils import outlines_list

INPUT_VIDEO = "data/overlay_output_final26.avi"
OUTPUT_FOLDER = "output"
MASK_ONLY_OUTPUT = os.path.join(OUTPUT_FOLDER, "mask_only.avi")
COMBINED_OUTPUT = os.path.join(OUTPUT_FOLDER, "combined_overlay.avi")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Error: Could not open input video at '{INPUT_VIDEO}'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
mask_writer = cv2.VideoWriter(MASK_ONLY_OUTPUT, fourcc, fps, (width, height))
combined_writer = cv2.VideoWriter(COMBINED_OUTPUT, fourcc, fps, (width, height))

model = models.CellposeModel(gpu=False)

frame_id = 0

print("Processing video with Cellpose segmentation...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    masks, flows, styles, diams = model.eval(frame_rgb, diameter=None, channels=[0, 0])

    mask_img = np.zeros_like(frame_rgb)
    outlines = outlines_list(masks)
    for out in outlines:
        for y, x in out:
            mask_img[y, x] = [255, 0, 0]

    mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
    mask_writer.write(mask_bgr)

    color_mask = np.zeros_like(frame_rgb)
    color_mask[masks > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(frame_rgb, 0.7, color_mask, 0.4, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    combined_writer.write(overlay_bgr)

    frame_id += 1
    if frame_id % 10 == 0:
        print(f"Processed frame {frame_id}")

cap.release()
mask_writer.release()
combined_writer.release()

print("\n=== DONE ===")
print("Mask-only video saved as:", MASK_ONLY_OUTPUT)
print("Combined overlay video saved as:", COMBINED_OUTPUT)
