from cellpose import models
import cv2
import matplotlib.pyplot as plt

video_path = "data/overlay_output_final26.avi"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read the first frame from the video")
    exit()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

model = models.CellposeModel(model_type='cyto')

masks, flows, styles, diams = model.eval(frame_rgb, diameter=None, channels=[2,1])

print("Segmentation done")
print("Number of objects detected: ", masks.max())

plt.figure(figsize=(10,5))
plt.imshow(frame_rgb)
plt.imshow(masks, alpha=0.5)
plt.title("Segmentation Overlay")
plt.axis('off')
plt.show()