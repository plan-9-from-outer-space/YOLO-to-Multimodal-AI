
import cv2
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow
import numpy as np
from PIL import Image
import depth_pro

model = YOLO("yolo11s.pt")
image_path = "Images/image1.jpeg" 
image_input = cv2.imread(image_path)
results = model(image_input)
person_boxes = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    for box, cls in zip(boxes, classes):
      # Only care about the "person" class for this script.
      if result.names[int(cls)] == "person":
        x1, y1, x2, y2 = map(int, box[:4])
        person_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(image_input, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2_imshow(image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
depth_np = depth.squeeze().cpu().numpy()

for x1, y1, x2, y2 in person_boxes:
  center_x = (x1 + x2)//2
  center_y = (y1 + y2)//2
  depth_value = depth_np[center_y, center_x]
  text = f"Depth: {depth_value:.2f} m"
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  font_thickness = 2
  text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
  text_x = x1
  text_y = y1 - 10
  rect_x1 = text_x - 5
  rect_y1 = text_y - text_size[1] - 10
  rect_x2 = text_x + text_size[0] + 5
  rect_y2 = text_y + 5

  cv2.rectangle(image_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0,0,0), -1)
  cv2.putText(image_input, text, (text_x, text_y), font, font_scale, (255,255,255), font_thickness)

cv2.imshow(image_input) # cv2_imshow
cv2.imwrite("depth.jpg", image_input)
cv2.waitKey(0)
# cv2.destroyAllWindows()  

depth_np_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
inv_depth_np_normalized = 1.0 - depth_np_normalized
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imshow(depth_colormap) # cv2_imshow
cv2.imwrite("depth_colormap.jpg", depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
