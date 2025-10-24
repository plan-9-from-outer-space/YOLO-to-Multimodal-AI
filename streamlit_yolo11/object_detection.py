# Object Detection on Image
from ultralytics import YOLO

# Load the YOLO11 Pre-Trained Model
model = YOLO("yolo11n.pt")

# Perform Object Detection on an Image
results = model.predict("resources/images/bus.jpg", save=True)

results[0].show()

print(results[0].save_dir)
