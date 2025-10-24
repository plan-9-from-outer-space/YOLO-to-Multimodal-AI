import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize YOLO model
model = YOLO("yolo11n.pt")  # Replace with the path to your YOLO model

# Initialize plot for real-time graph
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.set_facecolor("#F3F3F3")
frame_numbers = []
class_counts = defaultdict(list)  # Dictionary to store counts for each class

# Video path
video_path = "resources/videos/video.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # Process the frame for object detection
    frame_number += 1
    results = model(frame)
    detections = results[0].boxes  # Detected boxes

    # Update the graph with object count for each detected class
    frame_numbers.append(frame_number)

    # Create a dictionary to store counts for the current frame
    current_counts = defaultdict(int)
    detected_classes = set()  # To keep track of which classes were detected in the current frame

    for det in detections:
        class_id = int(det.cls[0])
        current_counts[class_id] += 1
        detected_classes.add(class_id)  # Mark this class as detected

    # For each detected class, update the count in the class_counts dictionary
    for class_id in detected_classes:
        class_counts[class_id].append(current_counts[class_id])

    # For classes that were not detected, append 0 to their counts for this frame
    for class_id in model.names:
        if class_id not in detected_classes:
            class_counts[class_id].append(0)

    # Limit the number of data points in the plot to 45
    if len(frame_numbers) > 45:
        frame_numbers.pop(0)
        for key in class_counts:
            class_counts[key].pop(0)

    ax.clear()

    # Plot only detected classes
    for class_id in detected_classes:
        ax.plot(frame_numbers, class_counts[class_id], label=model.names[class_id], linewidth=2)

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Object Count")
    ax.set_title("Real-Time Object Detection")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.draw()
    plt.pause(0.01)

    # Display processed frame with bounding boxes (optional)
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        label = f"{model.names[int(det.cls[0])]} {det.conf[0]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Disable interactive mode
plt.show()  # Display the final plot
