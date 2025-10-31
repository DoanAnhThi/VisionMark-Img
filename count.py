from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes

from src.utils import save_detection_results

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")

# Define custom classes
target_classes = ["person"]  # <--------- Change this to the class(es) you want to detect
model.set_classes(target_classes)

# Execute prediction on an image
results: Boxes = model.predict("samples/test02.png")

# Count total detections (bounding boxes) across all results
total_detections = sum(len(result.boxes) for result in results)

if total_detections == 0:
    print("No detections found for the specified classes.")
else:
    print(
        f"Total bounding boxes detected for {', '.join(target_classes)}: {total_detections}"
    )

# Save detection results as images
save_detection_results(results)

