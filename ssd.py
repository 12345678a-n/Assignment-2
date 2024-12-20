import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# Load a pretrained SSD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval().to(device)

# Define a list of COCO class labels (91 classes including background)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'TV', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Open the webcam
cap = cv2.VideoCapture(0)

# Define transformation for frames
transform = T.Compose([
    T.ToTensor(),  # Convert to tensor
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transformations
    input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()  # Bounding boxes
    scores = predictions[0]['scores'].cpu().numpy()  # Confidence scores
    labels = predictions[0]['labels'].cpu().numpy()  # Class labels

    # Draw bounding boxes on the frame
    for i, score in enumerate(scores):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = COCO_CLASSES[labels[i]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
