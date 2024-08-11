
from ultralytics import YOLO
import cv2

# Loading the model
model = YOLO("yolov8n.pt")

# Path to the input and output video files
input_video_path = "FENERTEPE_SISTEM ODA 2_20240710124459_20240710125958.mp4"
output_video_path = "Processed_Video_1.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)



# Get video properties for output video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

target_classes = [0]  # Person class in COCO dataset

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5)

    for pred in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = pred
        if int(class_id) in target_classes:
            # Draw the bounding box in red
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Put the class label on the rectangle
            class_label = "Person"  # Class name
            label_size, base_line = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (int(x1), int(y1) - label_size[1] - 10)
            bottom_right = (int(x1) + label_size[0], int(y1))
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), (int(x1) + label_size[0], int(y1)), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, class_label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Put the "Person Detected" text with a filled black rectangle background
            notification_label = "Person Detected"
            label_size, base_line = cv2.getTextSize(notification_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            top_left = (frame.shape[1] - label_size[0] - 30, 30)
            bottom_right = (frame.shape[1] - 10, 30 + label_size[1] + base_line)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, notification_label, (frame.shape[1] - label_size[0] - 20, 30 + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to the output video file
    out.write(frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
