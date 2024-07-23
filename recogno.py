import cv2
import numpy as np

# List of class labels MobileNetSSD was trained to detect
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Load the pre-trained MobileNetSSD model from the disk
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Initialize the webcam using DirectShow backend for better compatibility on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened.")

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the dimensions of the frame
    h, w = frame.shape[:2]

    # Print the frame dimensions for debugging
    print(f"Frame dimensions: {w}x{h}")

    # Prepare the frame to be fed to the neural network
    # Resize the frame to 300x300 pixels, normalize it, and convert it to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Print blob shape for debugging
    print(f"Blob shape: {blob.shape}")

    # Set the input to the neural network
    net.setInput(blob)

    # Perform forward pass and get the network's output
    detections = net.forward()

    # Print the number of detections for debugging
    print(f"Number of detections: {detections.shape[2]}")

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.2:
            # Get the index of the class label from the detection
            idx = int(detections[0, 0, i, 1])

            # Calculate the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box coordinates are within the frame dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Draw the bounding box and label on the frame
            label = f"{class_names[idx]}: {round(confidence*100)}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print detection details for debugging
            print(f"Detected {class_names[idx]} with confidence {confidence:.2f} at [{startX}, {startY}, {endX}, {endY}]")

    # Display the frame with the detections
    cv2.imshow("Object Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop.")
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Webcam released and OpenCV windows closed.")
