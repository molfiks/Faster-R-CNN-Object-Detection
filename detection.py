import torch
import torchvision.transforms as T
import torchvision.models.detection
import cv2
from PIL import Image, ImageDraw
import numpy as np

def main():
    # Load your trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 3  # Number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('fasterrcnn_finetuned.pth'))
    model.eval()

    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    cap = cv2.VideoCapture(0)  # Capture video from webcam

    class_names = {1: "bottle", 2: "can"}  # Adjust according to your dataset

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Convert PIL Image to tensor
        image_tensor = T.ToTensor()(pil_image).unsqueeze(0).to(device)

        # Perform object detection
        with torch.no_grad():
            predictions = model(image_tensor)

        prediction = predictions[0]

        # Draw bounding boxes and labels for bottles or cans
        draw = ImageDraw.Draw(pil_image)
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > 0.5 and label.item() in [1, 2]:  # Filter for bottles (label 1) or cans (label 2)
                class_name = class_names[label.item()]
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                draw.text((box[0], box[1]), f"{class_name}: {score:.2f}", fill="red")

        # Convert PIL Image back to OpenCV format
        frame_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Object Detection', frame_with_boxes)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()