import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import VOCDataset  # Import the VOCDataset class from your dataset file

# Define collate_fn outside the main function
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Dataset paths
    dataset_path = "C:/Users/murat/Desktop/deneme/train"  # Replace with your dataset path

    # Data transformations
    transform = T.Compose([T.ToTensor()])
    dataset = VOCDataset(root=dataset_path, transforms=transform)  # Use VOCDataset
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Load the pretrained model and modify it for your dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3  # Adjust according to your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Train the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (images, targets) in enumerate(data_loader):
            # Skip batches where either images or targets are None
            if images is None or targets is None:
                continue
            
            # Ensure both images and targets are not None
            valid_images = [image.to(device) for image in images if image is not None]
            valid_targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]

            # Skip batches where there are no valid images or targets
            if not valid_images or not valid_targets:
                continue

            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Iteration [{i+1}/{len(data_loader)}], Loss: {losses.item()}")

    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), 'fasterrcnn_finetuned.pth')
    print("Model saved to fasterrcnn_finetuned.pth")

if __name__ == "__main__":
    main()