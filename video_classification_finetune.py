import pandas as pd
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import Compose, Resize, Normalize
import torchvision.io as io
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV manifests
train_manifest = pd.read_csv("manifests/train_fold_1.csv")  
test_manifest = pd.read_csv("manifests/test_fold_1.csv")  

# Encode labels
label_encoder = LabelEncoder()
train_manifest["Emotion"] = label_encoder.fit_transform(train_manifest["Emotion"])
test_manifest["Emotion"] = label_encoder.transform(test_manifest["Emotion"])

# Define constants
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5
NUM_FRAMES = 8  # Number of frames sampled per video
RESOLUTION = 224  # Resize video frames to 224x224
FPS = 50  # Original frame rate
TARGET_FPS = FPS // 2  # Use half of the frames
BEST_MODEL_DIR = "best_model"

# Load Pretrained TimeSformer with mismatched sizes ignored
model_name = "facebook/timesformer-base-finetuned-k600"
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model = TimesformerForVideoClassification.from_pretrained(
    model_name,
    num_labels=2,  # Set the number of labels for your task
    ignore_mismatched_sizes=True  # Ignore size mismatches
)

# Modify the classifier to match your task
model.classifier = nn.Linear(model.config.hidden_size, 2)  # 2 classes: IDS and ADS
model.to(device)

# Define Transformations
transform = Compose([
    Resize((RESOLUTION, RESOLUTION)),
    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Define Dataset Class
class VideoDataset(Dataset):
    def __init__(self, manifest, transform=None, num_frames=NUM_FRAMES):
        self.file_paths = manifest["file_path"].values
        self.labels = manifest["Emotion"].values
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load video frames using torchvision
        frames, _, info = io.read_video(video_path, pts_unit="sec")

        # Calculate duration based on frame rate and total frames
        total_frames = frames.shape[0]
        duration = total_frames / FPS  # Duration in seconds
        max_frames = TARGET_FPS * 8  # Max frames for 8 seconds at target FPS

        # Clip to first 8 seconds if necessary
        if duration > 8:
            frames = frames[:max_frames]

        # Downsample frames to half the FPS by removing even frames
        frames = frames[::2]

        # Sample frames evenly if more than required
        total_frames = frames.shape[0]
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        sampled_frames = frames[indices]

        # Apply transformations
        if self.transform:
            sampled_frames = torch.stack([self.transform(frame.permute(2, 0, 1).float() / 255.0) for frame in sampled_frames])
        return sampled_frames, label

# Create datasets and dataloaders
train_dataset = VideoDataset(train_manifest, transform=transform)
test_dataset = VideoDataset(test_manifest, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define Optimizer and Criterion
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training Loop
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

best_accuracy = 0.0
if not os.path.exists(BEST_MODEL_DIR):
    os.makedirs(BEST_MODEL_DIR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # Add progress bar for each epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for videos, labels in progress_bar:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=videos, labels=labels)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

    # Evaluate the model on the test set
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(pixel_values=videos)
            predictions = torch.argmax(outputs.logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy after epoch {epoch + 1}: {accuracy * 100:.2f}%")

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"Saving the best model with accuracy: {best_accuracy * 100:.2f}%")
        model.save_pretrained(BEST_MODEL_DIR)
        feature_extractor.save_pretrained(BEST_MODEL_DIR)

# Final Test Accuracy
print(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")

