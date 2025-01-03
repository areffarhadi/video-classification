import pandas as pd
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import Compose, Resize, Normalize
import torchvision.io as io
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV manifest
test_manifest = pd.read_csv("manifests/test_fold_1.csv") 

# Encode labels to integers
label_encoder = LabelEncoder()
test_manifest["Emotion"] = label_encoder.fit_transform(test_manifest["Emotion"])

# Define constants
BATCH_SIZE = 4
NUM_FRAMES = 8  # Number of frames sampled per video
RESOLUTION = 224  # Resize video frames to 224x224
FPS = 50  # Original frame rate
TARGET_FPS = FPS // 2  # Use half of the frames
BEST_MODEL_DIR = "best_model"  # Directory where the fine-tuned model is saved

# Load the fine-tuned model and feature extractor
feature_extractor = AutoImageProcessor.from_pretrained(BEST_MODEL_DIR)
model = TimesformerForVideoClassification.from_pretrained(BEST_MODEL_DIR)
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
        label = self.labels[idx]  # Now numeric due to preprocessing

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

        return sampled_frames, torch.tensor(label, dtype=torch.long)

# Create test dataset and dataloader
test_dataset = VideoDataset(test_manifest, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Evaluate the model
model.eval()
total_correct = 0
total_samples = 0
predicted_labels = []
wrong_items = []

with torch.no_grad():
    for videos, labels in tqdm(test_loader, desc="Evaluating"):
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(pixel_values=videos)
        predictions = torch.argmax(outputs.logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # Collect all predicted labels
        predicted_labels.extend(predictions.cpu().tolist())

        # Collect wrong predictions
        for i, (pred, true_label) in enumerate(zip(predictions, labels)):
            if pred != true_label:
                wrong_items.append({
                    "file_path": test_dataset.file_paths[i],
                    "predicted": pred.item(),
                    "true_label": true_label.item()
                })

accuracy = total_correct / total_samples

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save results
pd.DataFrame(predicted_labels, columns=["Predicted Labels"]).to_csv("results/predicted_labels_1.csv", index=False)
pd.DataFrame(wrong_items).to_csv("results/wrong_items_1.csv", index=False)
print("Predicted labels saved to 'predicted_labels'")
print("Wrong predictions saved to 'wrong_items'")

