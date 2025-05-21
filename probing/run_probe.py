import torch
import os
import random
import json
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    CLIPModel,
    CLIPProcessor,
    ViTModel,
    ViTFeatureExtractor,
    AutoModel,
    AutoFeatureExtractor,
)
import timm  
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description="Run probing with a specified model.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
parser.add_argument("--unfreeze_encoder", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to update encoder weights")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = args.model_name
req_grad = args.unfreeze_encoder

if "clip" in model_name.lower():
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    feature_extraction_method = "clip"
elif "vit" in model_name.lower():
    model = ViTModel.from_pretrained(model_name).to(device)
    processor = ViTFeatureExtractor.from_pretrained(model_name)
    feature_extraction_method = "vit"
else:
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extraction_method = "auto"

for param in model.parameters():
    param.requires_grad = req_grad

class ImagePairDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if '3D' not in img][:500]
        
        self.pairs = []
        for image1_path in self.image_paths:
            self.pairs.append((image1_path, image1_path, 1))
            negative_image_path = random.choice([img for img in self.image_paths if img != image1_path])
            self.pairs.append((image1_path, negative_image_path, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image1_path, image2_path, label = self.pairs[idx]
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")
        image1 = self.processor(images=image1, return_tensors="pt")["pixel_values"].squeeze(0)
        image2 = self.processor(images=image2, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return image1, image2, torch.tensor(label, dtype=torch.float32)

image_dir = "../data-1-updated/images"  
dataset = ImagePairDataset(image_dir, processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

class ProbingModel(nn.Module):
    def __init__(self, model, feature_extraction_method):
        super(ProbingModel, self).__init__()
        self.model = model
        self.feature_extraction_method = feature_extraction_method
        if feature_extraction_method == "clip":
            self.projection_dim = model.config.projection_dim
        else:
            self.projection_dim = getattr(model.config, 'hidden_size', 768)  # Fallback to default size
        self.probing_layer = nn.Linear(self.projection_dim * 2, 1)  # Binary classification

    def forward(self, image1, image2):
        with torch.no_grad() if not req_grad else torch.enable_grad():
            if self.feature_extraction_method == "clip":
                image1_features = self.model.get_image_features(image1).float()
                image2_features = self.model.get_image_features(image2).float()
            elif self.feature_extraction_method == "dino":
                image1_features = self.model.forward_features(image1).mean(dim=1).float()
                image2_features = self.model.forward_features(image2).mean(dim=1).float()
            else:
                image1_features = self.model(image1).last_hidden_state.mean(dim=1).float()
                image2_features = self.model(image2).last_hidden_state.mean(dim=1).float()

        combined_features = torch.cat((image1_features, image2_features), dim=1)
        output = self.probing_layer(combined_features)
        return torch.sigmoid(output), image1_features, image2_features

probing_model = ProbingModel(model, feature_extraction_method).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    probing_model.parameters() if req_grad else probing_model.probing_layer.parameters(),
    lr=0.00001
)

output_answers = []

for epoch in range(30):  
    print(f"Epoch {epoch + 1}")
    total_samples = 0
    correct_predictions = 0
    total_loss = 0
    total_cosine_similarity = 0
    num_negative_pairs = 0
    all_predictions = []

    for images1, images2, labels in dataloader:
        images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)


        outputs, image1_features, image2_features = probing_model(images1, images2)
        outputs = outputs.squeeze()
        outputs = outputs.view(-1)
        labels = labels.view(-1)


        loss = criterion(outputs, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predicted_labels = (outputs >= 0.5).float()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

        all_predictions.extend(predicted_labels.cpu().numpy().tolist())

        for i in range(labels.size(0)):
            if labels[i] == 0: 
                total_cosine_similarity += F.cosine_similarity(
                    image1_features[i].unsqueeze(0),
                    image2_features[i].unsqueeze(0),
                    dim=1
                ).item()
                num_negative_pairs += 1

    accuracy = correct_predictions / total_samples * 100
    avg_loss = total_loss / len(dataloader)
    avg_cosine_similarity = total_cosine_similarity / num_negative_pairs if num_negative_pairs > 0 else 0

    output_data = {
        "model_name": model_name,
        "epoch": epoch + 1,
        "accuracy": accuracy,
        "average_loss": avg_loss,
        "average_cosine_similarity": avg_cosine_similarity,
        "predictions": all_predictions
    }

    with open(f"{model_name.replace('/', '_')}_probe_unfreeze_outputs.txt", 'a') as f:
        f.write(f"------------------------------Epoch {epoch + 1}---------------------------------\n")
        f.write(f"Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}, Avg. Similarity (negatives only): {avg_cosine_similarity:.4f}\n")
    
    print(f"Epoch {epoch + 1}: Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}, Avg. Similarity (negatives only): {avg_cosine_similarity:.4f}")

    output_answers.append(output_data)

    output_filename = f"{model_name.replace('/', '_')}_probe_{req_grad}_outputs_2500.json"
    with open(output_filename, "w") as json_file:
        json.dump(output_answers, json_file, indent=4)

print(f"Results saved to {output_filename}")