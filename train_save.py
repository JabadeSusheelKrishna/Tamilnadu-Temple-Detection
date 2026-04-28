import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class TempleMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=15):
        super(TempleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TempleDataset(Dataset):
    def __init__(self, root_dir, temple_names, processor):
        self.data = []
        self.temple_to_idx = {name: i for i, name in enumerate(temple_names)}
        
        for temple_name in temple_names:
            folder_path = os.path.join(root_dir, temple_name)
            if not os.path.exists(folder_path):
                continue
            
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.data.append((os.path.join(folder_path, img_name), self.temple_to_idx[temple_name]))
        
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load metadata and temple names
    with open('temples_metadata.json', 'r') as f:
        metadata = json.load(f)
    temple_names = list(metadata.keys())

    # Load CLIP
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval() # Freeze CLIP

    # Dataset
    dataset = TempleDataset("sample_images", temple_names, clip_processor)
    print(f"Found {len(dataset)} images across {len(temple_names)} temples.")
    
    # Pre-compute embeddings to speed up training
    embeddings = []
    labels = []
    
    print("Extracting CLIP embeddings...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            outputs = clip_model.get_image_features(**inputs)
            # Ensure we get the 512-dim projection
            if hasattr(outputs, "image_embeds"):
                img_emb = outputs.image_embeds
            elif torch.is_tensor(outputs):
                img_emb = outputs
            else:
                img_emb = outputs[0]
            
            # If still has sequence dim, take CLS token or pool
            if len(img_emb.shape) == 3:
                img_emb = img_emb[:, 0, :]
                
            # Normalize
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(img_emb.cpu())
            labels.append(label)
    
    X = torch.cat(embeddings)
    y = torch.tensor(labels)

    # MLP Training
    mlp = TempleMLP(input_dim=768, hidden_dim=256, output_dim=len(temple_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    print("Starting MLP training...")
    mlp.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = mlp(X.to(device))
        loss = criterion(outputs, y.to(device))
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y.to(device)).sum().item() / y.size(0)
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    # Save model and mapping
    torch.save({
        'state_dict': mlp.state_dict(),
        'temple_names': temple_names,
        'input_dim': 768,
        'hidden_dim': 256,
        'output_dim': len(temple_names)
    }, 'temple_mlp.pt')
    
    print("Model saved to temple_mlp.pt")

if __name__ == "__main__":
    train()
