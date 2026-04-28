import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageFilter
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2
import json
import os

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

class TempleClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        with open('temples_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.temple_names = list(self.metadata.keys())
        self.negative_class = "A photo of a person, animal, city street, or object that is not a temple"
        self.all_classes = self.temple_names + [self.negative_class]
        
        # Load Trained MLP if exists
        self.mlp = None
        self.mlp_path = 'temple_mlp.pt'
        if os.path.exists(self.mlp_path):
            try:
                checkpoint = torch.load(self.mlp_path, map_location=self.device)
                self.mlp = TempleMLP(
                    input_dim=checkpoint.get('input_dim', 768),
                    hidden_dim=checkpoint.get('hidden_dim', 256),
                    output_dim=checkpoint.get('output_dim', len(self.temple_names))
                ).to(self.device)
                self.mlp.load_state_dict(checkpoint['state_dict'])
                self.mlp.eval()
                print("Loaded trained MLP model successfully.")
            except Exception as e:
                print(f"Error loading MLP: {e}")

    def get_probabilities(self, image, text_queries):
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs.detach().cpu().numpy()[0]

    def predict_v1(self, image):
        """Version 1: Baseline Zero-Shot with Temple Names"""
        queries = self.all_classes
        probs = self.get_probabilities(image, queries)
        return self._format_result(probs)

    def predict_v2(self, image):
        """Version 2: Prompt Engineering"""
        queries = [f"A majestic photo of the {name}, a famous Hindu temple in Tamil Nadu, India" for name in self.temple_names]
        queries.append(self.negative_class)
        probs = self.get_probabilities(image, queries)
        return self._format_result(probs)

    def predict_v3(self, image):
        """Version 3: Region of Interest (ROI) Focus"""
        width, height = image.size
        left, top, right, bottom = width * 0.15, height * 0.15, width * 0.85, height * 0.85
        roi_image = image.crop((left, top, right, bottom))
        queries = [f"Architectural details of {name} temple" for name in self.temple_names]
        queries.append(self.negative_class)
        probs = self.get_probabilities(roi_image, queries)
        return self._format_result(probs)

    def predict_v4(self, image):
        """Version 4: Image Enhancement"""
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(open_cv_image, -1, kernel)
        enhanced_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        queries = [f"High quality photo of {name}" for name in self.temple_names]
        queries.append(self.negative_class)
        probs = self.get_probabilities(enhanced_image, queries)
        return self._format_result(probs)

    def predict_v5(self, image):
        """Version 5: CLIP + Trained MLP Head"""
        if self.mlp is None:
            # Fallback to simulation if not trained
            return self._predict_v5_simulated(image)
            
        # 1. Extract CLIP Image Embedding
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, "image_embeds"):
                img_emb = outputs.image_embeds
            elif torch.is_tensor(outputs):
                img_emb = outputs
            else:
                img_emb = outputs[0]
            
            if len(img_emb.shape) == 3:
                img_emb = img_emb[:, 0, :]
                
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            
            # 2. Pass through MLP
            logits = self.mlp(img_emb)
            probs = logits.softmax(dim=1).detach().cpu().numpy()[0]
            
        # Since the MLP only has 15 classes (temples), we'll handle the negative class separately or check confidence
        results = []
        for i, prob in enumerate(probs):
            results.append({"name": self.temple_names[i], "prob": float(prob)})
        
        # Sort and return
        results = sorted(results, key=lambda x: x['prob'], reverse=True)
        top_result = results[0]
        
        # Threshold for "Not a Temple" (heuristic since MLP wasn't trained with negative class)
        # If top probability is very low, or we can use a separate CLIP check
        is_temple = top_result['prob'] > 0.4 
        
        return {
            "prediction": top_result['name'] if is_temple else "None",
            "confidence": top_result['prob'],
            "is_temple": is_temple,
            "all_probs": results
        }

    def _predict_v5_simulated(self, image):
        p1 = self.predict_v2(image)
        p2 = self.predict_v4(image)
        combined_probs = {}
        for res in p1['all_probs']: combined_probs[res['name']] = res['prob'] * 0.6
        for res in p2['all_probs']: combined_probs[res['name']] = combined_probs.get(res['name'], 0) + res['prob'] * 0.4
        sorted_probs = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        top_name, top_prob = sorted_probs[0]
        return {
            "prediction": top_name if top_name != self.negative_class else "None",
            "confidence": top_prob,
            "is_temple": top_name != self.negative_class,
            "all_probs": [{"name": k, "prob": float(v)} for k, v in sorted_probs]
        }

    def _format_result(self, probs):
        results = []
        for i, prob in enumerate(probs):
            results.append({"name": self.all_classes[i] if i < len(self.temple_names) else self.negative_class, "prob": float(prob)})
        results = sorted(results, key=lambda x: x['prob'], reverse=True)
        top_result = results[0]
        is_temple = top_result['name'] != self.negative_class
        return {
            "prediction": top_result['name'] if is_temple else "None",
            "confidence": top_result['prob'],
            "is_temple": is_temple,
            "all_probs": results
        }
