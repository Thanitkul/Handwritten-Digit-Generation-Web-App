import torch
model = torch.load("models/generator.pth", map_location="cpu")  # or try loading into class
print("Model loaded successfully")
