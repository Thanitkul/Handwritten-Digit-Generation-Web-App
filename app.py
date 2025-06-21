import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- Generator Model Definition ---
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0, bias=False),  # 1x1 → 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),          # 7x7 → 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),            # 14x14 → 28x28
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.label_emb(labels)               # shape: [B, latent_dim]
        x = z * label_emb                                # element-wise mult
        x = x.view(x.size(0), x.size(1), 1, 1)           # reshape for conv input
        return self.model(x)


# --- App Settings ---
st.set_page_config(page_title="MNIST Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("Pick a digit (0–9) and view 5 generated samples using a GAN.")

# --- Top-of-page Digit Selector ---
digit = st.selectbox("Select Digit (0–9)", list(range(10)), index=0)

# --- Load Generator Model ---
@st.cache_resource
def load_generator():
    latent_dim = 100
    num_classes = 10
    model = Generator(latent_dim, num_classes)
    model.load_state_dict(torch.load("models/generator.pth", map_location="cpu"))
    model.eval()
    return model

G = load_generator()
latent_dim = 100

# --- Generate Images ---
z = torch.randn(5, latent_dim)
labels = torch.tensor([digit] * 5)
samples = G(z, labels)

# --- Show Images ---
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(samples[i][0].detach().numpy(), cmap="gray")
    axes[i].axis("off")
st.pyplot(fig)
