import io
import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Auto-download weights from Hugging Face ──────────────────────────
WEIGHTS_URL  = "https://huggingface.co/abrarimtiyaz01/unet-weights/resolve/main/best_attention_unet.pth"
WEIGHTS_PATH = "best_attention_unet.pth"

if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("Downloading model weights (123 MB) — first launch only..."):
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)

# ── Attention U-Net Definition ───────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg   = nn.Sequential(nn.Conv2d(g_ch,    inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.Wx   = nn.Sequential(nn.Conv2d(x_ch,    inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi  = nn.Sequential(nn.Conv2d(inter_ch, 1,       1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g_up  = F.interpolate(self.Wg(g), size=x.shape[2:], mode='bilinear', align_corners=True)
        alpha = self.psi(self.relu(g_up + self.Wx(x)))
        return x * alpha


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        f = [64, 128, 256, 512, 1024]
        self.enc1 = DoubleConv(1, f[0]);  self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2]); self.enc4 = DoubleConv(f[2], f[3])
        self.bottleneck = DoubleConv(f[3], f[4])
        self.pool = nn.MaxPool2d(2)
        self.ag4 = AttentionGate(f[4], f[3], f[3]//2); self.up4 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2); self.dec4 = DoubleConv(f[4], f[3])
        self.ag3 = AttentionGate(f[3], f[2], f[2]//2); self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2); self.dec3 = DoubleConv(f[3], f[2])
        self.ag2 = AttentionGate(f[2], f[1], f[1]//2); self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2); self.dec2 = DoubleConv(f[2], f[1])
        self.ag1 = AttentionGate(f[1], f[0], f[0]//2); self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2); self.dec1 = DoubleConv(f[1], f[0])
        self.out  = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.ag4(b,  e4), self.up4(b)],  1))
        d3 = self.dec3(torch.cat([self.ag3(d4, e3), self.up3(d4)], 1))
        d2 = self.dec2(torch.cat([self.ag2(d3, e2), self.up2(d3)], 1))
        d1 = self.dec1(torch.cat([self.ag1(d2, e1), self.up1(d2)], 1))
        return self.out(d1)


IMAGE_SIZE = 256

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model  = AttentionUNet().to(device)
    state  = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def preprocess(pil_img):
    img = pil_img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    t   = torch.tensor(arr[np.newaxis, np.newaxis], dtype=torch.float32)
    return t, arr


def predict_tta(model, device, tensor, threshold):
    """Test-Time Augmentation — averages 4 predictions for better accuracy."""
    augs    = [lambda x: x,
               lambda x: torch.flip(x, [3]),
               lambda x: torch.flip(x, [2]),
               lambda x: torch.rot90(x, 1, [2, 3])]
    de_augs = [lambda x: x,
               lambda x: torch.flip(x, [3]),
               lambda x: torch.flip(x, [2]),
               lambda x: torch.rot90(x, -1, [2, 3])]
    acc = torch.zeros_like(tensor)
    with torch.no_grad():
        for a, da in zip(augs, de_augs):
            logits = model(a(tensor.to(device)))
            acc   += da(torch.sigmoid(logits)).cpu()
    probs = (acc / 4).squeeze().numpy()
    return probs, (probs > threshold).astype(np.float32)


def make_overlay(gray, mask, alpha=0.45):
    rgb = np.stack([gray] * 3, axis=-1)
    ov  = rgb.copy()
    ov[:, :, 0] = np.clip(rgb[:, :, 0] + alpha * mask,        0, 1)
    ov[:, :, 1] = np.clip(rgb[:, :, 1] - alpha * mask * 0.6,  0, 1)
    ov[:, :, 2] = np.clip(rgb[:, :, 2] - alpha * mask * 0.6,  0, 1)
    return ov


# ── UI ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tumor Segmentation — Attention U-Net",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Ultrasound Tumor Segmentation")
st.caption(
    "Attention U-Net · 31.6M parameters · 256×256 · TTA · "
    "Precision 81.9% · Accuracy 95.6%  |  "
    "B.Tech Project — Dept. of ECE, IUST Kashmir · Supervisor: Dr. Shakeel Ah Malik"
)

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Prediction threshold", 0.1, 0.9, 0.54, 0.01,
        help="Optimal threshold found during training: 0.54. Lower = more sensitive (more recall), higher = more precise."
    )
    st.markdown("---")
    st.markdown("**Model:** Attention U-Net")
    st.markdown("**Parameters:** 31.6M")
    st.markdown("**Resolution:** 256×256")
    st.markdown("**TTA:** ✅ 4 augmentations averaged")
    st.markdown("**Accuracy:** 95.6%")
    st.markdown("**Precision:** 81.9%")
    st.markdown("---")
    st.warning(
        "This is a **research demo** trained on the BUSI dataset. "
        "Not a certified diagnostic tool. Always consult a qualified radiologist."
    )

uploaded = st.file_uploader(
    "Upload a breast ultrasound scan",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Best results with BUSI-style grayscale breast ultrasound images."
)

try:
    model, device = load_model()
    model_loaded  = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {e}")

if uploaded and model_loaded:
    pil_img = Image.open(io.BytesIO(uploaded.read()))

    with st.spinner("Running Attention U-Net with TTA..."):
        tensor, gray_arr = preprocess(pil_img)
        probs, mask      = predict_tta(model, device, tensor, threshold)
        overlay          = make_overlay(gray_arr, mask)

    tumor_pct = 100.0 * mask.sum() / mask.size
    max_prob  = probs.max()

    # Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📷 Input Scan")
        st.image(gray_arr, clamp=True, use_container_width=True)
    with col2:
        st.subheader("🎯 Predicted Mask")
        st.image(mask, clamp=True, use_container_width=True)
    with col3:
        st.subheader("🔴 Overlay")
        st.image(overlay, use_container_width=True)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tumor Coverage",    f"{tumor_pct:.2f}%")
    m2.metric("Max Probability",   f"{max_prob:.3f}")
    m3.metric("Threshold Used",    f"{threshold:.2f}")
    m4.metric("Model Accuracy",    "95.6%")

    if tumor_pct < 0.1:
        st.info("No significant tumor region detected. Try lowering the threshold in the sidebar.")
    elif tumor_pct > 30:
        st.warning("Large tumor region detected — please verify with a radiologist.")
    else:
        st.success(f"Tumor region detected covering {tumor_pct:.2f}% of the scan area.")

elif not uploaded:
    st.info("👆 Upload an ultrasound image above to run segmentation.")
