from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import joblib
import numpy as np
import io
import traceback
import pandas as pd   # ← VERY IMPORTANT - was missing!

app = FastAPI(title="Smart Rice Farming - Disease + Fertilizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== PATHS & CONFIG ===================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

DISEASE_MODEL_PATH = BASE_DIR / "models" / "model.pth"
DISEASE_CLASSES_PATH = BASE_DIR / "classes.txt"
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # back to auto

# Fertilizer config
PPO_MODEL_PATH = BASE_DIR / "models" / "ppo_dap_checkjach_best.pt"
STATE_DIM = 14
ACTION_DIM = 3
FERT_MIN = torch.tensor([0.0, 0.5, 0.4], device=DEVICE)
FERT_MAX = torch.tensor([3.3, 4.4, 3.4], device=DEVICE)

FEATURE_ORDER = [
    "ph", "organic_matter", "total_nitrogen",
    "p2o5", "potassium", "boron", "zinc",
    "sand", "clay", "slit",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos"
]

# =================== LOAD DISEASE MODEL (already working) ===================
with open(DISEASE_CLASSES_PATH, "r") as f:
    DISEASE_CLASSES = [line.strip() for line in f if line.strip()]

disease_model = timm.create_model(
    "vit_tiny_patch16_224", pretrained=False, num_classes=len(DISEASE_CLASSES)
)
state = torch.load(DISEASE_MODEL_PATH, map_location=DEVICE, weights_only=False)
disease_model.load_state_dict(state, strict=False)
disease_model.to(DEVICE)
disease_model.eval()

disease_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =================== LOAD PPO ===================
class PPOActor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        std = torch.exp(self.log_std)
        return mu, std

ppo_actor = PPOActor(STATE_DIM, ACTION_DIM).to(DEVICE)
ckpt = torch.load(PPO_MODEL_PATH, map_location=DEVICE, weights_only=False)
ppo_actor.load_state_dict(ckpt["actor_state_dict"])
ppo_actor.eval()

scaler_X = ckpt.get("scaler_X")
ppo_ready = scaler_X is not None
if not ppo_ready:
    print("!!! WARNING: PPO scaler_X missing → PPO predictions disabled !!!")

# =================== LOAD MLR & RFR ===================
mlr_pipeline = joblib.load(BASE_DIR / "models/mlr_fertilizer_pipeline.pkl")
rfr_pipeline = joblib.load(BASE_DIR / "models/fert_recommendation_rf_pipeline.joblib")

# =================== STATIC & UI ===================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

# =================== SCHEMAS & HELPERS ===================
class SoilInput(BaseModel):
    lat: float
    lon: float
    ph: float
    organic_matter: float
    total_nitrogen: float
    p2o5: float
    potassium: float
    boron: float
    zinc: float
    sand: float
    clay: float
    slit: float

def encode_latlon(lat: float, lon: float):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    return np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)

def prepare_soil_features(data: SoilInput):
    lat_sin, lat_cos, lon_sin, lon_cos = encode_latlon(data.lat, data.lon)
    row = {
        "ph": data.ph, "organic_matter": data.organic_matter,
        "total_nitrogen": data.total_nitrogen, "p2o5": data.p2o5,
        "potassium": data.potassium, "boron": data.boron, "zinc": data.zinc,
        "sand": data.sand, "clay": data.clay, "slit": data.slit,
        "lat_sin": lat_sin, "lat_cos": lat_cos, "lon_sin": lon_sin, "lon_cos": lon_cos
    }
    return pd.DataFrame([row])[FEATURE_ORDER]

# =================== ENDPOINTS ===================

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    # Your working disease endpoint - kept minimal
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_t = disease_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = disease_model(img_t)
        pred_idx = out.argmax(1).item()
        conf = torch.softmax(out, 1)[0, pred_idx].item()
    return {"disease": DISEASE_CLASSES[pred_idx], "confidence": round(conf, 4)}

@app.post("/predict/ppo")
def predict_ppo(data: SoilInput):
    if not ppo_ready:
        raise HTTPException(503, "PPO model not ready (missing scaler)")
    df = prepare_soil_features(data)
    X_scaled = scaler_X.transform(df.values)
    X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    with torch.no_grad():
        mu, _ = ppo_actor(X_tensor)
        action_scaled = torch.tanh(mu)
        action_real = FERT_MIN + (action_scaled + 1) / 2 * (FERT_MAX - FERT_MIN)
        action_real = torch.clamp(action_real, FERT_MIN, FERT_MAX)
    return {
        "UREA": round(float(action_real[0,0]), 2),
        "DAP": round(float(action_real[0,1]), 2),
        "MOP": round(float(action_real[0,2]), 2)
    }

@app.post("/predict/mlr")
def predict_mlr(data: SoilInput):
    df = prepare_soil_features(data)
    preds = mlr_pipeline.predict(df)[0]
    return {
        "UREA": round(max(float(preds[0]), 0), 2),
        "DAP": round(max(float(preds[1]), 0), 2),
        "MOP": round(max(float(preds[2]), 0), 2)
    }

@app.post("/predict/rfr")
def predict_rfr(data: SoilInput):
    df = prepare_soil_features(data)
    preds = rfr_pipeline.predict(df)[0]
    return {
        "UREA": round(max(float(preds[0]), 0), 2),
        "DAP": round(max(float(preds[1]), 0), 2),
        "MOP": round(max(float(preds[2]), 0), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)