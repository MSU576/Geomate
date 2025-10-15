# app.py — GeoMate V2 (single-file)
# Save this as app.py in your HuggingFace Space (or local folder)

# 0) Page config (must be first Streamlit command)
import streamlit as st
st.set_page_config(page_title="GeoMate V2", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# 1) Standard imports
import os
import io
import json
import math
import base64
import tempfile
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import streamlit as st
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu

# Visualization & PDF
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Ensure icon fonts load (fix desktop icon display for option_menu)
st.markdown("""
<!-- Load icon fonts used by streamlit_option_menu -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Optional imports handled gracefully
try:
    import geemap
    import ee
    EE_AVAILABLE = True
except Exception:
    EE_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Groq client import — we will require key
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# 2) Secrets check (strict)
REQUIRED_SECRETS = ["GROQ_API_KEY", "SERVICE_ACCOUNT", "EARTHENGINE_TOKEN"]
missing = [s for s in REQUIRED_SECRETS if not os.environ.get(s)]
if missing:
    st.sidebar.error(f"Missing required secrets: {', '.join(missing)}. Please add these to your HF Space secrets.")
    st.error("Required secrets missing. Please set GROQ_API_KEY, SERVICE_ACCOUNT, and EARTH_ENGINE_KEY in Secrets and reload the app.")
    st.stop()

# If Groq lib missing, still stop because user requested Groq usage
if not GROQ_AVAILABLE:
    st.sidebar.error("Python package 'groq' not installed. Add it to requirements.txt and redeploy.")
    st.error("Missing required library 'groq'. Please add to requirements and redeploy.")
    st.stop()

# 3) Global constants & helper functions
MAX_SITES = 4


# ----------------------------
# Soil Recognizer Page (Integrated 6-Class ResNet18)
# ----------------------------
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import streamlit as st

# ----------------------------
# Load Soil Model (6 Classes)
# ----------------------------
@st.cache_resource
def load_soil_model(path="soil_best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 6)  # 6 soil classes

        # Load checkpoint
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"⚠️ Could not load soil model: {e}")
        return None, device

soil_model, device = load_soil_model()

# ----------------------------
# Soil Classes & Transform
# ----------------------------
SOIL_CLASSES = ["Clay", "Gravel", "Loam", "Peat", "Sand", "Silt"]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# ----------------------------
# Prediction Function
# ----------------------------
def predict_soil(img: Image.Image):
    if soil_model is None:
        return "Model not loaded", {}

    img = img.convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = soil_model(inp)
        probs = torch.softmax(logits[0], dim=0)

    top_idx = torch.argmax(probs).item()
    predicted_class = SOIL_CLASSES[top_idx]

    result = {SOIL_CLASSES[i]: float(probs[i]) for i in range(len(SOIL_CLASSES))}
    return predicted_class, result

# ----------------------------
# Soil Recognizer Page
# ----------------------------
def soil_recognizer_page():
    st.header("🖼️ Soil Recognizer (ResNet18)")

    site = st.session_state["sites"]  # your existing site getter
    if site is None:
        st.warning("⚠️ No active site selected. Please add or select a site from the sidebar.")
        return

    uploaded = st.file_uploader("Upload soil image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded soil image", use_column_width=True)

        predicted_class, confidence_scores = predict_soil(img)
        st.success(f"✅ Predicted: **{predicted_class}**")

        st.subheader("Confidence Scores")
        for cls, score in confidence_scores.items():
            st.write(f"{cls}: {score:.2%}")

        if st.button("Save to site"):
            # Save predicted soil class into Soil Profile field
            st.session_state["sites"][st.session_state["active_site"]]["Soil Class"] = predicted_class
            st.session_state["sites"][st.session_state["active_site"]]["Soil Recognizer Confidence"] = confidence_scores[predicted_class]
            save_active_site(site)
            st.success("Saved prediction to active site memory.")

# Pre-defined dropdown text mappings (as you requested) — exact text with mapping numbers for logic backend
DILATANCY_OPTIONS = [
    "1. Quick to slow",
    "2. None to very slow",
    "3. Slow",
    "4. Slow to none",
    "5. None",
    "6. Null?"
]
TOUGHNESS_OPTIONS = [
    "1. None",
    "2. Medium",
    "3. Slight?",
    "4. Slight to Medium?",
    "5. High",
    "6. Null?"
]
DRY_STRENGTH_OPTIONS = [
    "1. None to slight",
    "2. Medium to high",
    "3. Slight to Medium",
    "4. High to very high",
    "5. Null?"
]

# Map option text to numeric codes used in your USCS logic
DILATANCY_MAP = {DILATANCY_OPTIONS[i]: i+1 for i in range(len(DILATANCY_OPTIONS))}
TOUGHNESS_MAP = {TOUGHNESS_OPTIONS[i]: i+1 for i in range(len(TOUGHNESS_OPTIONS))}
DRY_STRENGTH_MAP = {DRY_STRENGTH_OPTIONS[i]: i+1 for i in range(len(DRY_STRENGTH_OPTIONS))}

# Engineering characteristics dictionary (expanded earlier; trimmed to representative entries but detailed)
ENGINEERING_CHARACTERISTICS = {
    "Gravel": {
        "Settlement": "None",
        "Quicksand": "Impossible",
        "Frost-heaving": "None",
        "Groundwater_lowering": "Possible",
        "Cement_grouting": "Possible",
        "Silicate_bitumen_injections": "Unsuitable",
        "Compressed_air": "Possible (see notes)"
    },
    "Coarse sand": {
        "Settlement": "None",
        "Quicksand": "Impossible",
        "Frost-heaving": "None",
        "Groundwater_lowering": "Possible",
        "Cement_grouting": "Possible only if very coarse",
        "Silicate_bitumen_injections": "Suitable",
        "Compressed_air": "Suitable"
    },
    "Medium sand": {
        "Settlement": "None",
        "Quicksand": "Unlikely",
        "Frost-heaving": "None",
        "Groundwater_lowering": "Suitable",
        "Cement_grouting": "Impossible",
        "Silicate_bitumen_injections": "Suitable",
        "Compressed_air": "Suitable"
    },
    "Fine sand": {
        "Settlement": "None",
        "Quicksand": "Liable",
        "Frost-heaving": "None",
        "Groundwater_lowering": "Suitable",
        "Cement_grouting": "Impossible",
        "Silicate_bitumen_injections": "Not possible in very fine sands",
        "Compressed_air": "Suitable"
    },
    "Silt": {
        "Settlement": "Occurs",
        "Quicksand": "Liable (very coarse silts may behave differently)",
        "Frost-heaving": "Occurs",
        "Groundwater_lowering": "Generally not suitable (electro-osmosis possible)",
        "Cement_grouting": "Impossible",
        "Silicate_bitumen_injections": "Impossible",
        "Compressed_air": "Suitable"
    },
    "Clay": {
        "Settlement": "Occurs",
        "Quicksand": "Impossible",
        "Frost-heaving": "None",
        "Groundwater_lowering": "Impossible (generally)",
        "Cement_grouting": "Only in stiff fissured clay",
        "Silicate_bitumen_injections": "Impossible",
        "Compressed_air": "Used for support only in special cases"
    }
}

# USCS & AASHTO verbatim logic (function)
from math import floor

def classify_uscs_aashto(inputs: Dict[str, Any]) -> Tuple[str, str, int, Dict[str, str], str]:
    """
    Verbatim USCS + AASHTO classifier based on the logic you supplied.
    inputs: dictionary expected keys:
      opt: 'y' or 'n'
      P2 (float): % passing #200 (0.075 mm)
      P4 (float): % passing #4 (4.75 mm)
      D60, D30, D10 (float mm) - can be 0 if unknown
      LL, PL (float)
      nDS, nDIL, nTG (int) mapped from dropdowns
    Returns:
      result_text (markdown), aashto_str, GI, engineering_characteristics (dict), uscs_str
    """
    opt = str(inputs.get("opt","n")).lower()
    if opt == 'y':
        uscs = "Pt"
        uscs_expl = "Peat / organic soil — compressible, high organic content; poor engineering properties for load-bearing without special treatment."
        aashto = "Organic (special handling)"
        GI = 0
        chars = {"summary":"Highly organic peat — large settlement, low strength, not suitable for foundations without improvement."}
        res_text = f"According to USCS, the soil is **{uscs}** — {uscs_expl}\n\nAccording to AASHTO, the soil is **{aashto}**."
        return res_text, aashto, GI, chars, uscs

    # parse numeric inputs with defaults
    P2 = float(inputs.get("P2", 0.0))
    P4 = float(inputs.get("P4", 0.0))
    D60 = float(inputs.get("D60", 0.0))
    D30 = float(inputs.get("D30", 0.0))
    D10 = float(inputs.get("D10", 0.0))
    LL = float(inputs.get("LL", 0.0))
    PL = float(inputs.get("PL", 0.0))
    PI = LL - PL if (LL is not None and PL is not None) else 0.0

    Cu = (D60 / D10) if (D10 > 0 and D60 > 0) else 0.0
    Cc = ((D30 ** 2) / (D10 * D60)) if (D10 > 0 and D30 > 0 and D60 > 0) else 0.0

    uscs = "Unknown"
    uscs_expl = ""
    if P2 <= 50:
        # Coarse-Grained Soils
        if P4 <= 50:
            # Gravels
            if Cu != 0 and Cc != 0:
                if Cu >= 4 and 1 <= Cc <= 3:
                    uscs = "GW"; uscs_expl = "Well-graded gravel (good engineering properties, high strength, good drainage)."
                else:
                    uscs = "GP"; uscs_expl = "Poorly-graded gravel (less favorable gradation)."
            else:
                if PI < 4 or PI < 0.73 * (LL - 20):
                    uscs = "GM"; uscs_expl = "Silty gravel (fines may reduce permeability and strength)."
                elif PI > 7 and PI > 0.73 * (LL - 20):
                    uscs = "GC"; uscs_expl = "Clayey gravel (clayey fines increase plasticity)."
                else:
                    uscs = "GM-GC"; uscs_expl = "Gravel with mixed silt/clay fines."
        else:
            # Sands
            if Cu != 0 and Cc != 0:
                if Cu >= 6 and 1 <= Cc <= 3:
                    uscs = "SW"; uscs_expl = "Well-graded sand (good compaction and drainage)."
                else:
                    uscs = "SP"; uscs_expl = "Poorly-graded sand (uniform or gap-graded)."
            else:
                if PI < 4 or PI <= 0.73 * (LL - 20):
                    uscs = "SM"; uscs_expl = "Silty sand (fines are low-plasticity silt)."
                elif PI > 7 and PI > 0.73 * (LL - 20):
                    uscs = "SC"; uscs_expl = "Clayey sand (clayey fines present; higher plasticity)."
                else:
                    uscs = "SM-SC"; uscs_expl = "Transition between silty sand and clayey sand."
    else:
        # Fine-Grained Soils
        nDS = int(inputs.get("nDS", 5))
        nDIL = int(inputs.get("nDIL", 6))
        nTG = int(inputs.get("nTG", 6))
        if LL < 50:
            if 20 <= LL < 50 and PI <= 0.73 * (LL - 20):
                if nDS == 1 or nDIL == 3 or nTG == 3:
                    uscs = "ML"; uscs_expl = "Silt (low plasticity)."
                elif nDS == 3 or nDIL == 3 or nTG == 3:
                    uscs = "OL"; uscs_expl = "Organic silt (low plasticity)."
                else:
                    uscs = "ML-OL"; uscs_expl = "Mixed silt/organic silt."
            elif 10 <= LL <= 30 and 4 <= PI <= 7 and PI > 0.72 * (LL - 20):
                if nDS == 1 or nDIL == 1 or nTG == 1:
                    uscs = "ML"; uscs_expl = "Silt"
                elif nDS == 2 or nDIL == 2 or nTG == 2:
                    uscs = "CL"; uscs_expl = "Clay (low plasticity)."
                else:
                    uscs = "ML-CL"; uscs_expl = "Mixed silt/clay"
            else:
                uscs = "CL"; uscs_expl = "Clay (low plasticity)."
        else:
            if PI < 0.73 * (LL - 20):
                if nDS == 3 or nDIL == 4 or nTG == 4:
                    uscs = "MH"; uscs_expl = "Silt (high plasticity)"
                elif nDS == 2 or nDIL == 2 or nTG == 4:
                    uscs = "OH"; uscs_expl = "Organic silt/clay (high plasticity)"
                else:
                    uscs = "MH-OH"; uscs_expl = "Mixed high-plasticity silt/organic"
            else:
                uscs = "CH"; uscs_expl = "Clay (high plasticity)"

    # === AASHTO (verbatim) ===
    if P2 <= 35:
        if P2 <= 15 and P4 <= 30 and PI <= 6:
            aashto = "A-1-a"
        elif P2 <= 25 and P4 <= 50 and PI <= 6:
            aashto = "A-1-b"
        elif P2 <= 35 and P4 > 0:
            if LL <= 40 and PI <= 10:
                aashto = "A-2-4"
            elif LL >= 41 and PI <= 10:
                aashto = "A-2-5"
            elif LL <= 40 and PI >= 11:
                aashto = "A-2-6"
            elif LL >= 41 and PI >= 11:
                aashto = "A-2-7"
            else:
                aashto = "A-2"
        else:
            aashto = "A-3"
    else:
        if LL <= 40 and PI <= 10:
            aashto = "A-4"
        elif LL >= 41 and PI <= 10:
            aashto = "A-5"
        elif LL <= 40 and PI >= 11:
            aashto = "A-6"
        else:
            aashto = "A-7-5" if PI <= (LL - 30) else "A-7-6"

    # Group Index
    a = P2 - 35
    a = 0 if a < 0 else (40 if a > 40 else a)
    b = P2 - 15
    b = 0 if b < 0 else (40 if b > 40 else b)
    c = LL - 40
    c = 0 if c < 0 else (20 if c > 20 else c)
    d = PI - 10
    d = 0 if d < 0 else (20 if d > 20 else d)
    GI = floor(0.2 * a + 0.005 * a * c + 0.01 * b * d)

    aashto_expl = f"{aashto} (Group Index = {GI})"

    # engineering characteristics pick
    char_summary = {}
    found_key = None
    for key in ENGINEERING_CHARACTERISTICS:
        if key.lower() in uscs.lower() or key.lower() in uscs_expl.lower():
            found_key = key
            break
    if found_key:
        char_summary = ENGINEERING_CHARACTERISTICS[found_key]
    else:
        # fallback selection by starting letter
        if uscs.startswith("G") or uscs.startswith("S"):
            char_summary = ENGINEERING_CHARACTERISTICS.get("Coarse sand", {})
        else:
            char_summary = ENGINEERING_CHARACTERISTICS.get("Silt", {})

    res_text_lines = [
        f"According to USCS, the soil is **{uscs}** — {uscs_expl}",
        f"According to AASHTO, the soil is **{aashto_expl}**",
        "",
        "Engineering characteristics (summary):"
    ]
    for k,v in char_summary.items():
        res_text_lines.append(f"- **{k}**: {v}")

    result_text = "\n".join(res_text_lines)
    return result_text, aashto_expl, GI, char_summary, uscs

# Helper: GSD interpolation to find diameters D10,D30,D60
def compute_gsd_metrics(diams: List[float], passing: List[float]) -> Dict[str, float]:
    """
    diams: list of diameters in mm (descending)
    passing: corresponding % passing (0-100)
    returns D10, D30, D60, Cu, Cc
    """
    # ensure descending diam, convert to float arrays
    if len(diams) < 2 or len(diams) != len(passing):
        raise ValueError("Diameters and passing arrays must match and have at least 2 items.")
    # linear interpolation on log(d)
    import numpy as np
    d = np.array(diams)
    p = np.array(passing)
    # make sure p is decreasing or increasing? passing decreases as diameter decreases, but we will handle general interpolation by sorting by diameter descending
    order = np.argsort(-d)
    d = d[order]
    p = p[order]
    # drop duplicates etc
    # function to find Dx = diameter at which passing = x (percent)
    def find_D(x):
        if x <= p.min():
            return float(d[p.argmin()])
        if x >= p.max():
            return float(d[p.argmax()])
        # linear interpolation on p vs log(d)
        from math import log, exp
        ld = np.log(d)
        # interpolate ld as function of p
        ld_interp = np.interp(x, p[::-1], ld[::-1])  # reverse because interp expects ascending x
        return float(math.exp(ld_interp))
    D10 = find_D(10.0)
    D30 = find_D(30.0)
    D60 = find_D(60.0)
    Cu = D60 / D10 if D10 > 0 else 0.0
    Cc = (D30 ** 2) / (D10 * D60) if (D10 > 0 and D60 > 0) else 0.0
    return {"D10":D10, "D30":D30, "D60":D60, "Cu":Cu, "Cc":Cc}



import streamlit as st
import os
import pandas as pd

# -------------------
# 1) Session state initialization
# -------------------
if "sites" not in st.session_state:
    st.session_state["sites"] = [
        {
            # ---------------------------
            # Basic Project Info
            # ---------------------------
            "Site Name": "Default Site",
            "Project Name": "Project - Default Site",
            "Site ID": 0,
            "Coordinates": "",
            "lat": None,
            "lon": None,
            "Project Description": "",

            # ---------------------------
            # Soil Recognition / Classifier
            # ---------------------------
            "Soil Class": None,
            "Soil Recognizer Confidence": None,
            "USCS": None,
            "AASHTO": None,
            "GI": None,
            "GSD": None,
            "classifier_inputs": {},
            "classifier_decision": None,

            # ---------------------------
            # Site Characterization
            # ---------------------------
            "Topography": None,
            "Drainage": None,
            "Current Land Use": None,
            "Regional Geology": None,

            # ---------------------------
            # Investigations & Lab
            # ---------------------------
            "Field Investigation": [],
            "Laboratory Results": [],

            # ---------------------------
            # Geotechnical Parameters
            # ---------------------------
            "Load Bearing Capacity": None,
            "Skin Shear Strength": None,
            "Relative Compaction": None,
            "Rate of Consolidation": None,
            "Nature of Construction": None,
            "Borehole Count": None,
            "Max Depth (m)": None,
            "SPT N (avg)": None,
            "CBR (%)": None,
            "Allowable Bearing (kPa)": None,

            # ---------------------------
            # Earth Engine Data
            # ---------------------------
            "Soil Profile": {
                "Clay": None,
                "Sand": None,
                "Silt": None,
                "OrganicCarbon": None,
                "pH": None
            },
            "Topo Data": None,
            "Seismic Data": None,
            "Flood Data": None,
            "Environmental Data": {
                "Landcover Stats": None,
                "Forest Loss": None,
                "Urban Fraction": None
            },
            "Weather Data": {
                "Rainfall": None,
                "Temperature": None,
                "Humidity": None
            },
            "Atmospheric Data": {
                "AerosolOpticalDepth": None,
                "NO2": None,
                "CO": None,
                "PM2.5": None
            },

            # ---------------------------
            # Map & Visualization
            # ---------------------------
            "ROI": None,
            "roi_coords": None,
            "map_snapshot": None,

            # ---------------------------
            # AI / Reporting
            # ---------------------------
            "chat_history": [],
            "LLM_Report_Text": None,
            "report_convo_state": 0,
            "report_missing_fields": [],
            "report_answers": {}
        }
    ]

# -------------------
# Session state defaults
# -------------------
if "active_site" not in st.session_state:
    st.session_state["active_site"] = 0

if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "groq/compound"


# -------------------
# API Keys
# -------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEM_API_KEY = None
if "GEM_API" in st.secrets:
    GEM_API_KEY = st.secrets["GEM_API"]


# -------------------
# Universal LLM Call
# -------------------
def llm_generate(prompt: str, model: str = None, max_tokens: int = 512) -> str:
    """Universal LLM call for Groq, Gemini, DeepSeek."""
    model_name = model or st.session_state["llm_model"]

    try:
        # ------------------- GEMINI -------------------
        if model_name.lower().startswith("gemini"):
            import requests
            if not GEM_API_KEY:
                return "[LLM error: No GEM_API key found in secrets]"

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={GEM_API_KEY}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": max_tokens
                }
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # ------------------- GROQ / DEEPSEEK / LLAMA -------------------
        elif "groq" in model_name or "llama" in model_name or "deepseek" in model_name:
            from groq import Groq
            if not GROQ_API_KEY:
                return "[LLM error: No GROQ_API key found in secrets/env]"

            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content

        else:
            return f"[LLM error: Unknown model {model_name}]"

    except Exception as e:
        return f"[LLM error: {e}]"


# -------------------
# UI helper: CSS
# -------------------
st.markdown("""
<style>
/* Background and card styling */
body { background: #0b0b0b; color: #e9eef6; }
.stApp > .main > .block-container { padding-top: 18px; }

/* Landing and cards */
.gm-card { background: linear-gradient(180deg, rgba(255,122,0,0.04), rgba(255,122,0,0.02));
           border-radius:12px; padding:14px; border:1px solid rgba(255,122,0,0.06);}
.gm-cta { background: linear-gradient(90deg,#ff7a00,#ff3a3a); color:white;
          padding:10px 14px; border-radius:10px; font-weight:700; }

/* Chat bubbles */
.chat-bot { background: #0f1720; border-left:4px solid #FF7A00;
            padding:10px 12px; border-radius:12px; margin:6px 0; color:#e9eef6; }
.chat-user { background: #1a1f27; padding:10px 12px; border-radius:12px;
             margin:6px 0; color:#cfe6ff; text-align:right;}
.small-muted { color:#9aa7bf; font-size:12px; }
</style>
""", unsafe_allow_html=True)


# -------------------
# Sidebar: Navigation + Model + Site Management
# -------------------
from streamlit_option_menu import option_menu

with st.sidebar:
    st.markdown("### ⚙️ LLM Model Selector")
    st.session_state["llm_model"] = st.selectbox(
        "Choose AI Model",
        [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.1-8b-instant",
            "meta-llama/llama-guard-4-12b",
            "llama-3.3-70b-versatile",
            "groq/compound",
            "deepseek-r1-distill-llama-70b",   # ✅ DeepSeek
            "gemini-1.5-pro",                  # ✅ Gemini Pro
            "gemini-1.5-flash",     # ✅ Gemini Flash
            "gemini-2.5-pro"
        ],
        index=0 if "llm_model" not in st.session_state else
        ["meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.1-8b-instant",
        "meta-llama/llama-guard-4-12b",
        "llama-3.3-70b-versatile",
        "groq/compound",
        "deepseek-r1-distill-llama-70b",   # ✅ DeepSeek
        "gemini-1.5-pro",                  # ✅ Gemini Pro
        "gemini-1.5-flash",  # ✅ Gemini Flash
        "gemini-2.5-pro"
        ].index(
            st.session_state.get("llm_model", "groq/compound")
        )
    )
    # -------------------
    # Site management
    # -------------------
    st.markdown("### Project Sites")

    site_names = [s.get("Site Name", f"Site {i}") for i, s in enumerate(st.session_state["sites"])]

    # Add new site
    new_site_name = st.text_input("New site name", value="", key="new_site_name_input")
    if st.button("➕ Add / Create Site"):
        if new_site_name.strip() == "":
            st.warning("Enter a name for the new site.")
        elif len(st.session_state["sites"]) >= MAX_SITES:
            st.error(f"Maximum of {MAX_SITES} sites allowed.")
        else:
            idx = len(st.session_state["sites"])
            st.session_state["sites"].append({
                "Site Name": new_site_name.strip(),
                "Project Name": "Project - " + new_site_name.strip(),
                "Site ID": idx,
                "Coordinates": "",
                "lat": None,
                "lon": None,
                "Project Description": "",
                "Soil Class": None,
                "Soil Recognizer Confidence": None,
                "USCS": None,
                "AASHTO": None,
                "GI": None,
                "GSD": None,
                "classifier_inputs": {},
                "classifier_decision": None,
                "Topography": None,
                "Drainage": None,
                "Current Land Use": None,
                "Regional Geology": None,
                "Field Investigation": [],
                "Laboratory Results": [],
                "Load Bearing Capacity": None,
                "Skin Shear Strength": None,
                "Relative Compaction": None,
                "Rate of Consolidation": None,
                "Nature of Construction": None,
                "Borehole Count": None,
                "Max Depth (m)": None,
                "SPT N (avg)": None,
                "CBR (%)": None,
                "Allowable Bearing (kPa)": None,
                "Soil Profile": {"Clay": None, "Sand": None, "Silt": None, "OrganicCarbon": None, "pH": None},
                "Topo Data": None,
                "Seismic Data": None,
                "Flood Data": None,
                "Environmental Data": {"Landcover Stats": None, "Forest Loss": None, "Urban Fraction": None},
                "Weather Data": {"Rainfall": None, "Temperature": None, "Humidity": None},
                "Atmospheric Data": {"AerosolOpticalDepth": None, "NO2": None, "CO": None, "PM2.5": None},
                "ROI": None,
                "roi_coords": None,
                "map_snapshot": None,
                "chat_history": [],
                "LLM_Report_Text": None,
                "report_convo_state": 0,
                "report_missing_fields": [],
                "report_answers": {}
            })
            st.success(f"Site '{new_site_name.strip()}' created.")
            st.session_state["active_site"] = idx
            st.rerun()

    # Active site selector
    if site_names:
        active_index = st.selectbox("Active Site", options=list(range(len(site_names))),
                                    format_func=lambda x: site_names[x],
                                    index=st.session_state["active_site"])
        st.session_state["active_site"] = active_index

    st.markdown("---")

    # Display site info as table instead of JSON
    active_site = st.session_state["sites"][st.session_state["active_site"]]
    df = pd.DataFrame(active_site.items(), columns=["Field", "Value"])
    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("© GeoMate • Advanced geotechnical copilot", unsafe_allow_html=True)

# 7) Pages implementation
import streamlit as st
import os

# -------------------- SESSION INIT --------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Landing"
if "sites" not in st.session_state:
    st.session_state["sites"] = [{"Site Name": "Default Site", "USCS": None, "AASHTO": None, "GSD": None}]
if "active_site" not in st.session_state:
    st.session_state["active_site"] = 0


# -------------------- LANDING PAGE --------------------
import streamlit as st

def landing_page():
    # Fixed gradient background (no slideshow)
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #1c1c1c 50%, #2e004f 100%);
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .gm-hero {
            padding: 48px 28px;
            border-radius: 16px;
            margin-bottom: 22px;
            background: linear-gradient(135deg, rgba(11,11,11,0.8) 0%, rgba(20,0,40,0.75) 100%);
            box-shadow: 0 8px 28px rgba(0,0,0,0.65);
            text-align: center;
            animation: fadeIn 1.8s ease-in-out;
        }
        .gm-hero h1 {
            font-size: 3.5rem;
            font-weight: 900;
            margin: 0;
            background: linear-gradient(90deg,#FF8C00,#a020f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .gm-hero p {
            font-size: 1.2rem;
            margin-top: 10px;
            color: #e0e0e0;
        }
        .gm-card {
            background: rgba(25,25,25,0.85);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.5);
        }
        .quick-btn {
            display: inline-block;
            margin: 6px;
            padding: 12px 24px;
            border-radius: 50px;
            background: linear-gradient(90deg, #ff4d00, #a020f0);
            color: white !important;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .quick-btn:hover {
            background: linear-gradient(90deg, #a020f0, #ff4d00);
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        }
        @keyframes fadeIn {
            from { opacity:0; transform: translateY(40px);}
            to { opacity:1; transform: translateY(0);}
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div class="gm-hero">
            <h1>GeoMate V2</h1>
            <p>AI Geotechnical Copilot — soil recognition, classification, 
            Earth Engine locator, RAG-powered Q&A, OCR, and dynamic reports.</p>
        </div>
    """, unsafe_allow_html=True)

    # Content layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='gm-card'>", unsafe_allow_html=True)
        st.write(
            "GeoMate helps geotechnical engineers: classify soils (USCS/AASHTO), "
            "plot grain size distributions (GSD), fetch Earth Engine data, chat with a RAG-backed LLM, "
            "run OCR on site logs, and generate professional reports."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🚀 Quick Actions")
        c1, c2, c3 = st.columns(3)
        if c1.button("🧪 Classifier"):
            st.session_state["page"] = "Classifier"; st.rerun()
        if c2.button("📈 Soil Recognizer"):
            st.session_state["page"] = "Soil recognizer"; st.rerun()
        if c3.button("🌍 Locator"):
            st.session_state["page"] = "Locator"; st.rerun()

        c4, c5, c6 = st.columns(3)
        if c4.button("🤖 Ask GeoMate"):
            st.session_state["page"] = "RAG"; st.rerun()
        if c5.button("📷 OCR"):
            st.session_state["page"] = "OCR"; st.rerun()
        if c6.button("📑 Reports"):
            st.session_state["page"] = "Reports"; st.rerun()

    with col2:
        st.markdown("<div class='gm-card' style='text-align:center'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#FF8C00'>📊 Live Site Summary</h3>", unsafe_allow_html=True)
        if "sites" in st.session_state and st.session_state.get("active_site") is not None:
            site = st.session_state["sites"][st.session_state["active_site"]]
            st.write(f"**Site:** {site.get('Site Name','N/A')}")
            st.write(f"USCS: {site.get('USCS','-')} | AASHTO: {site.get('AASHTO','-')}")
            st.write(f"GSD Saved: {'✅' if site.get('GSD') else '❌'}")
        else:
            st.info("No active site selected.")
        st.markdown("</div>", unsafe_allow_html=True)

    
# Soil Classifier page (conversational, step-by-step)
def soil_classifier_page():
    st.header("🧪 Soil Classifier — Conversational (USCS & AASHTO)")
    site = st.session_state["sites"][st.session_state["active_site"]]

    # conversation state machine: steps list
    steps = [
        {"id":"intro", "bot":"Hello — I am the GeoMate Soil Classifier. Ready to start?"},
        {"id":"organic", "bot":"Is the soil at this site organic (contains high organic matter, feels spongy or has odour)?", "type":"choice", "choices":["No","Yes"]},
        {"id":"P2", "bot":"Please enter the percentage passing the #200 sieve (0.075 mm). Example: 12", "type":"number"},
        {"id":"P4", "bot":"What is the percentage passing the sieve no. 4 (4.75 mm)? (enter 0 if unknown)", "type":"number"},
        {"id":"hasD", "bot":"Do you know the D10, D30 and D60 diameters (in mm)?", "type":"choice","choices":["No","Yes"]},
        {"id":"D60", "bot":"Enter D60 (diameter in mm corresponding to 60% passing).", "type":"number"},
        {"id":"D30", "bot":"Enter D30 (diameter in mm corresponding to 30% passing).", "type":"number"},
        {"id":"D10", "bot":"Enter D10 (diameter in mm corresponding to 10% passing).", "type":"number"},
        {"id":"LL", "bot":"What is the liquid limit (LL)?", "type":"number"},
        {"id":"PL", "bot":"What is the plastic limit (PL)?", "type":"number"},
        {"id":"dry", "bot":"Select the observed dry strength of the fine soil (if applicable).", "type":"select", "options":DRY_STRENGTH_OPTIONS},
        {"id":"dilat", "bot":"Select the observed dilatancy behaviour.", "type":"select", "options":DILATANCY_OPTIONS},
        {"id":"tough", "bot":"Select the observed toughness.", "type":"select", "options":TOUGHNESS_OPTIONS},
        {"id":"confirm", "bot":"Would you like me to classify now?", "type":"choice", "choices":["No","Yes"]}
    ]

    if "classifier_step" not in st.session_state:
        st.session_state["classifier_step"] = 0
    if "classifier_inputs" not in st.session_state:
        st.session_state["classifier_inputs"] = dict(site.get("classifier_inputs", {}))

    step_idx = st.session_state["classifier_step"]

    # chat history display
    st.markdown("<div class='gm-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-bot'>{}</div>".format("GeoMate: Hello — soil classifier ready. Use the controls below to answer step-by-step."), unsafe_allow_html=True)
    # Show stored user answers sequentially for context
    # render question up to current step
    for i in range(step_idx+1):
        s = steps[i]
        # show bot prompt
        st.markdown(f"<div class='chat-bot'>{s['bot']}</div>", unsafe_allow_html=True)
        # show user answer if exists in classifier_inputs
        key = s["id"]
        val = st.session_state["classifier_inputs"].get(key)
        if val is not None:
            st.markdown(f"<div class='chat-user'>{val}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Render input widget for current step
    current = steps[step_idx]
    step_id = current["id"]
    proceed = False
    user_answer = None

    cols = st.columns([1,1,1])
    with cols[0]:
        if current.get("type") == "choice":
            choice = st.radio(current["bot"], options=current["choices"], index=0, key=f"cls_{step_id}")
            user_answer = choice
        elif current.get("type") == "number":
            # numeric input without +/- spinner (we use text_input and validate)
            raw = st.text_input(current["bot"], value=str(st.session_state["classifier_inputs"].get(step_id,"")), key=f"cls_{step_id}_num")
            # validate numeric
            try:
                if raw.strip() == "":
                    user_answer = None
                else:
                    user_answer = float(raw)
            except:
                st.warning("Please enter a valid number (e.g., 12 or 0).")
                user_answer = None
        elif current.get("type") == "select":
            opts = current.get("options", [])
            sel = st.selectbox(current["bot"], options=opts, index=0, key=f"cls_{step_id}_sel")
            user_answer = sel
        else:
            # just a message step — proceed
            user_answer = None

    # controls: Next / Back
    coln, colb, colsave = st.columns([1,1,1])
    with coln:
        if st.button("➡️ Next", key=f"next_{step_id}"):
            # store answer if provided
            if current.get("type") == "number":
                if user_answer is None:
                    st.warning("Please enter a numeric value or enter 0 if unknown.")
                else:
                    st.session_state["classifier_inputs"][step_id] = user_answer
                    st.session_state["classifier_step"] = min(step_idx+1, len(steps)-1)
                    st.rerun()
            elif current.get("type") in ("choice","select"):
                st.session_state["classifier_inputs"][step_id] = user_answer
                st.session_state["classifier_step"] = min(step_idx+1, len(steps)-1)
                st.rerun()
            else:
                # message-only step
                st.session_state["classifier_step"] = min(step_idx+1, len(steps)-1)
                st.rerun()
    with colb:
        if st.button("⬅️ Back", key=f"back_{step_id}"):
            st.session_state["classifier_step"] = max(0, step_idx-1)
            st.rerun()
    with colsave:
        if st.button("💾 Save & Classify now", key="save_and_classify"):
            # prepare inputs in required format for classify_uscs_aashto
            ci = st.session_state["classifier_inputs"].copy()
            # Normalize choices into expected codes
            if isinstance(ci.get("dry"), str):
                ci["nDS"] = DRY_STRENGTH_MAP.get(ci.get("dry"), 5)
            if isinstance(ci.get("dilat"), str):
                ci["nDIL"] = DILATANCY_MAP.get(ci.get("dilat"), 6)
            if isinstance(ci.get("tough"), str):
                ci["nTG"] = TOUGHNESS_MAP.get(ci.get("tough"), 6)
            # map 'Yes'/'No' for organic and hasD
            ci["opt"] = "y" if ci.get("organic","No")=="Yes" or ci.get("organic",ci.get("organic"))=="Yes" else ci.get("organic","n")
            # our field names in CI may differ: convert organic stored under 'organic' step to 'opt'
            if "organic" in ci:
                ci["opt"] = "y" if ci["organic"]=="Yes" else "n"
            # map D entries: D60 etc may be present
            # call classification
            try:
                res_text, aashto, GI, chars, uscs = classify_uscs_aashto(ci)
            except Exception as e:
                st.error(f"Classification error: {e}")
                res_text = f"Error during classification: {e}"
                aashto = "N/A"; GI = 0; chars = {}; uscs = "N/A"
            # save into active site
            site["USCS"] = uscs
            site["AASHTO"] = aashto
            site["GI"] = GI
            site["classifier_inputs"] = ci
            site["classifier_decision"] = res_text
            st.success("Classification complete. Results saved to site.")
            st.write("### Classification Results")
            st.markdown(res_text)
            # Keep classifier_step at end so user can review
            st.session_state["classifier_step"] = len(steps)-1

# GSD Curve Page
def gsd_page():
    st.header("📈 Grain Size Distribution (GSD) Curve")
    site = st.session_state["sites"][st.session_state["active_site"]]
    st.markdown("Enter diameters (mm) and % passing (comma-separated). Use descending diameters (largest to smallest).")
    diam_input = st.text_area("Diameters (mm) comma-separated", value=site.get("GSD",{}).get("diameters","75,50,37.5,25,19,12.5,9.5,4.75,2,0.85,0.425,0.25,0.18,0.15,0.075") if site.get("GSD") else "75,50,37.5,25,19,12.5,9.5,4.75,2,0.85,0.425,0.25,0.18,0.15,0.075")
    pass_input = st.text_area("% Passing comma-separated", value=site.get("GSD",{}).get("passing","100,98,96,90,85,78,72,65,55,45,35,25,18,14,8") if site.get("GSD") else "100,98,96,90,85,78,72,65,55,45,35,25,18,14,8")
    if st.button("Compute GSD & Save"):
        try:
            diams = [float(x.strip()) for x in diam_input.split(",") if x.strip()]
            passing = [float(x.strip()) for x in pass_input.split(",") if x.strip()]
            metrics = compute_gsd_metrics(diams, passing)
            # plot
            fig, ax = plt.subplots(figsize=(7,4))
            ax.semilogx(diams, passing, marker='o')
            ax.set_xlabel("Particle size (mm)")
            ax.set_ylabel("% Passing")
            ax.invert_xaxis()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_title("Grain Size Distribution")
            st.pyplot(fig)
            # save into site
            site["GSD"] = {"diameters":diams, "passing":passing, **metrics}
            st.success(f"Saved GSD for site. D10={metrics['D10']:.4g} mm, D30={metrics['D30']:.4g} mm, D60={metrics['D60']:.4g} mm")
        except Exception as e:
            st.error(f"GSD error: {e}")

# OCR Page
def ocr_page():
    st.header("📷 OCR — extract values from an image")
    site = st.session_state["sites"][st.session_state["active_site"]]
    if not OCR_AVAILABLE:
        st.warning("OCR dependencies not available (pytesseract/PIL). Add pytesseract and pillow to requirements to enable OCR.")
    uploaded = st.file_uploader("Upload an image (photo of textbook question or sieve data)", type=["png","jpg","jpeg"])
    if uploaded:
        if OCR_AVAILABLE:
            try:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded", use_column_width=True)
                text = pytesseract.image_to_string(img)
                st.text_area("Extracted text", value=text, height=180)
                # Basic parsing: try to find LL, PL, D10 etc via regex
                import re
                found = {}
                for key in ["LL","PL","D10","D30","D60","P2","P4","CBR"]:
                    pattern = re.compile(rf"{key}[:=]?\s*([0-9]+\.?[0-9]*)", re.I)
                    m = pattern.search(text)
                    if m:
                        found[key] = float(m.group(1))
                        site.setdefault("classifier_inputs",{})[key] = float(m.group(1))
                if found:
                    st.success(f"Parsed values: {found}")
                    st.write("Values saved into classifier inputs.")
                else:
                    st.info("No clear numeric matches found automatically.")
            except Exception as e:
                st.error(f"OCR failed: {e}")
        else:
            st.warning("OCR not available in this deployment.")
# Locator Page (with Earth Engine auth at top)
# Locator Page (with Earth Engine auth at top)

import os
import json
import streamlit as st
import geemap.foliumap as geemap
import ee
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from streamlit_folium import st_folium

# =====================================================
# EE Init Helper
# =====================================================
def initialize_ee():
    EARTHENGINE_TOKEN = os.getenv("EARTHENGINE_TOKEN")
    SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")

    if "ee_initialized" in st.session_state and st.session_state["ee_initialized"]:
        return True

    if EARTHENGINE_TOKEN and SERVICE_ACCOUNT:
        try:
            creds = ee.ServiceAccountCredentials(email=SERVICE_ACCOUNT, key_data=EARTHENGINE_TOKEN)
            ee.Initialize(creds)
            st.session_state["ee_initialized"] = True
            return True
        except Exception as e:
            st.warning(f"Service account init failed: {e}, falling back...")

    try:
        ee.Initialize()
        st.session_state["ee_initialized"] = True
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            st.session_state["ee_initialized"] = True
            return True
        except Exception as e:
            st.error(f"Earth Engine auth failed: {e}")
            return False

# =====================================================
# Safe reducers
# =====================================================
def safe_get_reduce(region, image, band, scale=1000, default=None, max_pixels=int(1e7)):
    try:
        rr = image.reduceRegion(ee.Reducer.mean(), region, scale=scale, maxPixels=max_pixels)
        val = rr.get(band)
        return float(val.getInfo()) if val else default
    except Exception:
        return default

def safe_reduce_histogram(region, image, band, scale=1000, max_pixels=int(1e7)):
    try:
        rr = image.reduceRegion(ee.Reducer.frequencyHistogram(), region, scale=scale, maxPixels=max_pixels)
        hist = rr.get(band)
        return hist.getInfo() if hist else {}
    except Exception:
        return {}

def safe_time_series(region, collection, band, start, end,
                     reducer=None, scale=1000, max_pixels=int(1e7)):
    try:
        if reducer is None:
            reducer = ee.Reducer.mean()   # ✅ assign inside function

        def per_image(img):
            date = img.date().format("YYYY-MM-dd")
            val = img.reduceRegion(reducer, region, scale=scale, maxPixels=max_pixels).get(band)
            return ee.Feature(None, {"date": date, "val": val})

        feats = collection.filterDate(start, end).map(per_image).filter(ee.Filter.notNull(["val"])).getInfo()
        pts = []
        for f in feats.get("features", []):
            p = f.get("properties", {})
            if p.get("val") is not None:
                pts.append((p.get("date"), float(p.get("val"))))
        return pts
    except Exception:
        return []

# =====================================================
# Map snapshot (in-memory, no disk bloat)
# =====================================================
def export_map_snapshot(m, width=800, height=600):
    """Return PNG snapshot bytes of geemap Map."""
    try:
        from io import BytesIO
        buf = BytesIO()
        m.screenshot(filename=None, region=None, dimensions=(width, height), out_file=buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.warning(f"Map snapshot failed: {e}")
        return None

# =====================================================
# Locator page
# =====================================================
def locator_page():
    st.title("🌍 GeoMate Interactive Earth Explorer")
    st.markdown(
        "Draw a polygon (or rectangle) on the map using the drawing tool. "
        "Then press **Compute Summaries** to compute soil, elevation, seismic, flood, landcover, NDVI, and atmospheric data."
    )

    # --- Auth
    if not initialize_ee():
        st.stop()

    # --- Map setup
    m = geemap.Map(center=[28.0, 72.0], zoom=5, plugin_Draw=True, draw_export=True, locate_control=True)
    
    # ✅ Add a basemap explicitly
    m.add_basemap("HYBRID")      # Google Satellite Hybrid
    m.add_basemap("ROADMAP")     # Google Roads
    m.add_basemap("Esri.WorldImagery")
    m.add_basemap("OpenStreetMap")

    # Restore ROI (if available)
    if "roi_geojson" in st.session_state:
        import folium
        try:
            saved = st.session_state["roi_geojson"]
            folium.GeoJson(saved, name="Saved ROI",
                           style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0.1}).add_to(m)
        except Exception as e:
            st.warning(f"Could not re-add saved ROI: {e}")

    # --- Datasets
    try:
        dem = ee.Image("NASA/NASADEM_HGT/001"); dem_band_name = "elevation"
    except Exception:
        dem, dem_band_name = None, None

    soil_img, chosen_soil_band = None, None
    try:
        soil_img = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
        bands = soil_img.bandNames().getInfo()
        chosen_soil_band = st.selectbox("Select soil clay band", options=bands, index=bands.index("b200") if "b200" in bands else 0)
    except Exception:
        soil_img, chosen_soil_band = None, None

    try:
        seismic_img = ee.Image("SEDAC/GSHAPSeismicHazard"); seismic_band = "gshap"
    except Exception:
        seismic_img, seismic_band = None, None

    try:
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater"); water_band = "occurrence"
    except Exception:
        water, water_band = None, None

    try:
        landcover = ee.Image("ESA/WorldCover/v200"); lc_band = "Map"
    except Exception:
        landcover, lc_band = None, None

    try:
        ndvi_col = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI")
    except Exception:
        ndvi_col = None

    # Atmospheric datasets
    try:
        precip_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    except Exception:
        precip_col = None

    try:
        temp_col = ee.ImageCollection("MODIS/061/MOD11A2").select("LST_Day_1km")
    except Exception:
        temp_col = None

    try:
        pm25_img = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI").select("absorbing_aerosol_index").mean()
    except Exception:
        pm25_img = None

    # --- Render & capture ROI
    result = st_folium(m, width=800, height=600, returned_objects=["last_active_drawing"])
    roi, flat_coords = None, None

    if result and "last_active_drawing" in result and result["last_active_drawing"]:
        feat = result["last_active_drawing"]
        geom = feat.get("geometry")
        if geom:
            try:
                roi = ee.Geometry(geom)
                coords = geom.get("coordinates", None)
                st.session_state["roi_geojson"] = feat
                if coords:
                    if geom["type"] in ["Polygon", "MultiPolygon"]:
                        flat_coords = [(lat, lon) for ring in coords for lon, lat in ring]
                    elif geom["type"] == "Point":
                        lon, lat = coords; flat_coords = [(lat, lon)]
                    elif geom["type"] == "LineString":
                        flat_coords = [(lat, lon) for lon, lat in coords]
                if flat_coords: st.session_state["roi_coords"] = flat_coords
                st.success("✅ ROI captured!")
            except Exception as e:
                st.error(f"Failed to convert geometry: {e}")

    if roi is None and "roi_geojson" in st.session_state:
        try:
            geom = st.session_state["roi_geojson"].get("geometry")
            if geom:
                roi = ee.Geometry(geom)
                coords = geom.get("coordinates", None)
                if coords:
                    if geom["type"] in ["Polygon", "MultiPolygon"]:
                        flat_coords = [(lat, lon) for ring in coords for lon, lat in ring]
                    elif geom["type"] == "Point":
                        lon, lat = coords; flat_coords = [(lat, lon)]
                    elif geom["type"] == "LineString":
                        flat_coords = [(lat, lon) for lon, lat in coords]
                if flat_coords: st.session_state["roi_coords"] = flat_coords
            st.info("♻️ ROI restored from session")
        except Exception as e:
            st.warning(f"Could not restore ROI: {e}")

    # Show coordinates
    if "roi_coords" in st.session_state:
        st.markdown("### 📍 ROI Coordinates (Lat, Lon)")
        st.write(st.session_state["roi_coords"])

    # --- Compute summaries
    if st.button("Compute Summaries"):
        if roi is None:
            st.error("⚠️ No ROI found. Please draw first.")
        else:
            st.success("ROI ready — computing...")

            soil_val = safe_get_reduce(roi, soil_img.select(chosen_soil_band), chosen_soil_band, 1000) if soil_img and chosen_soil_band else None
            elev_val = safe_get_reduce(roi, dem, dem_band_name, 1000) if dem else None
            seismic_val = safe_get_reduce(roi, seismic_img, seismic_band, 5000) if seismic_img else None
            flood_val = safe_get_reduce(roi, water.select(water_band), water_band, 30) if water else None
            lc_stats = safe_reduce_histogram(roi, landcover, lc_band, 30) if landcover else {}
            ndvi_ts = []
            if ndvi_col:
                end = datetime.utcnow().strftime("%Y-%m-%d")
                start = (datetime.utcnow() - timedelta(days=365*2)).strftime("%Y-%m-%d")
                ndvi_ts = safe_time_series(roi, ndvi_col, "NDVI", start, end)

            precip_ts, temp_ts, pm25_val = [], [], None
            if precip_col:
                end = datetime.utcnow().strftime("%Y-%m-%d")
                start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
                precip_ts = safe_time_series(roi, precip_col, "precipitation", start, end, scale=5000)
            if temp_col:
                end = datetime.utcnow().strftime("%Y-%m-%d")
                start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
                temp_ts = safe_time_series(roi, temp_col, "LST_Day_1km", start, end, scale=1000)
            if pm25_img:
                pm25_val = safe_get_reduce(roi, pm25_img, "absorbing_aerosol_index", 10000)

            # Save to site
            active = st.session_state.get("active_site", 0)
            if "sites" in st.session_state:
                site = st.session_state["sites"][active]
                try:
                    site["ROI"] = roi.getInfo()
                except Exception:
                    site["ROI"] = "Not available"
                site["Soil Profile"] = f"{soil_val} ({chosen_soil_band})" if soil_val else "N/A"
                site["Topo Data"] = f"{elev_val} m" if elev_val else "N/A"
                site["Seismic Data"] = seismic_val if seismic_val else "N/A"
                site["Flood Data"] = flood_val if flood_val else "N/A"
                site["Environmental Data"] = {"Landcover": lc_stats, "NDVI": ndvi_ts}
                site["Atmospheric Data"] = {"Precipitation": precip_ts, "Temperature": temp_ts, "PM2.5": pm25_val}

            st.session_state["soil_json"] = {
                "Soil": soil_val, "Soil Band": chosen_soil_band,
                "Elevation": elev_val, "Seismic": seismic_val,
                "Flood": flood_val, "Landcover Stats": lc_stats,
                "NDVI TS": ndvi_ts,
                "Precipitation TS": precip_ts,
                "Temperature TS": temp_ts,
                "PM2.5": pm25_val
            }

            # Snapshot
            map_bytes = export_map_snapshot(m)
            if map_bytes:
                st.session_state["last_map_snapshot"] = map_bytes
                if "sites" in st.session_state:
                    st.session_state["sites"][active]["map_snapshot"] = map_bytes
                st.image(map_bytes, caption="Map Snapshot", use_column_width=True)

            import plotly.express as px
            import plotly.graph_objects as go
            
            # -------------------------------
            # 📊 Display Summaries (Locator)
            # -------------------------------
            st.subheader("📊 Summary")
            
            # --- Metric Cards
            c1, c2, c3 = st.columns(3)
            c1.metric("🟤 Soil (%)", f"{soil_val:.2f}" if soil_val else "N/A", help="Soil clay content")
            c2.metric("⛰️ Elevation (m)", f"{elev_val:.1f}" if elev_val else "N/A", help="Mean elevation")
            c3.metric("🌪️ Seismic PGA", f"{seismic_val:.3f}" if seismic_val else "N/A", help="Seismic hazard index")
            
            c4, c5, c6 = st.columns(3)
            c4.metric("🌊 Flood Occurrence", f"{flood_val:.2f}" if flood_val else "N/A")
            c5.metric("💨 PM2.5 Index", f"{pm25_val:.2f}" if pm25_val else "N/A")
            c6.metric("🟢 NDVI Count", f"{len(ndvi_ts)} pts" if ndvi_ts else "0")
            
            # --- Pie Chart for Landcover
            # --- Donut Chart for Landcover
            if lc_stats:
                labels = list(map(str, lc_stats.keys()))
                values = list(lc_stats.values())
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,  # donut
                    textinfo="percent+label",  # show % and class
                    insidetextorientation="radial",
                    marker=dict(colors=px.colors.sequential.Oranges_r)
                )])
                fig.update_layout(
                    title="🌍 Landcover Distribution",
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            
            # --- Time Series with Plotly
            def plot_timeseries(ts, title, ylab, color):
                if ts:
                    dates, vals = zip(*ts)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines+markers", line=dict(color=color)))
                    fig.update_layout(
                        title=title,
                        xaxis_title="Date",
                        yaxis_title=ylab,
                        template="plotly_dark",
                        hovermode="x unified",
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            plot_timeseries(ndvi_ts, "🌿 NDVI Trend (2 years)", "NDVI", "#FF7A00")
            plot_timeseries(precip_ts, "🌧️ Precipitation Trend (1 year)", "mm/day", "#00BFFF")
            plot_timeseries(temp_ts, "🌡️ Land Surface Temp (1 year)", "K", "#FF3333")


# GeoMate Ask (RAG) — simple chat with memory per site and auto-extract numeric values
import re, json, pickle
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -------------------
# Load FAISS DB once
# -------------------
@st.cache_resource
def load_faiss():
    # Adjust path to where you unzip faiss_books_db.zip
    faiss_dir = "faiss_books_db"
    # embeddings must match the one you used when creating index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with open(f"{faiss_dir}/index.pkl", "rb") as f:
        data = pickle.load(f)
    vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_faiss()

# -------------------
# RAG Chat Page
# -------------------
def rag_page():
    st.header("🤖 GeoMate Ask (RAG + LLM)")
    site = st.session_state["sites"][st.session_state["active_site"]]

    # --- Ensure Site ID exists ---
    if site.get("Site ID") is None:
        site_id = st.session_state["sites"].index(site)
        site["Site ID"] = site_id
    else:
        site_id = site["Site ID"]

    # --- Initialize rag_history properly ---
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = {}
    if site_id not in st.session_state["rag_history"]:
        st.session_state["rag_history"][site_id] = []

    # --- Display chat history ---
    hist = st.session_state["rag_history"][site_id]
    for entry in hist:
        who, text = entry.get("who"), entry.get("text")
        if who == "bot":
            st.markdown(f"<div class='chat-bot'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-user'>{text}</div>", unsafe_allow_html=True)

    # --- User input ---
    user_msg = st.text_input("You:", key=f"rag_input_{site_id}")
    if st.button("Send", key=f"rag_send_{site_id}"):
        if not user_msg.strip():
            st.warning("Enter a message.")
        else:
            # Save user msg
            st.session_state["rag_history"][site_id].append(
                {"who": "user", "text": user_msg}
            )

            # --- Retrieve from FAISS ---
            docs = vectorstore.similarity_search(user_msg, k=3)
            context_text = "\n".join([d.page_content for d in docs])

            # --- Build context for LLM ---
            context = {
                "site": {
                    k: v
                    for k, v in site.items()
                    if k in [
                        "Site Name",
                        "lat",
                        "lon",
                        "USCS",
                        "AASHTO",
                        "GI",
                        "Load Bearing Capacity",
                        "Soil Profile",
                        "Flood Data",
                        "Seismic Data",
                    ]
                },
                "chat_history": st.session_state["rag_history"][site_id],
            }

            prompt = (
                f"You are GeoMate AI, an expert geotechnical assistant.\n\n"
                f"Relevant references:\n{context_text}\n\n"
                f"Site context: {json.dumps(context)}\n\n"
                f"User: {user_msg}\n\n"
                f"Answer concisely, include citations [ref:source]. "
                f"If user provides numeric engineering values, return them in the format: [[FIELD: value unit]]."
            )

            # Call the unified LLM function
            resp = llm_generate(prompt, model=st.session_state["llm_model"], max_tokens=500)
            
            # Save bot reply
            st.session_state["rag_history"][site_id].append({"who": "bot", "text": resp})

            # Display reply
            st.markdown(f"<div class='chat-bot'>{resp}</div>", unsafe_allow_html=True)

            # Extract bracketed numeric values
            matches = re.findall(
                r"\[\[([A-Za-z0-9 _/-]+):\s*([0-9.+-eE]+)\s*([A-Za-z%\/]*)\]\]", resp
            )
            for m in matches:
                field, val, unit = m[0].strip(), m[1].strip(), m[2].strip()
                if "bearing" in field.lower():
                    site["Load Bearing Capacity"] = f"{val} {unit}"
                elif "skin" in field.lower():
                    site["Skin Shear Strength"] = f"{val} {unit}"
                elif "compaction" in field.lower():
                    site["Relative Compaction"] = f"{val} {unit}"

            st.success(
                "Response saved ✅ with citations and recognized numeric fields auto-stored in site data."
            )


# -------------------
# Report fields (still needed in reports_page)
# -------------------

REPORT_FIELDS = [
    ("Load Bearing Capacity", "kPa or psf"),
    ("Skin Shear Strength", "kPa"),
    ("Relative Compaction", "%"),
    ("Rate of Consolidation", "mm/yr or days"),
    ("Nature of Construction", "text"),
    ("Borehole Count", "number"),
    ("Max Depth (m)", "m"),
    ("SPT N (avg)", "blows/ft"),
    ("CBR (%)", "%"),
    ("Allowable Bearing (kPa)", "kPa"),
]

# -------------------------------
# Imports
# -------------------------------
import io, re, json, tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List

import streamlit as st
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# =============================
# LLM Helper (Groq API)
# =============================
import requests, json, os
import streamlit as st
from datetime import datetime
import tempfile
from typing import Dict, Any, Optional, List
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


# ----------------------------
# LLM Helper (Reports Analysis with Orchestration)
# ----------------------------
def groq_llm_analyze(prompt: str, section_title: str, max_tokens: int = 500) -> str:
    """
    Query the selected model (Groq / Gemini / DeepSeek).
    If token limit or error occurs, automatically switch to backup models
    and continue the analysis seamlessly.
    """

    # Primary model (user choice from sidebar)
    model_chain = [st.session_state.get("llm_model", "groq/compound")]

    # Add fallback chain (priority order: DeepSeek → Gemini → Groq)
    if "deepseek" not in model_chain:
        model_chain.append("deepseek-r1-distill-llama-70b")
    if "gemini" not in model_chain:
        model_chain.append("gemini-1.5-pro")
    if "groq/compound" not in model_chain:
        model_chain.append("groq/compound")

    system_message = (
        "You are GeoMate, a geotechnical engineering assistant. "
        "Respond professionally with concise analysis and insights."
    )

    full_prompt = (
        f"{system_message}\n\n"
        f"Section: {section_title}\n\n"
        f"Input: {prompt}\n\n"
        f"Write a professional engineering analysis for this section."
    )

    final_response = ""
    remaining_prompt = full_prompt

    # Try each model in the chain until completion
    for model_name in model_chain:
        try:
            response = llm_generate(remaining_prompt, model=model_name, max_tokens=max_tokens)

            if not response or "[LLM error" in response:
                # If failed, continue to next model
                continue

            final_response += response.strip()

            # If response length is close to max_tokens, assume it cut off → continue with next model
            if len(response.split()) >= (max_tokens - 20):
                # Add continuation instruction for the next model
                remaining_prompt = (
                    f"Continue the analysis from where the last model stopped. "
                    f"So far the draft is:\n\n{final_response}\n\n"
                    f"Continue writing professionally without repeating."
                )
                continue
            else:
                break  # Finished properly, exit loop

        except Exception as e:
            final_response += f"\n[LLM orchestration error @ {model_name}: {e}]\n"
            continue

    return final_response if final_response else "[LLM error: All models failed]"


# =============================
# Build Full Geotechnical Report
# =============================
def build_full_geotech_pdf(
    site: Dict[str, Any],
    filename: str,
    include_map_image: Optional[bytes] = None,
    ext_refs: Optional[List[str]] = None
):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=22,
                                 alignment=1, textColor=colors.HexColor("#FF6600"), spaceAfter=12)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                        textColor=colors.HexColor("#1F4E79"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=13)

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    elems = []

    # Title Page
    elems.append(Paragraph("GEOTECHNICAL INVESTIGATION REPORT", title_style))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"<b>Client:</b> {site.get('Company Name','-')}", body))
    elems.append(Paragraph(f"<b>Contact:</b> {site.get('Company Contact','-')}", body))
    elems.append(Paragraph(f"<b>Project:</b> {site.get('Project Name','-')}", body))
    elems.append(Paragraph(f"<b>Site:</b> {site.get('Site Name','-')}", body))
    elems.append(Paragraph(f"<b>Date:</b> {datetime.today().strftime('%Y-%m-%d')}", body))
    elems.append(PageBreak())

    # TOC
    elems.append(Paragraph("TABLE OF CONTENTS", h1))
    toc_items = [
        "1.0 Summary", "2.0 Introduction", "3.0 Site Description & Geology",
        "4.0 Field & Laboratory Testing", "5.0 Evaluation of Geotechnical Properties",
        "6.0 Provisional Classification", "7.0 Recommendations",
        "8.0 LLM Analysis", "9.0 Figures & Tables", "10.0 Appendices & References"
    ]
    for i, t in enumerate(toc_items, 1):
        elems.append(Paragraph(f"{i}. {t}", body))
    elems.append(PageBreak())

    # Sections with LLM calls
    elems.append(Paragraph("1.0 SUMMARY", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(site, indent=2), "Summary"), body))
    elems.append(PageBreak())

    elems.append(Paragraph("2.0 INTRODUCTION", h1))
    elems.append(Paragraph(groq_llm_analyze(site.get("Project Description",""), "Introduction"), body))

    elems.append(Paragraph("3.0 SITE DESCRIPTION & GEOLOGY", h1))
    geology_text = f"Topo: {site.get('Topography')}, Drainage: {site.get('Drainage')}, Land Use: {site.get('Current Land Use')}, Geology: {site.get('Regional Geology')}"
    elems.append(Paragraph(groq_llm_analyze(geology_text, "Geology & Site Description"), body))
    elems.append(PageBreak())

    elems.append(Paragraph("4.0 FIELD & LABORATORY TESTING", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(site.get('Laboratory Results',[]), indent=2), "Field & Lab Testing"), body))
    elems.append(PageBreak())

    elems.append(Paragraph("5.0 EVALUATION OF GEOTECHNICAL PROPERTIES", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(site, indent=2), "Evaluation of Properties"), body))

    elems.append(Paragraph("6.0 PROVISIONAL CLASSIFICATION", h1))
    class_text = f"USCS={site.get('USCS')}, AASHTO={site.get('AASHTO')}"
    elems.append(Paragraph(groq_llm_analyze(class_text, "Soil Classification"), body))

    elems.append(Paragraph("7.0 RECOMMENDATIONS", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(site, indent=2), "Recommendations"), body))

    elems.append(Paragraph("8.0 LLM ANALYSIS (GeoMate)", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(site, indent=2), "LLM Insights"), body))

    # Map snapshot
    if include_map_image:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(include_map_image)
        tmp.flush()
        elems.append(PageBreak())
        elems.append(Paragraph("9.0 MAP SNAPSHOT", h1))
        elems.append(RLImage(tmp.name, width=160*mm, height=90*mm))

    # References
    elems.append(PageBreak())
    elems.append(Paragraph("10.0 REFERENCES", h1))
    if ext_refs:
        for r in ext_refs:
            elems.append(Paragraph(f"- {r}", body))
    else:
        elems.append(Paragraph("No external references provided.", body))

    doc.build(elems)
    return filename


# =============================
# Build Classification Report
# =============================
def build_classification_pdf(
    site: Dict[str, Any],
    classification: Dict[str, Any],
    filename: str
):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18,
                                 textColor=colors.HexColor("#FF6600"), alignment=1)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=12, textColor=colors.HexColor("#1F4E79"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10)

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    elems = []

    # Title Page
    elems.append(Paragraph("SOIL CLASSIFICATION REPORT", title_style))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"Site: {site.get('Site Name','Unnamed')}", body))
    elems.append(Paragraph(f"Date: {datetime.today().strftime('%Y-%m-%d')}", body))
    elems.append(PageBreak())

    # Sections
    elems.append(Paragraph("1.0 DETERMINISTIC RESULTS", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(classification, indent=2), "Deterministic Results"), body))
    elems.append(PageBreak())

    elems.append(Paragraph("2.0 ENGINEERING CHARACTERISTICS", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(classification.get('engineering_characteristics',{}), indent=2), "Engineering Characteristics"), body))
    elems.append(PageBreak())

    elems.append(Paragraph("3.0 DECISION PATHS", h1))
    dp_text = f"USCS path: {classification.get('USCS_decision_path')}, AASHTO path: {classification.get('AASHTO_decision_path')}"
    elems.append(Paragraph(groq_llm_analyze(dp_text, "Decision Paths"), body))
    elems.append(PageBreak())

    elems.append(Paragraph("4.0 LLM ANALYSIS", h1))
    elems.append(Paragraph(groq_llm_analyze(json.dumps(classification, indent=2), "LLM Analysis"), body))

    doc.build(elems)
    return filename

# -------------------------------
# Reports Page
# -------------------------------
def reports_page():
    st.header("📑 Reports — Classification & Full Geotechnical")
    site = st.session_state["sites"][st.session_state["active_site"]]

    # =====================================================
    # Classification Report
    # =====================================================
    st.subheader("📘 Classification-only Report")
    if site.get("classifier_decision"):
        st.markdown("You have a saved classification for this site.")
        if st.button("Generate Classification PDF"):
            fname = f"classification_{site['Site Name'].replace(' ','_')}.pdf"

            # Collect references from rag_history
            refs = []
            if "rag_history" in st.session_state and site.get("Site ID") in st.session_state["rag_history"]:
                for h in st.session_state["rag_history"][site["Site ID"]]:
                    if h["who"] == "bot" and "[ref:" in h["text"]:
                        for m in re.findall(r"\[ref:([^\]]+)\]", h["text"]):
                            refs.append(m)

            # Build classification PDF
            buffer = io.BytesIO()
            build_classification_pdf(site, site.get("classifier_decision"), buffer)
            buffer.seek(0)

            st.download_button("⬇️ Download Classification PDF", buffer, file_name=fname, mime="application/pdf")
    else:
        st.info("No classification saved for this site yet. Use the Classifier page.")

    # =====================================================
    # Quick Report Form
    # =====================================================
    st.markdown("### ✍️ Quick report form (edit values and request LLM analysis)")
    with st.form(key="report_quick_form"):
        cols = st.columns([2, 1, 1])
        cols[0].markdown("**Parameter**")
        cols[1].markdown("**Value**")
        cols[2].markdown("**Unit / Notes**")

        inputs = {}
        for (fld, unit) in REPORT_FIELDS:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"**{fld}**")
            default_val = site.get(fld, "")
            inputs[fld] = c2.text_input(fld, value=str(default_val), label_visibility="collapsed", key=f"quick_{fld}")
            c3.markdown(unit)

        submitted = st.form_submit_button("Save values to site")
        if submitted:
            for fld, _ in REPORT_FIELDS:
                val = inputs.get(fld, "").strip()
                site[fld] = val if val != "" else "Not provided"
            st.success("✅ Saved quick report values to active site.")

    # =====================================================
    # LLM Analysis (Humanized Report Text)
    # =====================================================
    st.markdown("#### 🤖 LLM-powered analysis")
    if st.button("Ask GeoMate (generate analysis & recommendations)"):
        context = {
            "site_name": site.get("Site Name"),
            "project": site.get("Project Name"),
            "site_summary": {
                "USCS": site.get("USCS"), "AASHTO": site.get("AASHTO"), "GI": site.get("GI"),
                "Soil Profile": site.get("Soil Profile"),
                "Key lab results": [r.get("sampleId") for r in site.get("Laboratory Results", [])]
            },
            "inputs": {fld: site.get(fld, "Not provided") for fld, _ in REPORT_FIELDS}
        }
        prompt = (
            "You are GeoMate AI, an engineering assistant. Given the following site context and "
            "engineering parameters (some may be 'Not provided'), produce:\n1) short executive summary, "
            "2) geotechnical interpretation (classification, key risks), 3) recommended remedial/improvement "
            "options and 4) short design notes. Provide any numeric outputs in the format [[FIELD: value unit]].\n\n"
            f"Context: {json.dumps(context)}"
        )
        resp = groq_llm_analyze(prompt, section_title="GeoMate Analysis")

        st.markdown("**GeoMate analysis**")
        st.markdown(resp)

        # Extract structured values from [[FIELD: value unit]]
        matches = re.findall(r"\[\[([A-Za-z0-9 _/-]+):\s*([0-9.+-eE]+)\s*([A-Za-z%\/]*)\]\]", resp)
        for m in matches:
            field, val, unit = m[0].strip(), m[1].strip(), m[2].strip()
            if "bearing" in field.lower():
                site["Load Bearing Capacity"] = f"{val} {unit}"
            elif "skin" in field.lower():
                site["Skin Shear Strength"] = f"{val} {unit}"
            elif "compaction" in field.lower():
                site["Relative Compaction"] = f"{val} {unit}"

        site["LLM_Report_Text"] = resp
        st.success("✅ LLM analysis saved to site under 'LLM_Report_Text'.")

    # =====================================================
    # Full Geotechnical Report
    # =====================================================
    st.markdown("---")
    st.subheader("📕 Full Geotechnical Report")

    ext_ref_text = st.text_area("Optional: External references (one per line)", value="")
    ext_refs = [r.strip() for r in ext_ref_text.splitlines() if r.strip()]

    # Add FAISS / rag references
    faiss_refs = []
    if "rag_history" in st.session_state and site.get("Site ID") in st.session_state["rag_history"]:
        for h in st.session_state["rag_history"][site["Site ID"]]:
            if h["who"] == "bot" and "[ref:" in h["text"]:
                for m in re.findall(r"\[ref:([^\]]+)\]", h["text"]):
                    faiss_refs.append(m)
    all_refs = list(set(ext_refs + faiss_refs))

    if st.button("Generate Full Geotechnical Report PDF"):
        outname = f"Full_Geotech_Report_{site.get('Site Name','site')}.pdf"
        mapimg = site.get("map_snapshot")

        # ✅ Classification results also included inside full report
        build_full_geotech_pdf(site, outname, include_map_image=mapimg, ext_refs=all_refs)

        with open(outname, "rb") as f:
            st.download_button("⬇️ Download Full Geotechnical Report", f, file_name=outname, mime="application/pdf")

# 8) Page router
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

page = st.session_state["page"]

# Option menu top (main nav)
# ===============================
# Navigation (Option Menu)
# ===============================
from streamlit_option_menu import option_menu

# Define all pages
PAGES = ["Home", "Soil recognizer", "Classifier", "GSD", "OCR", "Locator", "RAG", "Reports"]

# Set default page if not defined yet
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Build horizontal option menu
# ===============================
# Sidebar or Top-Bar Model Selector
# ===============================
    

# ===============================
# Page Menu (Horizontal)
# ===============================
selected = option_menu(
    None,
    PAGES,
    icons=[
        "house", "chart", "journal-code", "bar-chart", "camera",
        "geo-alt", "robot", "file-earmark-text"
    ],
    menu_icon="cast",
    default_index=PAGES.index(st.session_state["page"]) if st.session_state["page"] in PAGES else 0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0px", "background-color": "#0b0b0b"},
        "nav-link": {"font-size": "14px", "color": "#cfcfcf"},
        "nav-link-selected": {"background-color": "#FF7A00", "color": "white"},
    }
)

# Save selection into session_state
st.session_state["page"] = selected

# ===============================
# Page Routing
# ===============================
if selected == "Home":
    st.title("🏠 Welcome to GeoMate")
    st.write("Your geotechnical AI copilot.")
    st.info(f"Currently using **{st.session_state['llm_model']}** for analysis.")

elif selected == "Soil recognizer":
    st.title("🔎 Soil Recognizer")
    st.write("Upload soil images for classification.")

elif selected == "Classifier":
    st.title("📊 Soil Classifier")
    st.write("Enter lab/field parameters for classification.")

elif selected == "GSD":
    st.title("📉 Grain Size Distribution")
    st.write("Analyze particle size distribution.")

elif selected == "OCR":
    st.title("📷 OCR Extractor")
    st.write("Upload lab sheets for automatic text extraction.")

elif selected == "Locator":
    st.title("🌍 Locator Tool")
    st.write("Draw ROI on map and compute Earth Engine summaries.")

elif selected == "RAG":
    st.title("🤖 Knowledge Assistant")
    st.write("Query soil and geotechnical references with AI.")
    st.caption(f"Model in use: {st.session_state['llm_model']}")

elif selected == "Reports":
    st.title("📑 Reports")
    st.write("Generate classification and full reports.")
    st.caption(f"Analysis will run with: {st.session_state['llm_model']}")


# Display page content
if page == "Home":
    landing_page()
elif page == "Classifier":
    soil_classifier_page()
elif page == "GSD":
    gsd_page()
elif page == "OCR":
    ocr_page()
elif page == "Locator":
    locator_page()
elif page == "RAG":
    rag_page()
elif page == "Reports":
    reports_page()
elif page == "Soil recognizer":
    soil_recognizer_page()
else:
    landing_page()

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9aa7bf'>GeoMate V2 • AI geotechnical copilot • Built for HF Spaces</div>", unsafe_allow_html=True)
