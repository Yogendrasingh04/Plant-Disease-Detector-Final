import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import pandas as pd
import requests # NEW: File download karne ke liye
import time # NEW: Download speed check ke liye

# --- Configuration & Device Setup ---
# Streamlit deployment ke liye device ko CPU par set karein
device = torch.device('cpu')

# --- NEW: GOOGLE DRIVE CONFIGURATION ---
# IMPORTANT: Apne Google Drive sharing link ki ID yahan daalein.
GOOGLE_DRIVE_FILE_ID = '17OWRF9r-9AKvakdimu4nG8tr6Cu17E2p' # <-- Yahan ID badle!
MODEL_PATH = 'plant_disease_model.pth' 
# ----------------------------------------

# ImageNet means and standard deviations for normalization (Same as Colab)
# --- FIX FOR NameError: 'IMAGENET_MEAN' is not defined ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ----------------------------------------------------------

# --- NEW FUNCTION: Download from Google Drive ---
@st.cache_resource(show_spinner="⏳ Downloading Model from Google Drive...")
def download_model_from_gdrive(file_id, output_path):
    # Agar model pehle se downloaded hai to skip karein
    if os.path.exists(output_path):
        return

    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Download request
    response = requests.get(URL, stream=True)
    
    if response.status_code != 200:
        st.error(f"❌ Error downloading model (Status Code: {response.status_code}). Check your File ID and sharing permissions.")
        return

    # Write content to the file path
    total_size = int(response.headers.get('content-length', 0))
    start_time = time.time()
    
    with open(output_path, "wb") as f:
        # File ko chunks mein likhein (large files ke liye zaroori)
        for chunk in response.iter_content(chunk_size=1024 * 1024): 
            if chunk:
                f.write(chunk)

    end_time = time.time()
    st.success(f"✅ Model Downloaded ({total_size // (1024*1024)} MB) in {end_time - start_time:.2f} seconds.")

# --- Model Definition Function ---
def load_model(num_classes, model_path, device):
    # Pehle model download karein (agar nahi hua hai to)
    download_model_from_gdrive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH) 
    
    # Model Structure... 
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    try:
        # 'weights_only=False'
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}. Check the model file integrity.")
        st.stop()
        
        
# --- Data Transformations for Prediction ---
# FIX IS APPLIED HERE: IMAGENET_MEAN and IMAGENET_STD are now defined above
prediction_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# --- Disease Information (Zaroori! Please customize this fully!) ---
# [CONTENT REMOVED FOR BREVITY] (Ensure your full disease_info dictionary is here)
disease_info = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "symptoms": "Patto, phal aur tehniyon par olive-green se brown spots.",
        "treatment": "Infected patto ko hatayein, Captan ya Myclobutanil jaisi fungicid istemal karein, resistant kism chunein."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "symptoms": "Patte hare bhare, majboot tana, phal ka accha vikas.",
        "treatment": "Samay par pani dein, santulit khaad aur kitnashak niyantran rakhein."
    },
    'Apple___Black_rot': {
        "plant": "Apple",
        "disease": "Black Rot",
        "symptoms": "Patto par gol brown daag, phal par bade aur sade hue kaale spots.",
        "treatment": "Bimar tehniyon ko kaat dein, sade hue phal hata dein, fungicid lagayein."
    },
    'Apple___Cedar_apple_rust': {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "symptoms": "Apple ke patto par chamakdaar narangi spots. Phal ko bhi affect kar sakta hai.",
        "treatment": "Cedar ke pedon ke paas lagane se bachein, rust-resistant kism chunein."
    },
    'Tomato___Early_blight': {
        "plant": "Tomato",
        "disease": "Early Blight",
        "symptoms": "Purane patto par target-jaisi concentric rings wale gehre brown spots.",
        "treatment": "Faslon ka chakkar (crop rotation) karein, fungicid lagayein, bimar patton ko hata dein."
    },
    'Tomato___Late_blight': {
        "plant": "Tomato",
        "disease": "Late Blight",
        "symptoms": "Irregular, pani se bhege hue lesions jo jaldi failte hain, niche ki taraf safed fungal growth.",
        "treatment": "Resistant kism lagayein, fungicid use karein, acchi hawa ka sanchar banayein."
    },
    'Potato___healthy': {
        "plant": "Potato",
        "disease": "Healthy",
        "symptoms": "Pattiyan tez hari aur majboot vikas.",
        "treatment": "Uchit unchai tak mitti chadayein, santulit poshak tatva aur paryapt pani dein."
    },
    'Tomato___Tomato_mosaic_virus': {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "symptoms": "Mottling, curling, aur patton ka vikriti (distortion), ruka hua vikas.",
        "treatment": "Koi ilaaj nahi; sankramit paudhon ko hata dein, upkaranon ko sanitize karein."
    },
    # Please ensure all 38 classes are present here in your final file
}
disease_df = pd.DataFrame.from_dict(disease_info, orient='index')

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

NUM_CLASSES = len(class_names)


# --- Load Model (Caching for efficiency) ---
@st.cache_resource
def get_model():
    return load_model(NUM_CLASSES, MODEL_PATH, device)

model = get_model()

# --- Prediction Function ---
def predict_image_class(image, model, transforms, class_names):
    image = transforms(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_percent = confidence.cpu().item() * 100
    
    return predicted_class, confidence_percent


# --- Streamlit App UI (Stylish & Attractive - DARK MODE) ---

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for DARK MODE Styling
st.markdown("""
<style>
/* Main Background and Layout */
.stApp {
    background-color: #1E1E1E; /* Dark Grey / Black */
    color: #F0F0F0; /* Off-White Text */
    font-family: 'Arial', sans-serif;
}
/* Header Styling */
h1 {
    color: #4CAF50; /* Bright Green */
    text-align: center;
    font-size: 2.5em;
    padding-bottom: 0.5em;
    border-bottom: 2px solid #38761D; /* Darker Green underline */
}
/* Sub-header (h2, h3, h4) styling */
h3, h4 {
    color: #90EE90; /* Light Green */
}
/* Custom Button Styling */
.stButton>button {
    background-color: #4CAF50; /* Green Button */
    color: #1E1E1E; /* Dark Text */
    border-radius: 12px;
    padding: 10px 24px;
    font-size: 16px;
    transition-duration: 0.4s;
    border: 2px solid #4CAF50;
    margin-top: 15px;
    width: 100%; /* Full width button */
}
.stButton>button:hover {
    background-color: #38761D; /* Darker Green hover */
    color: #F0F0F0;
    border: 2px solid #4CAF50;
}

/* Info, Success, Warning boxes style */
div[data-testid="stAlert"] {
    border-radius: 10px;
}
div.stSuccess {
    background-color: #004d00; /* Darker green success box */
    color: #90EE90;
}
div.stWarning {
    background-color: #333300; /* Dark yellow warning box */
    color: #FFFF99;
}
div.stInfo {
    background-color: #003366; /* Dark blue info box */
    color: #ADD8E6;
}
/* Expander style */
.streamlit-expanderHeader {
    background-color: #333333; /* Darker Expander background */
    color: #4CAF50;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# --- UI Content ---

st.markdown("<h1 style='color: #4CAF50; text-align: center;'>🌿 Smart Plant Disease Diagnoser 🌿</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #CCCCCC;'>Upload a leaf image and get instant diagnosis and tailored treatment plans.</p>", unsafe_allow_html=True)

st.markdown("---")

# File Uploader Section
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    uploaded_file = st.file_uploader(
        "**Upload Your Plant Leaf Image Here:**",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Make sure the leaf is clearly visible and centered for best results."
    )

if uploaded_file is not None:
    # --- Display and Prediction Section ---
    image = Image.open(uploaded_file).convert('RGB')

    # Use columns to display image and results side-by-side
    st.markdown("## 🔎 Analysis in Progress")
    img_col, result_col = st.columns([1, 1])

    with img_col:
        st.subheader("Uploaded Leaf Image")
        st.image(image, use_column_width=True, caption=uploaded_file.name)

    with result_col:
        st.subheader("Diagnosis Status")
        # Check if the model has finished downloading (first run only)
        if not os.path.exists(MODEL_PATH):
             st.warning("Model file downloading in progress... (This may take a few minutes on the first run).")

        with st.spinner('🔬 Running AI Model Analysis... Please wait.'):
            predicted_class, confidence = predict_image_class(image, model, prediction_transforms, class_names)

        st.success("✅ Prediction Complete! See results below.")
        st.markdown("---")
        
        # --- Result Display ---
        st.markdown("#### **Prediction Result:**")

        if predicted_class in disease_df.index:
            info = disease_df.loc[predicted_class]

            # Displaying Plant and Disease Name (Using specific colors for contrast)
            st.markdown(f"**🌱 Plant Name:** <span style='color: #4CAF50;'>{info['plant']}</span>", unsafe_allow_html=True)
            st.markdown(f"**🚨 Disease Name:** <span style='color: #FF6347; font-weight: bold;'>{info['disease']}</span>", unsafe_allow_html=True)
            st.markdown(f"**📈 Confidence:** <span style='color: #ADD8E6;'>{confidence:.2f}%</span>", unsafe_allow_html=True)

            # --- Symptoms and Treatment in Expander ---
            st.markdown("---")
            st.markdown("**🔍 Key Symptoms:**")
            st.warning(info['symptoms'])

            with st.expander("🩺 **CLICK HERE FOR COMPLETE TREATMENT PLAN**", expanded=True):
                st.markdown(f"**Recommendation:** {info['treatment']}")
        else:
            st.error(f"Prediction found, but detailed information for: {predicted_class} is missing.")
            st.write(f"**Confidence:** {confidence:.2f}%")


st.markdown("---")


