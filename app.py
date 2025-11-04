import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models 
from PIL import Image
import requests # Hugging Face URL se load karne ke liye zaroori
import torch.nn.functional as F 
import pandas as pd
import io # Agar hum requests.get se load karein (yahan zaroorat nahi, par achhi practice hai)

# --- Configuration & Device Setup ---
# Streamlit deployment ke liye device ko CPU par set karein
device = torch.device('cpu')

# --- HUGGING FACE CONFIGURATION ---
# 1. ZAROORI: Yahan apna Hugging Face se copy kiya hua Direct Download Link daalein.
# Yeh link sidha model (.pth) file par hona chahiye, koi HTML page nahi.
# Maine aapka pichla link wapas daal diya hai.
HUGGING_FACE_MODEL_URL = "https://huggingface.co/Yogendra12/plant-disease-classifier/resolve/main/plant_disease_model.pth" 
# ----------------------------------------

# ImageNet means and standard deviations for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --- Model Class Names (38 Classes - Aapke code se copy kiye gaye) ---
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)_Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__healthy'
]
NUM_CLASSES = len(class_names) # 38 classes


# --- Disease Information (Aapke code se copy kiya gaya) ---
disease_info = {
    "Apple__Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "symptoms": "Patto, phal aur tehniyon par olive-green se brown spots.",
        "treatment": "Infected patto ko hatayein, Captan ya Myclobutanil jaisi fungicid istemal karein, resistant kism chunein."
    },
    "Tomato__healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "symptoms": "Patte hare bhare, majboot tana, phal ka accha vikas.",
        "treatment": "Samay par pani dein, santulit khaad aur kitnashak niyantran rakhein."
    },
    'Apple_Black_rot': {
        "plant": "Apple",
        "disease": "Black Rot",
        "symptoms": "Patto par gol brown daag, phal par bade aur sade hue kaale spots.",
        "treatment": "Bimar tehniyon ko kaat dein, sade hue phal hata dein, fungicid lagayein."
    },
    'Apple_Cedar_apple_rust': {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "symptoms": "Apple ke patto par chamakdaar narangi spots. Phal ko bhi affect kar sakta hai.",
        "treatment": "Cedar ke pedon ke paas lagane se bachein, rust-resistant kism chunein."
    },
    'Tomato_Early_blight': {
        "plant": "Tomato",
        "disease": "Early Blight",
        "symptoms": "Purane patto par target-jaisi concentric rings wale gehre brown spots.",
        "treatment": "Faslon ka chakkar (crop rotation) karein, fungicid lagayein, bimar patton ko hata dein."
    },
    'Tomato_Late_blight': {
        "plant": "Tomato",
        "disease": "Late Blight",
        "symptoms": "Irregular, pani se bhege hue lesions jo jaldi failte hain, niche ki taraf safed fungal growth.",
        "treatment": "Resistant kism lagayein, fungicid use karein, acchi hawa ka sanchar banayein."
    },
    'Potato_healthy': {
        "plant": "Potato",
        "disease": "Healthy",
        "symptoms": "Pattiyan tez hari aur majboot vikas.",
        "treatment": "Uchit unchai tak mitti chadayein, santulit poshak tatva aur paryapt pani dein."
    },
    'Tomato__Tomato_mosaic_virus': {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "symptoms": "Mottling, curling, aur patton ka vikriti (distortion), ruka hua vikas.",
        "treatment": "Koi ilaaj nahi; sankramit paudhon ko hata dein, upkaranon ko sanitize karein."
    },
    # Baki classes ki info yahan aaegi
}
disease_df = pd.DataFrame.from_dict(disease_info, orient='index')


# --- Model Definition and Loading (Hugging Face) ---
@st.cache_resource(show_spinner="⏳ Model weights Hugging Face se download/load kiye ja rahe hain...")
def load_model():
    """Hugging Face URL se ResNet-50 model weights load karta hai."""
    
    try:
        # Weights ko URL se download karke load karna
        weights = torch.hub.load_state_dict_from_url(
            HUGGING_FACE_MODEL_URL, 
            map_location=device, # device='cpu'
            progress=True
        )
        
        # 🚨 ResNet-50 Architecture Definition (Aapke size mismatch error ke anusaar)
        model = models.resnet50(weights=None) 

        # Final Fully Connected layer ko 38 classes ke anusaar badalna
        num_ftrs = model.fc.in_features # ResNet-50 ke liye yeh 2048 hoga
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
        # Weights ko model mein load karna
        # weights_only=False zaroori nahi hai kyunki hum seedha state_dict load kar rahe hain.
        model.load_state_dict(weights) 
        model.to(device)
        model.eval() 

        st.success("✅ ResNet-50 Model safaltapoorvak (successfully) load ho gaya hai!")
        return model
    
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Download Error: HTTP Status {e.response.status_code}. Hugging Face link ya permissions check karein.")
        st.stop()
    except Exception as e:
        # Size Mismatch Error ya koi aur error yahan aayega
        st.error(f"❌ Model Loading Error: {e}. Kripya dekhein ki aapne ResNet-50 aur 38 classes hi use ki thi.")
        st.stop()
        
# --- Load Model (Caching for efficiency) ---
model = load_model()

# --- Data Transformations for Prediction ---
prediction_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# --- Prediction Function (Aapke code se copy kiya gaya) ---
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
        "*Upload Your Plant Leaf Image Here:*",
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
        
        with st.spinner('🔬 Running AI Model Analysis... Please wait.'):
            # Model loading is wrapped in get_model() which handles download/load
            predicted_class, confidence = predict_image_class(image, model, prediction_transforms, class_names)

        st.success("✅ Prediction Complete! See results below.")
        st.markdown("---")
        
        # --- Result Display ---
        st.markdown("#### *Prediction Result:*")

        if predicted_class in disease_df.index:
            info = disease_df.loc[predicted_class]

            # Displaying Plant and Disease Name (Using specific colors for contrast)
            st.markdown(f"🌱 Plant Name:** <span style='color: #4CAF50;'>{info['plant']}</span>", unsafe_allow_html=True)
            st.markdown(f"🚨 Disease Name:** <span style='color: #FF6347; font-weight: bold;'>{info['disease']}</span>", unsafe_allow_html=True)
            st.markdown(f"📈 Confidence:** <span style='color: #ADD8E6;'>{confidence:.2f}%</span>", unsafe_allow_html=True)

            # --- Symptoms and Treatment in Expander ---
            st.markdown("---")
            st.markdown("🔍 Key Symptoms:")
            st.warning(info['symptoms'])

            with st.expander("🩺 *CLICK HERE FOR COMPLETE TREATMENT PLAN*", expanded=True):
                st.markdown(f"*Recommendation:* {info['treatment']}")
        else:
            st.error(f"Prediction found, but detailed information for: {predicted_class} is missing.")
            st.write(f"*Confidence:* {confidence:.2f}%")


st.markdown("---")
