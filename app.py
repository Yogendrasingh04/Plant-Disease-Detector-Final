import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models # ResNet ke liye zaroori
from PIL import Image
import io
import requests
import torch.nn.functional as F # Prediction ke liye zaroori

# ----------------- CONFIGURATION -----------------

# 1. ZAROORI: Yahan apna Hugging Face se copy kiya hua Direct Download Link daalein.
# Yeh link sidha model (.pth) file par hona chahiye, koi HTML page nahi.
HUGGING_FACE_MODEL_URL = "https://huggingface.co/Yogendra12/plant-disease-classifier/resolve/main/plant_disease_model.pth" 

# Model ke liye class names (Aapke original dataset se match hone chahiye)
CLASS_NAMES = [
    "Healthy", 
    "Bacterial Blight", 
    "Early Blight", 
    "Late Blight",
    # Agar aapke paas aur classes hain toh yahan daalein
]
NUM_CLASSES = len(CLASS_NAMES)


# ----------------- MODEL LOADING FUNCTION -----------------

# Model ko memory mein cache karna (sirf ek baar load hoga)
@st.cache_resource
def load_model():
    """Hugging Face se ResNet model weights download aur load karta hai."""
    st.info("🌐 Hugging Face se ResNet model weights download ki jaa rahi hain (Sirf pehli baar).")
    
    try:
        # Weights ko URL se download karke load karna
        # Ab yeh function seedha binary data download karega, HTML nahi.
        weights = torch.hub.load_state_dict_from_url(
            HUGGING_FACE_MODEL_URL, 
            map_location='cpu', # CPU par load karna sabse achha hai
            progress=True
        )
        
        # FINAL MODEL ARCHITECTURE DEFINITION (ResNet-18)
        # Agar aapne ResNet-18 ke bajaye koi aur model (jaise ResNet-50) use kiya tha, 
        # toh 'models.resnet18' ko 'models.resnet50' se badal dein.
        model = models.resnet18(pretrained=False) 

        # Final Fully Connected layer ko custom classes ke anusaar badalna
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
        
        # Weights ko model mein load karna
        model.load_state_dict(weights, strict=False) 
        model.eval() # Evaluation mode mein set karna
        st.success("✅ ResNet Model weights safaltapoorvak (successfully) load ho gaye hain!")
        return model
    
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Download Error: HTTP Status {e.response.status_code}. Link ya File ID check karein.")
        st.error("Kripya Hugging Face ka direct download link (jis par end mein `resolve/main/filename` ho) check karein.")
        return None
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        st.error(f"Pक्का करें ki model architecture (ResNet-18) aapke weights file se match karta hai. Original error: {e}")
        return None

# ----------------- PREDICTION AND UI LOGIC -----------------

def classify_image(model, image):
    """Image ko pre-process karta hai aur prediction deta hai."""
    
    # Image transformations: ResNet training ke anusaar
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # ResNet standard input size
        transforms.ToTensor(),
        # Standard ImageNet normalization values (Yeh aapke training se match honi chahiye)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Image ko tensor mein badalna
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0) # Batch dimension add karna
        
        # Model se prediction
        with torch.no_grad():
            # Model ko CPU par chalana
            output = model(input_batch)
            
        # Output ko probabilities mein badalna
        probabilities = F.softmax(output[0], dim=0)
        
        # Sabse zyada probability wala result
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index].item()
        
        return predicted_class, confidence

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return "Error", 0.0

# ----------------- STREAMLIT APP -----------------

def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

    st.markdown("""
        <style>
            .stApp {background-color: #f0f2f6;}
            .main-header {color: #1E90FF; text-align: center; font-size: 2.5em; margin-bottom: 0.5em;}
            /* Button styling for better look */
            .stButton>button {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                border-radius: 12px;
                padding: 10px 24px;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
        </style>
        <h1 class="main-header">🍃 Plant Disease Classifier</h1>
        <p style="text-align: center; color: #555;">Kisaan ki madad ke liye - Poudhe ke patte ki tasveer upload karein aur rog ki jaankari paayein.</p>
    """, unsafe_allow_html=True)
    
    # Model Load Karna
    model = load_model()

    if model is None:
        st.warning("Model load nahi ho paya. Kripya upar diye gaye errors ko theek karein.")
        return

    st.subheader("🖼️ Tasveer Upload Karein")
    uploaded_file = st.file_uploader(
        "PNG/JPG format mein poudhe ke patte ki tasveer chunein...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Image ko PIL format mein kholna
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            with col2:
                # Prediction button
                if st.button("🔍 Rog (Disease) Pehchanein"):
                    with st.spinner('Pehchaan ki jaa rahi hai...'):
                        predicted_class, confidence = classify_image(model, image)

                    st.markdown("---")
                    st.subheader("🎯 Nateeja (Result)")
                    
                    if predicted_class == "Error":
                         st.error("Prediction mein internal error aaya.")
                    else:
                        st.metric(
                            label="Pehchana Gaya Rog", 
                            value=predicted_class
                        )
                        # Confidence ko progress bar se dikhana
                        st.progress(confidence) 
                        st.write(f"Vishwaas Star (Confidence): **{confidence*100:.2f}%**")
                        
                        if predicted_class == "Healthy":
                            st.success("Bahut badhiya! Poudha swasth (healthy) lagta hai.")
                        else:
                            st.warning(f"Savdhani! Yeh {predicted_class} ho sakta hai. Ilaaj ke liye salah lein.")


        except Exception as e:
            st.error(f"Tasveer process karne mein error: {e}")

if __name__ == '__main__':
    main()
