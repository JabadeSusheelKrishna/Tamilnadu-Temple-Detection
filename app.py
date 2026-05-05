import streamlit as st
from PIL import Image
import json
from backend import TempleClassifier
import time

# Page Configuration
st.set_page_config(
    page_title="Tamil Nadu Temple Recognition",
    page_icon="🛕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

    :root {
        --primary: #FF9933;
        --secondary: #800000;
        --background: #FFFBF0;
        --card-bg: #FFFFFF;
    }

    .main {
        background-color: var(--background);
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: var(--secondary);
    }

    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 153, 51, 0.4);
    }

    .prediction-card {
        background: black;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border-left: 10px solid var(--primary);
        margin-bottom: 2rem;
    }

    .temple-title {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .location-tag {
        background: #f0f0f0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #666;
    }

    .metric-container {
        display: flex;
        gap: 2rem;
        margin: 1rem 0;
    }

    .info-section {
        background: #fdfdfd;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #eee;
    }
    
    .sidebar-text {
        font-size: 0.9rem;
        color: #444;
    }

    .info-item {
        background: white;
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary);
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #333;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Backend
@st.cache_resource
def get_classifier():
    return TempleClassifier()

classifier = get_classifier()

# Sidebar
with st.sidebar:
    st.markdown("# 🛕 TN Temple AI")
    st.markdown("---")
    version = st.selectbox(
        "Select Model Version",
        ["V1: Baseline Zero-Shot", 
         "V2: Prompt Engineering", 
         "V3: ROI Architecture Focus", 
         "V4: Image Enhancement", 
         "V5: Hybrid FFNN Approach"]
    )
    
    st.markdown("### 📋 Project Info")
    st.info("""
    **Topic**: T12.4 - Temples of TN
    **Team**: Naa Chaavu Nen Chastha
    **Member**: Susheel Krihna Jabade
    """)
    
    st.markdown("### 🚀 About this version")
    if "V1" in version:
        st.write("Uses raw temple names as labels for zero-shot classification.")
    elif "V2" in version:
        st.write("Uses context-rich prompts to improve CLIP's semantic understanding.")
    elif "V3" in version:
        st.write("Focuses on the central 70% of the image to capture architectural motifs.")
    elif "V4" in version:
        st.write("Applies sharpening and contrast normalization before inference.")
    elif "V5" in version:
        st.write("An ensemble method simulating a learned FFNN head for higher precision.")

# Main Header
st.markdown("<h1 style='text-align: center; color: #800000;'>Temples of Tamil Nadu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Explore the architectural wonders of the Sangam era and beyond with AI.</p>", unsafe_allow_html=True)

# Image Uploader
uploaded_file = st.file_uploader("Upload a temple image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, width="stretch", caption="Uploaded Image")
    
    with col2:
        with st.spinner("Analyzing architecture..."):
            # Progress bar for effect
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Predict
            if "V1" in version:
                result = classifier.predict_v1(image)
            elif "V2" in version:
                result = classifier.predict_v2(image)
            elif "V3" in version:
                result = classifier.predict_v3(image)
            elif "V4" in version:
                result = classifier.predict_v4(image)
            else:
                result = classifier.predict_v5(image)
            
        if result['is_temple']:
            temple_name = result['prediction']
            conf = result['confidence'] * 100
            metadata = classifier.metadata[temple_name]
            
            st.markdown(f"""
                <div class="prediction-card">
                    <span class="location-tag">📍 {metadata['location']}</span>
                    <h2 class="temple-title">{temple_name}</h2>
                    <div class="metric-container">
                        <div>
                            <p style='margin:0; font-size:0.8rem; color:#888;'>Confidence</p>
                            <p style='margin:0; font-size:1.5rem; font-weight:700; color:#FF9933;'>{conf:.1f}%</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📜 History", "🕒 Visit Info", "🗺️ Navigation", "📊 Technical"])
            
            with tab1:
                st.markdown("### Historical Significance")
                st.write(metadata['history'])
                
            with tab2:
                st.markdown("### Visitor Information")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🕒 Opening Hours")
                    for h in metadata['hours'].split(','):
                        if h.strip():
                            st.markdown(f'<div class="info-item">⏰ {h.strip()}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown("#### 🎫 Ticket Price")
                    # Split by comma or semicolon
                    tickets = metadata['tickets'].replace(';', ',').split(',')
                    for t in tickets:
                        if t.strip():
                            st.markdown(f'<div class="info-item">💰 {t.strip()}</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### Explore on Google Maps")
                st.link_button("Open in Google Maps", metadata['maps_url'])
                # Simple static map visualization (optional, since we have lat/long)
                st.map({"lat": [metadata['coordinates'][0]], "lon": [metadata['coordinates'][1]]})
                
            with tab4:
                st.markdown("### Prediction Probabilities")
                probs_data = result['all_probs'][:5] # Top 5
                for p in probs_data:
                    name = p['name']
                    prob = p['prob']
                    st.write(f"**{name}**")
                    st.progress(prob)
        else:
            st.warning("No famous Tamil Nadu temple identified in this image. Please try another photo.")
            with st.expander("Show detailed probabilities"):
                st.write(result['all_probs'][:5])

else:
    # Landing State
    st.markdown("---")
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Upload 📸")
        st.write("Upload a clear photo of a temple's gopuram, sanctum, or corridor.")
    with c2:
        st.markdown("#### 2. Analyze 🧠")
        st.write("CLIP model compares your image with 15 famous temple signatures.")
    with c3:
        st.markdown("#### 3. Discover 🏛️")
        st.write("Get instant history, visiting hours, and travel directions.")
    
    st.image("https://i.natgeofe.com/n/b9e9b8d1-fa08-4b90-96bb-310cace03847/meenakshi-amman-temple-india.jpg", caption="Meenakshi Amman Temple - One of the 15 temples we identify.", width=200)
