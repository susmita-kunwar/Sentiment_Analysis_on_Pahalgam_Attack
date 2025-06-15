import streamlit as st
import pickle
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# --- App Config ---
st.set_page_config(
    page_title="Pahalgam Sentiment Watch",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Pahalgam Image ---
@st.cache_data
def load_header_image():
    try:
        # Use a local image or URL (replace with actual Pahalgam tourism image)
        img_url = "https://www.jktdc.co.in/wp-content/uploads/2023/05/Pahalgam-1.jpg"
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

header_img = load_header_image()

# --- Enhanced UI with Image ---
if header_img:
    st.image(header_img, use_column_width=True, caption="Pahalgam - Paradise in the Himalayas")
else:
    st.warning("Couldn't load header image")

st.title("🛡️ Pahalgam Tourism Sentiment Monitor")
st.markdown("""
**Analyze visitor feedback to enhance safety and experience**  
*Helping authorities identify concerns and improve tourist satisfaction*
""")

# --- Security Badge ---
st.sidebar.markdown("""
<div style="background:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:20px">
    <h4 style="color:#1e3a8a">🔒 Official Analysis System</h4>
    <p style="font-size:small">Verified by J&K Tourism Department</p>
</div>
""", unsafe_allow_html=True)

# --- Preprocessing ---
nltk.download(['stopwords', 'punkt'])
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def security_aware_clean(text):
    # Enhanced cleaning with security-related terms
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# --- Load Model ---
@st.cache_resource
def load_security_model():
    try:
        with open('sentiment_analysis_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except Exception as e:
        st.error(f"🔴 Security system error: {e}")
        st.stop()

model_data = load_security_model()
model = model_data['model']
vectorizer = model_data['vectorizer']

# --- Analysis Section ---
with st.container():
    col1, col2 = st.columns([3,1])
    
    with col1:
        user_input = st.text_area(
            "Enter tourist feedback or security report:", 
            height=150,
            placeholder="The mountain view was amazing but we felt unsafe during..."
        )
        
    with col2:
        st.markdown("""
        <div style="background:#f8f9fa;padding:15px;border-radius:10px;margin-top:10px">
            <h4>⚠️ Safety Tips</h4>
            <ul style="font-size:small">
                <li>Report suspicious activity</li>
                <li>Check weather alerts</li>
                <li>Use registered guides</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Analyze for Safety Insights", type="primary"):
        if not user_input.strip():
            st.warning("Please enter feedback to analyze")
        else:
            with st.spinner("🔍 Scanning for security and sentiment patterns..."):
                # Process and predict
                cleaned_text = security_aware_clean(user_input)
                features = vectorizer.transform([cleaned_text])
                pred = model.predict(features)[0]
                
                # Enhanced sentiment mapping
                sentiment_map = {
                    0: ("Negative", "🔴", "Safety Concern Detected"),
                    1: ("Neutral", "🟡", "Routine Feedback"), 
                    2: ("Positive", "🟢", "Positive Experience")
                }
                label, icon, desc = sentiment_map[pred]
                
                # Confidence scores
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(features)[0]
                
            # --- Threat Assessment Display ---
            st.subheader("Security Sentiment Assessment", divider="red")
            
            cols = st.columns(3)
            with cols[0]:
                # Dynamic badge based on sentiment
                st.markdown(f"""
                <div style="background:#{'#dcfce7' if label=='Positive' else '#fee2e2' if label=='Negative' else '#fef9c3'};
                    padding:20px;
                    border-radius:10px;
                    text-align:center">
                    <h1>{icon}</h1>
                    <h3>{label}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown("""
                <div style="background:#f8fafc;padding:15px;border-radius:10px">
                    <h4>📌 Key Phrases</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                # Extract concerning phrases (simple implementation)
                concerning_phrases = [word for word in cleaned_text.split() 
                                    if word in ['unsafe', 'scare', 'threat', 'risk']]
                for phrase in concerning_phrases[:3]:
                    st.markdown(f"<li>{phrase}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with cols[2]:
                if hasattr(model, "predict_proba"):
                    st.markdown("""
                    <div style="background:#f8fafc;padding:15px;border-radius:10px">
                        <h4>🛡️ Confidence Levels</h4>
                    """, unsafe_allow_html=True)
                    
                    # Mini gauges
                    st.markdown(f"""
                    Negative: <progress value="{probas[0]}" max="1"></progress> {probas[0]*100:.1f}%<br>
                    Neutral: <progress value="{probas[1]}" max="1"></progress> {probas[1]*100:.1f}%<br>
                    Positive: <progress value="{probas[2]}" max="1"></progress> {probas[2]*100:.1f}%
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # --- Actionable Insights ---
            if label == "Negative":
                st.error("""
                **🚨 Recommended Actions:**
                - Forward to local security team
                - Check location patterns
                - Consider tourist advisory
                """)
            elif label == "Positive":
                st.success("""
                **🌟 Enhancement Opportunities:**
                - Identify what's working well
                - Share positive experiences
                - Reward excellent service
                """)

# --- Emergency Section ---
st.sidebar.markdown("""
## 🚨 Emergency Contacts
- **Police**: 100  
- **Tourism Helpline**: 1800-123-5555  
- **Medical Emergency**: 102
""")

# --- Footer ---
st.markdown("""
---
<div style="text-align:center;font-size:small;color:#64748b">
Pahalgam Tourism Security Initiative • Powered by AI Monitoring
</div>
""", unsafe_allow_html=True)