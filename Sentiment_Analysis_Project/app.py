import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# --- NLTK Initialization (Fixed) ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Preprocessing Function ---
def transform_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

# --- Model Loading ---
@st.cache_resource
def load_model():
    with open('sentiment_analysis_model.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_model()
model = artifacts['model']
vectorizer = artifacts['vectorizer']

# --- Streamlit App ---
st.title("Pahalgam Tourism Sentiment Analysis")
st.markdown("Analyze visitor feedback about Pahalgam attractions")

input_text = st.text_area("Enter a tourism-related comment:", height=150)

if st.button("Analyze Sentiment"):
    if not input_text.strip():
        st.warning("Please enter a comment")
    else:
        with st.spinner("Processing..."):
            # Transform and predict
            processed_text = transform_text(input_text)
            vector_input = vectorizer.transform([processed_text])
            prediction = model.predict(vector_input)[0]
            
            # Display results
            sentiment_map = {
                0: ("Negative", "Safety concern detected"),
                1: ("Neutral", "Neutral feedback"), 
                2: ("Positive", "Positive experience")
            }
            label, icon, desc = sentiment_map[prediction]
            
            # Show confidence scores if available
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(vector_input)[0]
                confidence = {k: v for k, v in zip(["Negative", "Neutral", "Positive"], probas)}
            
            # Result display
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment")
                if label == "Positive":
                    st.success(f"{icon} {label}")
                elif label == "Negative":
                    st.error(f"{icon} {label}")
                else:
                    st.info(f"{icon} {label}")
                st.caption(desc)
            
            with col2:
                if hasattr(model, "predict_proba"):
                    st.subheader("Confidence")
                    for sentiment, score in confidence.items():
                        st.progress(score, text=f"{sentiment}: {score:.1%}")
            
            # Show processed text
            with st.expander("See processed text"):
                st.code(processed_text)