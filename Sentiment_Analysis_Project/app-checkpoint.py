import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Download NLTK dependencies
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    y = [i for i in tokens if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return ' '.join(y)

# Load model artifacts
@st.cache_resource
def load_model():
    try:
        with open('sentiment_analysis_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

artifacts = load_model()
model = artifacts['model']
vectorizer = artifacts['vectorizer']
encoder = artifacts['encoder']

# Streamlit UI
st.title("Sentiment Analysis: Pahalgam Comments")

input_comment = st.text_area("Enter a comment")

if st.button('Predict Sentiment'):
    if input_comment.strip() == "":
        st.warning("Please enter a valid comment.")
    else:
        # Preprocess the input
        transformed_comment = transform_text(input_comment)
        
        # Vectorize the input
        vector_input = vectorizer.transform([transformed_comment])
        
        # Make prediction
        result = model.predict(vector_input)[0]
        
        # Display result
        if result == 2:
            st.success("Sentiment: Positive")
        elif result == 0:
            st.error("Sentiment: Negative")
        elif result == 1:
            st.info("Sentiment: Neutral")
        else:
            st.warning("Unexpected sentiment class predicted.")

        # Show confidence scores if available
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(vector_input)[0]
            st.write("Confidence scores:")
            st.write(f"Negative: {probas[0]:.2f}")
            st.write(f"Neutral: {probas[1]:.2f}")
            st.write(f"Positive: {probas[2]:.2f}")