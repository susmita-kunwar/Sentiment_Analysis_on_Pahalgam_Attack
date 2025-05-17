import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

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

# Get current directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Load vectorizer and model safely
try:
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit UI
st.title("Sentiment Analysis: Pahalgam Comments")

input_comment = st.text_area("Enter a comment")

if st.button('Predict Sentiment'):
    if input_comment.strip() == "":
        st.warning("Please enter a valid comment.")
    else:
        transformed_comment = transform_text(input_comment)
        vector_input = tfidf.transform([transformed_comment])
        result = model.predict(vector_input)[0]

        if result == 2:
            st.header("Sentiment: **Positive** 🙂")
        elif result == 0:
            st.header("Sentiment: **Negative** 🙁")
        elif result == 1:
            st.header("Sentiment: **Neutral** 😐")
        else:
            st.header("Unexpected sentiment class predicted.")
