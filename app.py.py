


import streamlit as st
import pickle
import re
import string


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>+', '', text)                  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
    text = re.sub(r'\n', '', text)                     
    text = re.sub(r'\w*\d\w*', '', text)                
    return text


@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as vf:
            vectorizer = pickle.load(vf)
        with open('model.pkl', 'rb') as mf:
            model = pickle.load(mf)
        return vectorizer, model
    except FileNotFoundError:
        return None, None

vectorizer, model = load_models()


st.title("📰 Fake News Detector")
st.markdown("Enter news text or a headline below to check whether it is predicted as **Real** or **Fake**.")


user_input = st.text_area("News Text / Article Headline:", height=180, placeholder="Paste article text here...")

if st.button("Predict", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif vectorizer is None or model is None:
        st.error("Model files (`model.pkl` or `vectorizer.pkl`) not found! Please run the training script first.")
    else:
        
        cleaned_input = clean_text(user_input)
        
        
        vec_input = vectorizer.transform([cleaned_input])
        
        
        prediction = model.predict(vec_input)[0]
        
        st.divider()
        
       
        if prediction == 1 or str(prediction).lower() == 'fake':
            st.error("🚨 **Prediction: Fake News**")
        else:
            st.success("✅ **Prediction: Real News**")
            
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec_input)[0]
            confidence = max(probs) * 100
            st.info(f"**Confidence Score:** {confidence:.2f}%")







