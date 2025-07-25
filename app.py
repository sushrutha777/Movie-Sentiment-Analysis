import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import re

# Load the GRU model
model_path = 'gru_model.keras'
model = load_model(model_path, custom_objects=None, compile=False, safe_mode=False)

max_features = 10000
max_len = 500
word_index = imdb.get_word_index()

def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]
    encoded = [i if i < max_features else 2 for i in encoded]
    return sequence.pad_sequences([encoded], maxlen=max_len)

# Sample review categories
positive_reviews = [
    "Amazing soundtrack, perfect pacing, and visuals that made the experience feel magical and cinematic.",
    "This movie was fantastic! The acting was great and the plot was thrilling.",
    "Exceeded expectations with inspiring storytelling, top-notch acting, and a powerful emotional message.",
 
]

negative_reviews = [
    "Visual effects were cheap, editing inconsistent, and the narrative failed to engage at all.",
    "Started strong but didn't maintain the energy or emotional impact",
    "Terrible experience! The film dragged endlessly and made no sense at all.",
]
# Streamlit configuration
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below and let the GRU model predict its sentiment!")

# Display sample categories
with st.expander("📁 Sample Review Library (Click to Explore)"):
    st.markdown("**Positive Reviews**")
    for rev in positive_reviews:
        st.markdown(f"- {rev}")
    st.markdown("**Negative Reviews**")
    for rev in negative_reviews:
        st.markdown(f"- {rev}")

# User input
review = st.text_area("💬 Enter your movie review (at least 10 words):")

# Prediction logic
if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    elif len(review.split()) < 10:
        st.warning("Please enter at least 10 words for better sentiment prediction.")
    else:
        padded = preprocess_text(review)
        prob = model.predict(padded)[0][0]
        sentiment = "Positive 😊" if prob >= 0.5 else "Negative 😞"

        st.subheader("🎯 Prediction Result")
        if prob >= 0.5:
            st.success(f"**Sentiment:** {sentiment}")
        else:
            st.error(f"**Sentiment:** {sentiment}")
        st.info(f"Confidence Score: {prob:.2f}")
