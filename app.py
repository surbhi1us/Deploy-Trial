import streamlit as st
import tensorflow as tf
import numpy as np
 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")
 
model = load_model()
 
st.title("XOR Prediction App")
 
x1 = st.number_input("Enter X1 (0 or 1)", min_value=0, max_value=1, step=1)
x2 = st.number_input("Enter X2 (0 or 1)", min_value=0, max_value=1, step=1)
 
if st.button("Predict"):
    pred = model.predict(np.array([[x1, x2]]))[0][0]
    st.write(f"Prediction: {round(pred)} (Probability: {pred:.4f})")