import google.generativeai as genai  
from dotenv import load_dotenv
import streamlit as st
import os
import pickle
from PIL import Image
import io
import hashlib
load_dotenv()
LLM_call = os.getenv("LLM_call")
if not LLM_call:
    st.error("Please configure your .env file.")
genai.configure(api_key=LLM_call)
def get_gemini_response(image, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")  
    response = model.generate_content([image, prompt])
    return response.text if response else "Error: No response from LLM."
def generate_invoice_id(image):
    """Generates a unique hash ID for the invoice using image bytes."""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return hashlib.md5(image_bytes.getvalue()).hexdigest()  
def save_to_pickle(data, filename="invoice_data.pkl"):
    """Saves extracted invoice details to a pickle file (appending new invoices)."""
    existing_data = load_from_pickle(filename)  
    existing_data.update(data)  
    
    with open(filename, "wb") as f:
        pickle.dump(existing_data, f)
def load_from_pickle(filename="invoice_data.pkl"):
    """Loads extracted invoice details from a pickle file."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return {}
st.set_page_config(page_title="Welcome to Invoice Analyzer")
st.header("📄 Invoice Analyzer")
uploaded_files = st.file_uploader("Upload invoice images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
input_prompt = """
You are an expert in understanding invoices.
You will receive images of invoices and extract relevant details from them in a tabulated manner every time with the same format.
If a detail is not available, put 'NIL'.
"""
if uploaded_files:
    invoice_data = {}  

    for uploaded_file in uploaded_files:
      
        image = Image.open(uploaded_file)
        st.image(image, caption=f"📸 Uploaded: {uploaded_file.name}", use_container_width=True)

        
        invoice_id = generate_invoice_id(image)

       
        with st.spinner(f"⏳ Extracting details for {uploaded_file.name}..."):
            response = get_gemini_response(image, input_prompt)

       
        st.subheader(f"📜 Extracted Details for {uploaded_file.name}")
        st.write(response)

       
        invoice_data[invoice_id] = {
            "filename": uploaded_file.name,
            "invoice_details": response
        }

    save_to_pickle(invoice_data)
    st.success("✅ All invoices saved in 'invoice_data.pkl'")
