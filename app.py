import streamlit as st  # Import Streamlit before setting page config
import google.generativeai as genai  
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
from PIL import Image
import io
import hashlib

# Set Page Configuration
st.set_page_config(page_title="Invoice Analyzer", layout="wide")

# Load environment variables
load_dotenv()

# Get API Key
LLM_call = os.getenv("LLM_call")
if not LLM_call:
    st.error("🚨 API Key missing! Ensure the .env file is correctly set up.")
    st.stop()

genai.configure(api_key=LLM_call)

# Function to get response from Gemini AI
def get_gemini_response(image, prompt):
    """Sends invoice image and prompt to Gemini AI and returns extracted details."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  
        response = model.generate_content([image, prompt])
        return response.text if response else "⚠️ No response from LLM."
    except Exception as e:
        return f"🚨 Error: {str(e)}"

# Function to generate a unique ID for each invoice
def generate_invoice_id(image):
    """Generates a unique hash ID for the invoice using image bytes."""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return hashlib.md5(image_bytes.getvalue()).hexdigest()  # Unique ID for each image

# Function to save extracted data to a pickle file
def save_to_pickle(data, filename="invoice_data.pkl"):
    """Saves extracted invoice details to a pickle file."""
    existing_data = load_from_pickle(filename)  
    existing_data.extend(data)  # Append new invoices

    with open(filename, "wb") as f:
        pickle.dump(existing_data, f)

# Function to load previously saved invoice data
def load_from_pickle(filename="invoice_data.pkl"):
    """Loads extracted invoice details from a pickle file."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return []

# Streamlit UI setup
st.header("📄 Invoice Analyzer")

# Upload Multiple Invoice Images
uploaded_files = st.file_uploader("📂 Upload invoice images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Fixed input prompt for the LLM
input_prompt = """
You are an expert in understanding invoices.
You will receive images of invoices and extract relevant details from them in a tabulated manner every time with the same format.
If a detail is not available, put 'NIL'.
"""

# Process images as soon as they are uploaded
if uploaded_files:
    invoice_data = []  # List to store extracted invoice details
    existing_data = {entry["invoice_id"] for entry in load_from_pickle()}  # Load existing IDs

    for uploaded_file in uploaded_files:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"📸 Uploaded: {uploaded_file.name}", use_column_width=True)

        # Generate a unique ID for each invoice
        invoice_id = generate_invoice_id(image)

        # Skip processing if already saved
        if invoice_id in existing_data:
            st.warning(f"⚠️ Invoice '{uploaded_file.name}' has already been processed. Skipping...")
            continue

        # Process the invoice
        with st.spinner(f"⏳ Extracting details for {uploaded_file.name}..."):
            response = get_gemini_response(image, input_prompt)

        # Display extracted details
        st.subheader(f"📜 Extracted Details for {uploaded_file.name}")
        st.write(response)

        # Save invoice details
        invoice_data.append({
            "invoice_id": invoice_id,
            "filename": uploaded_file.name,
            "invoice_details": response
        })

    # Save extracted invoices to a pickle file
    if invoice_data:
        save_to_pickle(invoice_data)
        st.success("✅ New invoices saved in 'invoice_data.pkl'")

# Button to load previous invoices
if st.button("📂 Load All Saved Invoice Data"):
    loaded_data = load_from_pickle()
    
    if loaded_data:
        st.subheader("📂 Previously Saved Invoices")
        for data in loaded_data:
            st.markdown(f"### 🏷️ Invoice: {data['filename']}")
            st.write(data["invoice_details"])
            st.markdown("---")  # Separator
    else:
        st.error("⚠️ No previously saved invoice data found.")

# Button to Download All Extracted Data as CSV
if st.button("📥 Download Invoice Data as CSV"):
    loaded_data = load_from_pickle()

    if loaded_data:
        df = pd.DataFrame(loaded_data)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Click to Download", csv, "invoice_data.csv", "text/csv")
    else:
        st.error("⚠️ No data available to download.")
