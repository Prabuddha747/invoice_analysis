import streamlit as st  # Import Streamlit before setting page config

# Set Page Configuration (MUST BE FIRST)
st.set_page_config(page_title="Welcome to Invoice Analyzer")

import google.generativeai as genai  
import pickle
from PIL import Image
import io
import hashlib

# Get API Key from Streamlit secrets
LLM_call = st.secrets.get("LLM_call")  # Using .get to avoid errors if key is missing
if not LLM_call:
    st.error("API key not found. Please configure your secrets.")
else:
    genai.configure(api_key=LLM_call)  # Configure Gemini API with the API key

# Function to get response from Gemini AI
def get_gemini_response(image, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")  
    response = model.generate_content([image, prompt])
    return response.text if response else "Error: No response from LLM."

# Function to generate a unique ID for each invoice
def generate_invoice_id(image):
    """Generates a unique hash ID for the invoice using image bytes."""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return hashlib.md5(image_bytes.getvalue()).hexdigest()  # Unique ID for each image

# Function to save extracted data to a pickle file
def save_to_pickle(data, filename="invoice_data.pkl"):
    """Saves extracted invoice details to a pickle file (appending new invoices)."""
    existing_data = load_from_pickle(filename)  # Load existing data
    existing_data.update(data)  # Add new invoice(s)
    
    with open(filename, "wb") as f:
        pickle.dump(existing_data, f)

# Function to load previously saved invoice data
def load_from_pickle(filename="invoice_data.pkl"):
    """Loads extracted invoice details from a pickle file."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return {}

# Streamlit UI setup
st.header("📄 Invoice Analyzer")

# Upload Multiple Invoice Images
uploaded_files = st.file_uploader("Upload invoice images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Fixed input prompt
input_prompt = """
You are an expert in understanding invoices.
You will receive images of invoices and extract relevant details from them in a tabulated manner every time with the same format.
If a detail is not available, put 'NIL'.
"""

# Process images as soon as they are uploaded
if uploaded_files:
    invoice_data = {}  # Dictionary to store extracted details

    for uploaded_file in uploaded_files:
        # Load and display each image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"📸 Uploaded: {uploaded_file.name}", use_container_width=True)

        # Generate a unique ID for each invoice
        invoice_id = generate_invoice_id(image)

        # Process the invoice
        with st.spinner(f"⏳ Extracting details for {uploaded_file.name}..."):
            response = get_gemini_response(image, input_prompt)

        # Display extracted details
        st.subheader(f"📜 Extracted Details for {uploaded_file.name}")
        st.write(response)

        # Save invoice details with unique ID
        invoice_data[invoice_id] = {
            "filename": uploaded_file.name,
            "invoice_details": response
        }

    # Save all extracted invoices to a pickle file
    save_to_pickle(invoice_data)
    st.success("✅ All invoices saved in 'invoice_data.pkl'")

# Button to load previous invoices
if st.button("📂 Load All Saved Invoice Data"):
    loaded_data = load_from_pickle()
    
    if loaded_data:
        st.subheader("📂 Previously Saved Invoices")
        for invoice_id, data in loaded_data.items():
            st.markdown(f"### 🏷️ Invoice: {data['filename']}")
            st.write(data["invoice_details"])
            st.markdown("---")  # Separator
    else:
        st.error("⚠️ No previously saved invoice data found.")
