import streamlit as st
import os
from dotenv import load_dotenv
import base64
from transformers.pipelines import pipeline  # Updated import for pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Convert image to base64 encoding
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error converting image to base64: {e}")

# Function to call Dolphin 2.9.1 Llama 3 70B model
# Initialize the pipeline
pipe = pipeline("text-generation", model="cognitivecomputations/dolphin-2.9.1-llama-3-70b", 
                tokenizer=AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.9.1-llama-3-70b"), 
                device=0 if torch.cuda.is_available() else -1)

def call_dolphin_llama_model(prompt):
    response = pipe(prompt, max_length=150)
    return response[0]['generated_text']

# Set up custom CSS for background and UI
def set_background():
    image_url = "https://raw.githubusercontent.com/rakshithjm97/iq/main/pexels-rickyrecap-1926988.jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .header {{
            font-size: 2.5em;
            font-weight: bold;
            color: #FFFFFF;
            text-align: center;
        }}
        .description {{
            font-size: 1.2em;
            color: #FFFFFF;
            text-align: center;
        }}
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
        }}
        </style>
        """, unsafe_allow_html=True
    )

def main():
    # Set the background and custom styles
    set_background()
    
    # Create a scrollable container for the entire page content
    with st.container():
        # Title of the app
        st.markdown('<div class="header">Welcome to AI Tutor (Mia)</div>', unsafe_allow_html=True)
        st.markdown('<div class="description">Your AI-powered tutor to help you with anything you need!</div>', unsafe_allow_html=True)
        
        # Input field for user to type their question
        user_input = st.text_input("Type your question here:")

        if user_input:
            # Use Dolphin 2.9.1 Llama 3 70B model to get response
            response = call_dolphin_llama_model(user_input)
            st.markdown(f"**Response:** {response}")

        # Add a fixed footer
        st.markdown(
            """
            <div class="footer">
                <p>Powered by Dolphin 2.9.1 Llama 3 70B Model</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
