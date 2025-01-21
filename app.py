import streamlit as st
import os
from dotenv import load_dotenv
import base64
import openai
import requests
from transformers import pipeline,AutoTokenizer, AutoModelForCausalLM

# Set up environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the environment variables.")

openai.api_key = openai_api_key

# Convert image to base64 encoding
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error converting image to base64: {e}")

# Function to call OpenAI API with fallback to Dolphin 2.9.1 Llama 3 70B model
def call_openai_api(prompt):
    try:
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        if "Rate limit" in str(e):
            st.warning("Rate limit exceeded. Switching to Dolphin 2.9.1 Llama 3 70B model.")
            return call_dolphin_llama_model(prompt)
        else:
            st.error(f"An error occurred: {e}")
            return None

# Function to call Dolphin 2.9.1 Llama 3 70B model
# Initialize the pipeline
pipe = pipeline("text-generation", model="cognitivecomputations/dolphin-2.9.1-llama-3-70b")
def call_dolphin_llama_model(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = pipe(messages)
    return response[0]['generated_text']

# Set up custom CSS for background and UI
def set_background():
    # Direct URLs to the images
    image_url_1 = "https://raw.githubusercontent.com/rakshithjm97/iq/main/pexels-rickyrecap-1926988.jpg"

def get_response_from_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    if response.choices:
        return response.choices[0].message['content']
    else:
        return "No response from OpenAI API"

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
            response = get_response_from_openai(user_input)
            st.markdown(f"**Response:** {response}")

        # Add a fixed footer
        st.markdown(
            """
            <div class="footer">
                <p>Powered by OpenAI</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
