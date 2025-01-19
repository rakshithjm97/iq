import streamlit as st
import os
from dotenv import load_dotenv
import base64
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch # Required for Hugging Face models

# Set up OpenAI API key handling directly from environment variables
openai.api_key = 'sk-9wnVyzYO0xzlJ-0H_OgvWGvBnnSe_9BCQfYMOIaTkLT3BlbkFJ83hEmXWmvhT-bwv2L_ITbB7BL8s0fz97kCVjLxH90A'  # Replace with your actual API key
  # Replace with your actual API key

# Convert image to base64 encoding
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Background image not found at {image_path}")
        return None

# Set up custom CSS for background and UI
def set_background():
    # Direct URLs to the images
    image_url_1 = "https://raw.githubusercontent.com/rakshithjm97/iq/main/pexels-rickyrecap-1926988.jpg"
    image_url_2 = "https://raw.githubusercontent.com/rakshithjm97/iq/main/pexels-lilartsy-1925536.jpg"
    
    # Set the background using the URLs
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url_1}) no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::after {{
            content: "";
            background: url({image_url_2}) no-repeat center center fixed;
            background-size: cover;
            position: absolute;
            top: 100%;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }}
        .header {{
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            margin-top: 20px;
            color: white;
        }}
        .description {{
            text-align: center;
            font-size: 24px;
            margin-top: 10px;
            color: white;
        }}
        .question {{
            text-align: center;
            font-size: 24px;
            margin-top: 20px;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_ai_response(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Updated model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError:
        # Fallback to Meta's LLM 3B Instruct model
        return get_meta_llm_response(question)

def get_meta_llm_response(question):
    # Load the model and tokenizer from Hugging Face
    model_name = "facebook/opt-1.3b"  # Example model, replace with the actual model you want to use
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create a pipeline for text generation
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    # Generate a response
    response = generator(question, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    # Set the background and custom styles
    set_background()
    
    # Create a scrollable container for the entire page content
    with st.container():
        # Title of the app
        st.markdown('<div class="header">Welcome to AI Tutor (Mia)</div>', unsafe_allow_html=True)
        st.markdown('<div class="description">Your AI-powered tutor to help you with anything you need!</div>', unsafe_allow_html=True)
        
        # Create a scrollable container for questions and answers
        with st.container():
            st.markdown('<div class="question">Ask Mia a question:</div>', unsafe_allow_html=True)
            question = st.text_input("", 
                                   placeholder="Type your question here...",
                                   help="Ask any question and Mia will help you learn!")
            
            if question:
                with st.spinner('Mia is thinking...'):
                    answer = get_ai_response(question)
                    if answer:
                        st.markdown(
                            f'<div class="answer-container">{answer}</div>',
                            unsafe_allow_html=True
                        )

        # Add a fixed footer
        st.markdown(
            """
            <div class="footer">
                <p>Powered by OpenAI <p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
