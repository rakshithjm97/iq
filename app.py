import streamlit as st
import os
from dotenv import load_dotenv
import base64
import openai
import aiml
#from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
#import torch # Required for Hugging Face models
import os

# Set up environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Convert image to base64 encoding
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error converting image to base64: {e}")
        return None

# Function to call OpenAI API with fallback to Dolphin 2.9.1 Llama 3 70B model
def call_openai_api(prompt):
    try:
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except openai.error.RateLimitError:
        st.warning("Rate limit exceeded. Switching to Dolphin 2.9.1 Llama 3 70B model.")
        # Fallback to Dolphin 2.9.1 Llama 3 70B model
        return call_dolphin_llama_model(prompt)

# Function to call Dolphin 2.9.1 Llama 3 70B model
def call_dolphin_llama_model(prompt):
    # Implement the call to Dolphin 2.9.1 Llama 3 70B model here
    # This is a placeholder implementation
    return "Response from Dolphin 2.9.1 Llama 3 70B model"

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

# Initialize the AIML kernel
kernel = aiml.Kernel()
try:
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")
except Exception as e:
    print(f"Error initializing AIML kernel: {e}")

def get_ai_response(question, kernel):
    try:
        response = kernel.respond(question)
        if not response:
            return "Sorry, I don't have an answer for that."
        return response
    except Exception as e:
        return str(e)

# Import necessary libraries for the commented-out function
#from transformers import AutoTokenizer, AutoModelForCausalLM

# def get_meta_llm_response(question):
#     # Load the model and tokenizer from Hugging Face
#     model_name = "facebook/opt-1.3b" 
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
    
#     # Create a pipeline for text generation
#     generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
#     # Generate a response
#     response = generator(question, max_length=150, num_return_sequences=1)
#     return response[0]['generated_text']

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
            st.text_input("Enter your name:", key="name_input")
            question = st.text_input("", 
                                   placeholder="Type your question here...",
                                   help="Ask any question and Mia will help you learn!")
            
            if question:
                with st.spinner('Mia is thinking...'):
                    answer = get_ai_response(question, kernel)
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

    # Example usage
    prompt = ""
    response = get_ai_response(prompt, kernel)
    st.write(response)

if __name__ == "__main__":
    main()