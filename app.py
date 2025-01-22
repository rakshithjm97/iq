import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face
@st.cache_resource()
def load_model():
    model_name = "meta-llama/Llama-3.2-1B"  # Llama model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Generate chat response using the model
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app UI
def main():
    st.title("Llama-3.2-1B Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    model, tokenizer = load_model()

    with st.form("chat_form"):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        # Display user input
        st.session_state["messages"].append(f"**You**: {user_input}")
        st.write(f"**You**: {user_input}")
        
        # Generate response
        with st.spinner("Llama-3.2-1B is typing..."):
            response = generate_response(model, tokenizer, user_input)
            st.session_state["messages"].append(f"**Llama-3.2-1B**: {response}")
            st.write(f"**Llama-3.2-1B**: {response}")
    
    # Display chat history
    if st.session_state["messages"]:
        st.markdown("---")
        for message in st.session_state["messages"]:
            st.write(message)

if __name__ == "__main__":
    main()
