import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from PIL import Image
import os
import requests
from io import BytesIO

# Set up the app layout
st.set_page_config(
    page_title="MediBot - Your Personal Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and display images
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Header with logo
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2781/2781395.png", width=100)
with header_col2:
    st.title("MediBot - Your Personal Medical Assistant")
    st.caption("AI-powered medical diagnosis and treatment recommendations")

# Sidebar with information
with st.sidebar:
    st.header("About MediBot")
    st.image("https://cdn-icons-png.flaticon.com/512/3309/3309933.png", width=100)
    st.write("""
    MediBot is an AI-powered medical assistant that can:
    - Provide preliminary disease diagnosis based on symptoms
    - Suggest appropriate medications
    - Offer health recommendations
    - Consider regional healthcare guidelines
    
    Note: This is for informational purposes only. Always consult a real doctor for medical advice.
    """)
    
    st.divider()
    st.subheader("Example Questions")
    st.write("- I have fever, headache, and muscle pain. What could it be?")
    st.write("- What's the treatment for seasonal allergies in India?")
    st.write("- I have a rash on my arms and itching. What should I do?")
    
    st.divider()
    st.write("Developed by [Your Name]")
    st.write("Version 1.0")

# Initialize LangChain
def initialize_llm():
    # You can switch between OpenAI and HuggingFace here
    # For OpenAI (make sure to set OPENAI_API_KEY in environment variables)
    # llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-instruct")
    
    # For HuggingFace (make sure to set HUGGINGFACEHUB_API_TOKEN)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    
    # Create prompt template
    template = """
    You are a highly experienced medical doctor named MediBot. Your task is to:
    1. Analyze the patient's symptoms and medical history
    2. Provide a possible diagnosis (list possible conditions by likelihood)
    3. Recommend appropriate medications (considering the patient's country: {country})
    4. Suggest lifestyle recommendations
    5. Advise when to seek immediate medical attention
    
    Current conversation:
    {history}
    
    Patient: {input}
    MediBot:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input", "country"], 
        template=template
    )
    
    # Create memory for conversation
    memory = ConversationBufferMemory(memory_key="history")
    
    # Create LLM chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    
    return llm_chain

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = initialize_llm()

# Main chat interface
def main():
    # Country selection
    country = st.selectbox(
        "Select your country for region-specific recommendations:",
        ("United States", "India", "United Kingdom", "Canada", "Australia", "Germany", "France", "Japan", "Brazil"),
        index=0
    )
    
    # Display conversation history
    st.subheader("Consultation Chat")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.conversation:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Describe your symptoms or ask a medical question...")
    
    if user_input:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing your symptoms..."):
            # Get response from LLM
            response = st.session_state.llm_chain.run(
                input=user_input,
                country=country
            )
            
            # Add assistant response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()

# Additional features
def additional_features():
    st.divider()
    
    # Three columns for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Common Conditions")
        if st.button("Cold & Flu"):
            st.session_state.conversation.append({"role": "user", "content": "What are the symptoms and treatment for cold and flu?"})
            st.rerun()
        if st.button("Allergies"):
            st.session_state.conversation.append({"role": "user", "content": "How to treat seasonal allergies?"})
            st.rerun()
        if st.button("Headache"):
            st.session_state.conversation.append({"role": "user", "content": "What causes headaches and how to relieve them?"})
            st.rerun()
    
    with col2:
        st.subheader("First Aid")
        if st.button("Burns"):
            st.session_state.conversation.append({"role": "user", "content": "What's the first aid for minor burns?"})
            st.rerun()
        if st.button("Cuts"):
            st.session_state.conversation.append({"role": "user", "content": "How to treat minor cuts and wounds?"})
            st.rerun()
        if st.button("Fever"):
            st.session_state.conversation.append({"role": "user", "content": "When should I worry about a fever?"})
            st.rerun()
    
    with col3:
        st.subheader("Wellness Tips")
        if st.button("Sleep Better"):
            st.session_state.conversation.append({"role": "user", "content": "How can I improve my sleep quality?"})
            st.rerun()
        if st.button("Healthy Diet"):
            st.session_state.conversation.append({"role": "user", "content": "What foods should I eat for better health?"})
            st.rerun()
        if st.button("Stress Relief"):
            st.session_state.conversation.append({"role": "user", "content": "What are effective ways to reduce stress?"})
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
    additional_features()
    
    # Disclaimer
    st.divider()
    st.warning("""
    **Important Disclaimer:** 
    MediBot is an AI assistant for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)
