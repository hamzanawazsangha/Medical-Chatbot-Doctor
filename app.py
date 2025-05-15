import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from PIL import Image
import requests
from io import BytesIO

# Set up the app layout
st.set_page_config(
    page_title="MediBot - Your Personal Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load image from internet
def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Header
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2781/2781395.png", width=100)
with header_col2:
    st.title("MediBot - Your Personal Medical Assistant")
    st.caption("AI-powered medical diagnosis and treatment recommendations")

# Sidebar
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
    st.markdown("- I have fever, headache, and muscle pain. What could it be?\n"
                "- What's the treatment for seasonal allergies in India?\n"
                "- I have a rash on my arms and itching. What should I do?")
    
    st.divider()
    st.write("Developed by [Your Name]")
    st.write("Version 1.0")

# Hugging Face LLM initialization
@st.cache_resource
def initialize_llm():
    try:
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-xxl",
            temperature=0.7,
            model_kwargs={"max_length": 512}
        )

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

        memory = ConversationBufferMemory(memory_key="history")

        return LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
            memory=memory
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

# Session state setup
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = initialize_llm()

# Main Chat Interface
def main():
    country = st.selectbox(
        "Select your country for region-specific recommendations:",
        ("United States", "India", "United Kingdom", "Canada", "Australia", 
         "Germany", "France", "Japan", "Brazil"),
        index=0
    )

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

    user_input = st.chat_input("Describe your symptoms or ask a medical question...")

    if user_input and st.session_state.llm_chain is not None:
        st.session_state.conversation.append({"role": "user", "content": user_input})

        with st.spinner("Analyzing your symptoms..."):
            try:
                response = st.session_state.llm_chain.invoke(
                    {"input": user_input, "country": country}
                )["text"]
                st.session_state.conversation.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")

# Extra buttons/features
def additional_features():
    st.divider()
    col1, col2, col3 = st.columns(3)

    def add_question_and_rerun(question):
        st.session_state.conversation.append({"role": "user", "content": question})
        st.rerun()

    with col1:
        st.subheader("Common Conditions")
        if st.button("Cold & Flu"):
            add_question_and_rerun("What are the symptoms and treatment for cold and flu?")
        if st.button("Allergies"):
            add_question_and_rerun("How to treat seasonal allergies?")
        if st.button("Headache"):
            add_question_and_rerun("What causes headaches and how to relieve them?")

    with col2:
        st.subheader("First Aid")
        if st.button("Burns"):
            add_question_and_rerun("What's the first aid for minor burns?")
        if st.button("Cuts"):
            add_question_and_rerun("How to treat minor cuts and wounds?")
        if st.button("Fever"):
            add_question_and_rerun("When should I worry about a fever?")

    with col3:
        st.subheader("Wellness Tips")
        if st.button("Sleep Better"):
            add_question_and_rerun("How can I improve my sleep quality?")
        if st.button("Healthy Diet"):
            add_question_and_rerun("What foods should I eat for better health?")
        if st.button("Stress Relief"):
            add_question_and_rerun("What are effective ways to reduce stress?")

# Educational Image Gallery
def sample_images_section():
    st.divider()
    st.subheader("🧠 Sample Medical Images (for educational use)")

    image_urls = {
        "Brain MRI (Tumor)": "https://upload.wikimedia.org/wikipedia/commons/7/75/MRI_T2_Meningioma.jpg",
        "Skin Rash": "https://upload.wikimedia.org/wikipedia/commons/8/81/Morbilliform_rash.JPG",
        "Allergic Reaction": "https://upload.wikimedia.org/wikipedia/commons/2/2e/Allergic_contact_dermatitis_due_to_catechu_-_2009.jpg",
        "Flu Symptoms Chart": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Cold_vs_flu_diagram.svg/2560px-Cold_vs_flu_diagram.svg.png"
    }

    with st.expander("🔍 Click to view sample images"):
        cols = st.columns(4)
        for idx, (desc, url) in enumerate(image_urls.items()):
            with cols[idx % 4]:
                img = load_image(url)
                if img:
                    st.image(img, caption=desc, use_container_width=True)

# Run the app
if __name__ == "__main__":
    try:
        main()
        additional_features()
        sample_images_section()

        st.divider()
        st.warning("""
        **Important Disclaimer:** 
        MediBot is an AI assistant for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        """)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
