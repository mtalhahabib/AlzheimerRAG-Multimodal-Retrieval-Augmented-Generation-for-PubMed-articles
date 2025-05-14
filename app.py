import streamlit as st
from PIL import Image
import base64
from claudeRag import AlzheimerRAG
from claudeRag import PubMedFetcher

# Set up page configuration
st.set_page_config(
    page_title="NeuroCare - Alzheimer's Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_background(image_file):
    """Set background image with overlay"""
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(18, 18, 18, 0.95), rgba(18, 18, 18, 0.95)), 
                            url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-attachment: fixed;
        }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def create_web_interface(rag_system):
    # Updated color scheme
    PRIMARY_COLOR = "#6366F1"  # Indigo
    SECONDARY_COLOR = "#3B82F6"  # Blue
    USER_COLOR = "#d95e29"     # Purple
    BOT_COLOR = "#a1754c"      # Teal
    ACCENT_COLOR = "#F59E0B"   # Amber
    TEXT_COLOR = "#F8FAFC"     # Light Slate
    BACKGROUND_COLOR = "#0F172A"  # Dark Navy
    CARD_BACKGROUND = "#1E293B"   # Dark Slate

    st.markdown(f"""
    <style>
        :root {{
            --primary: {PRIMARY_COLOR};
            --secondary: {SECONDARY_COLOR};
            --user: {USER_COLOR};
            --bot: {BOT_COLOR};
            --accent: {ACCENT_COLOR};
            --text: {TEXT_COLOR};
            --background: {BACKGROUND_COLOR};
            --card-bg: {CARD_BACKGROUND};
        }}

        body {{
            font-family: 'Arial', sans-serif;
            background: var(--background);
            color: var(--text);
        }}

        /* Chat Messages */
        .user-message {{
            background: linear-gradient(135deg, {USER_COLOR} 0%, #7C3AED 100%);
            color: white;
            border-radius: 20px 4px 20px 20px;
            margin: 12px 5% 12px 25%;
            padding: 18px 25px;
            position: relative;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateX(20px);
            transition: transform 0.3s ease;
        }}

        .user-message::before {{
            content: "👤";
            position: absolute;
            right: -40px;
            top: 12px;
            font-size: 1.4rem;
        }}

        .bot-message {{
            background: linear-gradient(135deg, {BOT_COLOR} 0%, #059669 100%);
            color: white;
            border-radius: 4px 20px 20px 20px;
            margin: 12px 25% 12px 5%;
            padding: 18px 25px;
            position: relative;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateX(-20px);
            transition: transform 0.3s ease;
        }}

        .bot-message::before {{
            content: "🧠";
            position: absolute;
            left: -40px;
            top: 12px;
            font-size: 1.4rem;
        }}

        .user-message:hover, .bot-message:hover {{
            transform: translateX(0);
        }}

        /* Chat Container */
        .chat-container {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 30px;
            margin: 2rem auto;
            max-width: 800px;
            height: 65vh;
            overflow-y: auto;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* Input Container */
        .input-container {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 16px;
            margin: 20px auto;
            max-width: 800px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}

        .stChatInput input {{
            background: rgba(255, 255, 255, 0.05) !important;
            color: var(--text) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 12px 20px !important;
        }}

        .stSpinner > div {{
            border-color: var(--accent) transparent transparent transparent !important;
        }}

        /* Title Styles */
        .title-container {{
            text-align: center;
            margin-top: 15vh;
        }}
        .title {{
            font-size: 2.8rem;
            font-weight: bold;
            color: var(--accent);
        }}
        .subtitle {{
            font-size: 1.2rem;
            color: var(--text);
        }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("logoo.jpg", width=120)
        st.markdown("""
        ## NeuroCare Assistant
        **Version:** 1.2.0  
        **Last Updated:** May 2025  
        
        ### Key Features:
        - 🧬 Genomic Research Insights
        - 🧪 Clinical Trial Analysis
        - 📈 Treatment Efficacy Data
        - 🩺 Caregiver Support Tools
        
        ---
        
        🔧 **Settings**
        """)
        include_images = st.checkbox("Enable Visual Analytics", True)
        st.markdown("""
        ---
        ⚠️ **Disclaimer**
        *This system provides research insights only. Consult healthcare professionals for medical decisions.*
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    if not st.session_state.messages:
        st.markdown("""
        <div class='title-container'>
            <div class='title'>NeuroCare Assistant 🧠</div>
            <div class='subtitle'>Your Alzheimer's Research Companion</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">{message["content"]}</div>
                """, unsafe_allow_html=True)
            else:
                content = message["content"]
                if "figures" in message and message["figures"]:
                    for fig in message["figures"]:
                        content += f"\n![Figure]({fig})"
                st.markdown(f"""
                <div class="bot-message">{content}</div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Input box
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    prompt = st.chat_input("Ask your research question...")
    st.markdown("</div>", unsafe_allow_html=True)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Analyzing research insights..."):
            try:
                response, figures = rag_system.answer(prompt, include_images)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "figures": figures if include_images else []
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
        set_background("background.jpg")
        if 'rag_system' not in st.session_state:
            with st.spinner("Loading knowledge base..."):
                st.session_state.rag_system = AlzheimerRAG()
        create_web_interface(st.session_state.rag_system)
    except Exception as e:
        st.error(f"Initialization Failed: {str(e)}")