import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
from datetime import datetime
import speech_recognition as sr

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def initialize_page():
    st.set_page_config(
        page_title="Disaster Response System",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
        /* Core theme */
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --accent: #238636;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --border: #30363d;
        }

        /* Global resets */
        .stApp {
            background-color: var(--bg-primary);
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        /* Column layout */
        .main-container {
            display: flex;
            gap: 1rem;
            padding: 1rem;
            height: calc(100vh - 2rem);
        }

        .column {
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 1rem;
            height: 100%;
            overflow-y: auto;
        }

        .column-left {
            flex: 1;
        }

        .column-middle {
            flex: 2;
        }

        .column-right {
            flex: 1;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 3px;
        }

        /* Components */
        .header {
            background: linear-gradient(to bottom, var(--bg-tertiary), var(--bg-secondary));
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid var(--accent);
            margin-bottom: 1rem;
        }

        .status-bar {
            display: flex;
            gap: 0.5rem;
            background: var(--bg-tertiary);
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }

        .status-indicator {
            color: var(--accent);
            font-weight: bold;
        }

        /* Chat components */
        .message-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin: 1rem 0;
        }

        .message {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }

        .message-query {
            color: var(--accent);
            margin-bottom: 0.5rem;
        }

        .message-response {
            color: var(--text-primary);
        }

        .message-meta {
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-align: right;
            margin-top: 0.5rem;
        }

        /* Input area */
        .input-container {
            display: flex;
            gap: 0.5rem;
            margin: 1rem 0;
        }

        .stTextInput > div > div > input {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
        }

        .stButton > button {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
        }

        .stButton > button:hover {
            border-color: var(--accent) !important;
            transform: translateY(-1px);
        }

        /* Quick access */
        .quick-access {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .quick-access button {
            text-align: left !important;
            padding: 0.75rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-specdec"
    )

def convert_speech_to_text():
    recognizer = sr.Recognizer()
    
    try:
        import pyaudio  # Ensure PyAudio is installed
        with sr.Microphone() as source:
            st.info("Adjusting for ambient noise. Please wait...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            st.info("Please speak something...")

            try:
                audio = recognizer.listen(source, timeout=5)
                st.info("Processing speech...")

                text = recognizer.recognize_google(audio)
                if text:
                    st.session_state.current_query = text
                    st.session_state.should_send = True
                return text

            except sr.WaitTimeoutError:
                return "No speech detected"
            except sr.RequestError:
                return "Could not connect to speech recognition service"
            except sr.UnknownValueError:
                return "Could not understand the audio"
    
    except OSError:
        st.error("Microphone not available. Please check your device settings.")
        return "Microphone not available"

def initialize_prompt():
    return ChatPromptTemplate.from_template("""
    You are a **Disaster Response System**, responsible for providing **accurate and actionable** guidance based on **official disaster response protocols**.  

    **RESPONSE GUIDELINES:**  
    1. **Use ONLY official disaster response documents** to generate responses. No external assumptions or opinions.  
    2. **Provide clear, actionable steps** from the protocols. Never tell users to "check the procedures themselves."  
    3. **For unrelated queries**, respond with:  
       ```
       "This information is not available in the disaster response database"
       ```  
    4. **If the query is relevant but not found in protocols**, respond with:  
       ```  
       "This specific scenario isn't covered in our protocols. Follow general disaster response guidelines:  
       - Ensure personal safety first  
       - Call emergency services if needed  
       - Follow evacuation routes if available  
       - Help others if it's safe to do so  
       - Listen to official announcements"  
       ```  
    5. **Prioritize clear, step-by-step instructions** for emergency situations.  

    **Response Format:**  
    - **Immediate Actions → Critical first steps**  
    - **Detailed Procedure → Step-by-step guidance**  
    - **Source Reference → Official protocol source**  

    **Disaster Response Data:**  
    {context}  

    **Query:** {input}  
    """)

@st.cache_resource
def process_documents_background():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./Data")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        return FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"Protocol database initialization error: {str(e)}")
        return None

def display_system_status():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**System Status:** <span class='status-online'>● ONLINE</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")
    with col3:
        st.markdown("**Protocol Database:** <span class='status-online'>● ACTIVE</span>", unsafe_allow_html=True)

def main():
    initialize_page()
    
    # Initialize vectors outside the query processing
    vectors = process_documents_background()
    if not vectors:
        st.error("Failed to initialize protocol database")
        return
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Three-column layout
    left_col, middle_col, right_col = st.columns([1, 2, 1])
    
    # Left Column - Quick Access
    with left_col:
        st.markdown("### Quick Access Protocols")
        protocols = {
            "🌪️ Tornado": "What to do during a tornado warning?",
            "🔥 Wildfire": "How to prepare for and respond to wildfires?",
            "🏠 Earthquake": "What are the safety steps during an earthquake?",
            "💨 Hurricane": "How to prepare for a hurricane?",
            "🚰 Flood": "What to do when flood waters are rising?",
            "⚡ Power Outage": "How to stay safe during extended power outages?",
            "🏥 First Aid": "Basic first aid procedures for common injuries"
        }
        
        for label, query in protocols.items():
            if st.button(label, key=f"quick_{label}"):
                st.session_state.current_query = f"{query}?"
                st.session_state.should_send = True
    
    # Middle Column - Chat Interface
    with middle_col:
        st.title("Disaster Response Assistant")
        st.markdown("##### Emergency Management System")
        
        # Status Bar
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown("**System:** 🟢 Online")
        with status_col2:
            st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d')}")
        with status_col3:
            st.markdown("**Database:** 🟢 Active")
        
        # Input Area with Enter key handling
        input_col1, input_col2, input_col3 = st.columns([3, 1, 1])
        with input_col1:
            question = st.text_input(
                "Query",
                value=st.session_state.get('current_query', ''),
                placeholder="Enter your query...",
                label_visibility="collapsed",
                key="query_input",
                on_change=lambda: setattr(st.session_state, 'should_send', True) if st.session_state.query_input else None
            )
            # Update current_query when input changes
            if question:
                st.session_state.current_query = question
        
        with input_col2:
            speak_btn = st.button("🎙️ Speak")
        with input_col3:
            send_btn = st.button("Send ➤")
            if send_btn and st.session_state.current_query:
                st.session_state.should_send = True
        
        # Handle voice input and auto-send
        if speak_btn:
            query = convert_speech_to_text()
            if query not in ["No speech detected", "Could not connect", "Could not understand"]:
                st.session_state.current_query = query
                st.session_state.should_send = True
                st.rerun()
        
        # Process query when triggered by Enter, Send button, or voice
        if st.session_state.get('should_send', False) and st.session_state.get('current_query'):
            with st.spinner("Analysing Protocols..."):
                try:
                    document_chain = create_stuff_documents_chain(llm, initialize_prompt())
                    retrieval_chain = create_retrieval_chain(vectors.as_retriever(), document_chain)
                    
                    start_time = time.time()
                    response = retrieval_chain.invoke({'input': st.session_state.current_query})
                    processing_time = time.time() - start_time
                    
                    # Store in chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        'question': st.session_state.current_query,
                        'answer': response['answer'],
                        'time': processing_time,
                        'context': response['context']
                    })
                    
                    # Reset the send flag
                    st.session_state.should_send = False
                    st.session_state.current_query = ''
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        # Display chat history (most recent first)
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            chat_container = st.container()
            with chat_container:
                for chat in reversed(st.session_state.chat_history):
                    st.markdown("""
                        <div style="background-color: var(--bg-tertiary); padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="color: var(--accent); margin-bottom: 0.5rem;">Query: {}</div>
                            <div style="color: var(--text-primary);"><strong>Response:</strong> <br>{}</div>
                            <div style="color: var(--text-secondary); font-size: 0.8rem; text-align: right; margin-top: 0.5rem;">
                                Response time: {:.2f}s
                            </div>
                        </div>
                    """.format(
                        chat['question'],
                        chat['answer'].replace('\n', '<br>'),
                        chat['time']
                    ), unsafe_allow_html=True)
    
    # Right Column - Reference Materials
    with right_col:
        st.markdown("### Reference Materials")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            for idx, doc in enumerate(latest['context'], 1):
                with st.expander(f"Reference {idx}"):
                    st.markdown(doc.page_content)

if __name__ == "__main__":
    main()