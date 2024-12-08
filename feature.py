from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 
import streamlit as st
from datetime import datetime


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Page Configuration
st.set_page_config(
    page_title='Groq Chat Assitant',
    page_icon="🤖",
    layout='wide',
    initial_sidebar_state='expanded'
)

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

def get_custom_prompt():
    persona = st.session_state.get('selected_persona','Default')

    personas = {
        'Default' :"""You are a helpful AI assistant.
                       Current conversation:
                       {history}
                       Human:{input}
                       AI:""",

        'Expert' :"""You are an expert consultant having deep knowlegde across various fields.
                      Please provide deatail and accurate responses
                      Current conversation:
                      {history}
                      Human:{input}
                      AI:""",

        'Creative' :"""You are a creative and imaginative AI assistant that thinks outside the box.
                        Feel free to use metaphors and analogies in your responses
                        Current conversation:
                        {history}
                        Human:{input}
                        AI:"""
    }
    
    return PromptTemplate(
        input_variables=['history','input'],
        template = personas[persona]
    )


def main():
    # Initalizing the session
    initialize_session_state()

    # Sidebar Configuration
    with st.sidebar:
        st.title("🛠️ Chat Settings")

        st.subheader("Select a Model")
        model = st.selectbox('Choose a Model',
                                 ["mixtral-8x7b-32768","llama2-70b-4096"],
                                 help="Select a AI model for Conversation")
        
        #Memory Configuration
        st.subheader("Memory Settings")
        memory_length = st.slider('Conversation Memory',1,10,5,
                                  help = "Number of Previous Converstions to save")
        
        #Persona Selection
        st.subheader("Assistant type")
        st.session_state.selected_persona = st.selectbox('Select conversation style:',
            ['Default', 'Expert', 'Creative'])
        
        #Chat Statistics
        if st.button("🗑️ Clear Chat History",use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.start_time = None
            st.rerun()

    #Main Chat Interface

    st.title("🤖 Groq AI Asisstant")

    #Initialize Chat COmponents
    memory = ConversationBufferWindowMemory(k=memory_length)

    groq_chat = ChatGroq(api_key=GROQ_API_KEY,
                         model=model)
    
    conversation = ConversationChain(llm=groq_chat,
                                     memory=memory,
                                     prompt = get_custom_prompt())
    
    #load chat history in memory 
    for message in st.session_state.chat_history:
        memory.save_context(
            {'input':message['human']},
            {'output':message['AI']}
        )
    #Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            st.write(f'👤You')
            st.info(message['human'])

            #AI Response
        with st.container():
            st.write(f"🤖 Assistant ({st.session_state.selected_persona} mode)")
            st.success(message['AI'])

        # Add some spaces
        st.write()

    # User input section
    st.markdown("### 💭 Your Message")
    user_question = st.text_area(
        "",
        height=100,
        placeholder="Type your message here... (Shift + Enter to send)",
        key="user_input",
        help="Type your message and press Shift + Enter or click the Send button"
    )

    # Input buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    with col3:
        if st.button("🔄 New Topic", use_container_width=True):
            memory.clear()
            st.success("Memory cleared for new topic!")

    if send_button and user_question:
        if not st.session_state.start_time:
            st.session_state.start_time = datetime.now()

        with st.spinner('🤔 Thinking...'):
            try:
                response = conversation(user_question)
                message = {
                    'human': user_question,
                    'AI': response['response']
                }
                st.session_state.chat_history.append(message)
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "Using Groq AI with "
        f"{st.session_state.selected_persona.lower()} persona | "
        f"Memory: {memory_length} messages"
    )

if __name__ == "__main__":
    main()

        




