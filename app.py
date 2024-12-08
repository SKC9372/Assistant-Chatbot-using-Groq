from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 
import streamlit as st


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    st.title("Groq Chat Application")

    st.sidebar.title("Select a Model...")
    model = st.sidebar.selectbox('Choose a Model',
                                 ["mixtral-8x7b-32768","llama2-70b-4096"])
    
    conversational_memory_length = st.sidebar.slider('Choose memory length:',1,10,5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask your question")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']},
                                {'output': message['AI']})
            
    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=model
    )

    conversation = ConversationChain(llm = groq_chat,
                                     memory=memory)
    
    if user_question:
        response = conversation(user_question)
        message = {'human':user_question,
                   'AI':response['response']
                   }
        st.session_state.chat_history.append(message)
        st.write(response['response'])

if __name__ == '__main__':
    main()
        
