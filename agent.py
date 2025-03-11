import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from PIL import Image
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler


st.set_page_config(
    page_title="Travel Advice for Alpes-Maritimes",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header('Welcome to VisitAlpesMaritimes 🏞️!!')

## CONSTANTS
MODEL_ID= "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id= MODEL_ID,
    task="text-generation",
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_p=0.95
)

### LLM
raw_documents = PyPDFLoader('ttourisme.pdf').load()
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
documents = text_splitter.split_documents(raw_documents)

db = FAISS.from_documents(documents, HuggingFaceEmbeddings())

tool = create_retriever_tool(
    db.as_retriever(),
    "alpes_maritimes_travel",
    "Searches and returns documents regarding Alpes-Maritimes."
)
tools = [tool]
memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","AI Assistant", "General Information"])

# Default page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if page == "Home":
    st.title("Welcome to Travel Advice for Alpes-Maritimes")
    st.write("Use the buttons on the left to navigate to different pages.")
    st.image('im_alp.jpg', caption='Stunning views of Alpes-Maritimes')

if page == "AI Assistant":
    st.title("AI assistant")
    st.write("This is the page for the LLM bot, your travel assistant in Alpes-Maritimes. What are you planning for your next trip?")
    
    ## LLM
    user_query = st.text_input(
        "**What do you want to know about Alpes-Maritimes?**",
        placeholder="Ask me anything!"
    )
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "memory" not in st.session_state:
        st.session_state['memory'] = memory
    
    # Add your LLM bot code here
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response['output'])

elif page == "General Information":
    st.title("General Information")
    st.write("This page contains General travel information about Alpes-Maritimes.")

    destinations = {
        "Nice": "A beautiful city known for its Promenade des Anglais and Mediterranean beaches.",
        "Cannes": "Famous for its annual film festival and glamorous waterfront.",
        "Antibes": "A charming town with beautiful beaches and a well-preserved medieval old town.",
        "Grasse": "Known for its perfume industry and stunning hilltop views."
    }
    search = st.text_input("Search for a destination:")
    if search:
        st.write(f"Results for {search}:")
        if search in destinations:
            st.write(destinations[search])
        else:
            st.write("Destination not found.")
    
    tips = [
        "Best time to visit: May to October.",
        "Languages spoken: French, with a variety of local dialects.",
        "Currency: Euro (€).",
    ]
    for tip in tips:
        st.write(f"- {tip}")
