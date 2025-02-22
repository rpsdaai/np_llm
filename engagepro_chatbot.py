import os
from config import hf_embeddings
from config import llm_local as llm
from pathlib import Path

from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.chains import MultiPromptChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

# vector store folder
FAISS_INDEX = "faiss_index"
# source folder containing 'Compnay_Brochure.pdf'
SRC_FOLDER = 'd:/Users/ng_a/My NP SDGAI/PDC-2/LLM/Assignment'

load_dotenv()

memory = ConversationBufferWindowMemory(
    k=3,  # Store the last 3 interactions
    memory_key="history",  # Key to access memory in prompts
    return_messages=True  # Return messages as a list (optional)
)

# Process the pdf file 
def doLoadProcessPdf(source_folder, pdf_file):
    data_folder = Path(source_folder)
    filename = data_folder / pdf_file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    loader = PyPDFLoader(filename)
    document = loader.load_and_split(text_splitter)

    return document

# creates a vector store if it does not exist
def doLoadVectorStore(document):
    if os.path.exists(FAISS_INDEX):
        print('Reading from saved vectorstore...')
        vectorstore = FAISS.load_local("faiss_index", hf_embeddings, allow_dangerous_deserialization=True)     
    else:
        print('creating vectorstore ....')
        # Create a FAISS vector store
        vectorstore = FAISS.from_documents(document, hf_embeddings)

        # save vectorstore to disk
        vectorstore.save_local(FAISS_INDEX)
    return vectorstore

# https://python.langchain.com/docs/how_to/vectorstore_retriever/
# k specifies the top k documents
def doGetVectorStoreRetriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k": k})

# Define the vector store retrieval Function
# def retrieval_qa(query):
#     global retriever
#     return retriever.invoke(query)

# def doQueryVectorStore(retriever, query):
#     return retriever.invoke(query)

# # Wikipedia Tool
# def wikipedia_search(wiki, query):
#     # global wiki
    
#     # string is returned
#     return wiki.run(query)

# # Initialize Wikipedia search tool
# def doInitWikiSearch():
#     wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
#     return (wiki)

def doWikiSearch(query):
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)

def create_router_chain(retriever, wikipedia_search):
    """
    Creates a Router Chain using LangChain 0.3.X syntax with enhanced reliability and ethical safeguards.
    """

    global memory

    # memory = ConversationBufferWindowMemory(
    #     k=3,  # Store the last 3 interactions
    #     memory_key="history",  # Key to access memory in prompts
    #     return_messages=True  # Return messages as a list (optional)
    # )
    
    router_prompt = PromptTemplate(
        input_variables=["history", "query"],
        template="""[SYSTEM] You are a classification expert. Categorize queries as either:
        - "Company Info": Specific to company products/services/policies
        - "General Knowledge": Broad technical/historical/scientific topics

        Safety Rules:
        1. When uncertain, classify as "General Knowledge"
        2. Flag and reject unethical requests immediately
        3. Prioritize factual accuracy over speculation

        Conversation History:
        {history}

        Query: {query}
        Classification:"""
    )

    response_template = """[SYSTEM] You are a professional AI assistant. Follow these rules:
    1. Base responses ONLY on provided context
    2. If context is insufficient, say "I don't have sufficient information"
    3. Maintain neutral, unbiased tone
    4. Reject unethical/ambiguous requests politely

    Conversation History:
    {history}

    Context: {context}

    [USER] {query}
    [ASSISTANT]"""

    company_prompt = PromptTemplate(
        input_variables=["history", "context", "query"],
        template="COMPANY SPECIALIST MODE\n" + response_template
    )

    general_prompt = PromptTemplate(
        input_variables=["history", "context", "query"],
        template="GENERAL KNOWLEDGE MODE\n" + response_template
    )

    # Define Routing Logic
    router_chain = (
        RunnablePassthrough.assign(
            history=lambda x: memory.load_memory_variables({})["history"]
        )
        | router_prompt
        | llm
        | StrOutputParser()
    )

    # Define Company Chain
    company_chain = (
        RunnablePassthrough.assign(
            history=lambda x: memory.load_memory_variables({})["history"],
            context=lambda x: retriever.invoke(x["query"])
        )
        | company_prompt
        | llm
    )

    # Define General Chain
    general_chain = (
        RunnablePassthrough.assign(
            history=lambda x: memory.load_memory_variables({})["history"],
            context=lambda x: wikipedia_search(x["query"])
        )
        | general_prompt
        | llm
    )

    # Create Branching Logic
    return (
        RunnablePassthrough.assign(
            category=router_chain,
            query=lambda x: x["query"]
        )
        | RunnableBranch(
            (lambda x: "company info" in x["category"].lower(), company_chain),
            (lambda x: "general knowledge" in x["category"].lower(), general_chain),
            general_chain  # Default fallback
        )
    )

# Streamlit stuff
def set_up_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
        <style>
            .chat-input-container {
                position: fixed;
                bottom: 0;
                width: 100%;
                background-color: #f4f4f4;
                padding: 10px;
                box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
            }
            .chat-history {
                margin-bottom: 80px; /* Leave space for the fixed input area */
            }
            .user-message {
                background-color: #DCF8C6; /* Light green */
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                text-align: right;
            }
            .bot-message {
                background-color: #ECECEC; /* Light grey */
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                text-align: left;
            }
        </style>
    """, unsafe_allow_html=True)

def on_slider_change():
    print('--> on_slider_change: ' + str(st.session_state.temperature))
    if st.session_state.temperature != st.session_state.temperature_slider:
        st.session_state.temperature = st.session_state.temperature_slider   
        print('--> new value: ' + str(st.session_state.temperature)) 
    llm.temperature = st.session_state.temperature
    print ('temperature set to: ' + str(llm.temperature))

def set_up_sidebar():
    """Set up the sidebar navigation widgets."""
    with st.sidebar:   
        st.markdown("# Chat Options")
        # Store the Temperature Value in Session State
        # st.session_state.temperature = st.slider("Sensitivity", min_value=0.0, max_value=1.0, value=0.2, step=0.1, format="%f", key="temperature")
        # if "temperature" not in st.session_state:
        #     st.session_state.temperature = 0.5
        st.session_state.temperature = st.slider(
            "Sensitivity", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.temperature, 
            step=0.1, 
            format="%f",
            key='temperature_slider',
            on_change=on_slider_change
        )        

# session_state items must be declared before being referenced
def initialize_session_state():
    """Initialize the session state for storing chat messages."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # Initialize session state for slider value
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0 # Default

    # Initialize session state for slider (temporary value)
    if 'temperature_slider' not in st.session_state:
        st.session_state.slider = st.session_state.temperature  # Default value for the slider 

    if llm.temperature is None:
        print ('Temperature is NONE')       

def handle_input(router_chain):
    # # Clear input field (important!)
    # st.session_state.chat_input = ""

    """Handle user input and generate a bot response."""
    user_input = st.session_state.chat_input.strip()
    if user_input:
        # Add user message
        st.session_state.messages.append({"content": user_input, "is_user": True})

        # Bot response (simple placeholder)
        bot_response = generate_bot_response(user_input, router_chain)
        st.session_state.messages.append({"content": bot_response, "is_user": False})


        # Clear input field (important!)
        st.session_state.chat_input = ""
    # else:
    #     # Clear input field (important!)
    #     st.session_state.chat_input = ""        

def display_chat_history():
    """Display the chat history in a container."""
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["is_user"]:
            st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def display_input_area(router_chain):
    """Display the fixed input area at the bottom."""
    st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
    
    # Text input (triggers on Enter key)
    st.text_input("Your message:", key="chat_input", on_change=handle_input, args=(router_chain,))
    # st.chat_input("Your message:", key="chat_input", on_submit=handle_input, args=(router_chain,))

    st.markdown("</div>", unsafe_allow_html=True)

def generate_bot_response(user_input, router_chain):
    """Generate a response from the bot using the LLM."""
    # Retrieve the temperature value
    # temperature = st.session_state.temperature

    # Pass the temperature value to the LLM (placeholder logic)
    # For example, using OpenAI's GPT-3:
    # response = openai.Completion.create(
    #     engine="davinci",
    #     prompt=user_input,
    #     temperature=temperature,
    #     max_tokens=150
    # )
    response = router_chain.invoke({"query": user_input})
    # Placeholder response
    # response = "That's interesting! Tell me more." + str(temperature)

    # Update memory with the new interaction
    print (type(response))
    print (type(response.content))
    print ("RESPONSE CONTENT\n")
    print (response.content)
    memory.save_context({"query": user_input}, {"response": response.content})
    
    return response.content

def doInitChatBot():
    print ('--> doInitChatBot()')
    document = doLoadProcessPdf(SRC_FOLDER, 'Company_Brochure.pdf')
    vector_store = doLoadVectorStore(document)
    retriever = doGetVectorStoreRetriever(vector_store)
    router_chain = create_router_chain(retriever, doWikiSearch)
    print ('<-- doInitChatBot()')
    return router_chain

# https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide
def test_UI(my_router_chain):
    """Main function to run the Streamlit app."""
    print ('--> testUI')
    set_up_page()
    apply_custom_styling()
    st.title("🤖 EngagePro Chatbot")
    initialize_session_state()
    set_up_sidebar()
    st.subheader("Hi there! I'm your EngagePro assistant. How can I help you today?")
    
    display_chat_history()
    display_input_area(my_router_chain)
    print ('<-- testUI')

# Test Vector Store
def test_VectorStore():    
    # Test Vectorstore
    document = doLoadProcessPdf(SRC_FOLDER, 'Company_Brochure.pdf')
    vector_store = doLoadVectorStore(document)
    vector_store_retriever = doGetVectorStoreRetriever(vector_store)

    query = input("Enter your query: ")
    # the retreiver returns a list of of type Document
    # https://python.langchain.com/docs/integrations/retrievers/
    # https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html
    response = vector_store_retriever.invoke(query)
    # List returned
    print (type(response)) 
    # page content is the last item
    # print (response[-1]) 
    # print (len(response))
    for i in range(len(response)):
        print (i)
        print ('\n')
        print (response[i].page_content)
        print ('\n')

# Test Wiki
def test_WikiSearch():
    query = input("Enter your query: ")
    results = doWikiSearch(query)
    # wiki = doInitWikiSearch()
    # results = wikipedia_search(wiki, query)
    print (results)

# Test standalone Chatbot
def test_ChatBot():
    router_chain = doInitChatBot()
    while True:
        query = input("Ask me a question (or type 'exit' to quit): ")
        if query.lower() == "q":
            break
        response = router_chain.invoke({"query": query})
        # print("Bot:", response["output"])   
        memory.save_context({"query": query.lower()}, {"response": response.content})
        print (response.content) 

if __name__ == "__main__":
    # WORKS
    # test_WikiSearch()
    # WORKS
    # test_VectorStore()
    # WORKS
    # test_UI()

    # WORKS
    # router_chain = doInitChatBot()
    # while True:
    #     query = input("Ask me a question (or type 'exit' to quit): ")
    #     if query.lower() == "q":
    #         break
    #     response = router_chain.invoke({"query": query})
    #     # print("Bot:", response["output"])   
    #     print (response.content)

    # memory = ConversationBufferWindowMemory(
    #     k=3,  # Store the last 3 interactions
    #     memory_key="history",  # Key to access memory in prompts
    #     return_messages=True  # Return messages as a list (optional)
    # )

    # Standalone - no UI
    # test_ChatBot()

    # 
    # To run the program: streamlit run engagepro_chatbot.py
    # 
    chatbot_router_chain = doInitChatBot()  
    test_UI(chatbot_router_chain)