import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from IPython.display import Markdown as md

output_parser = StrOutputParser()
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

# Set up Gemini API key
f = open(r"D:\ML\Innomatics_Research_Lab_Internship\task9_Openai_api_chat\keys\.gemini_api_key.txt")
api_key = f.read()
def set_custom_style():
    st.markdown(
        """
        <style>
        .stTextInput>div>div>div>textarea {
            height: 200px; /* Adjust the height as needed */
        }
        
        .st-bj { background-color: #FFFFFF; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Initialize your components (this should be adapted based on your actual setup)
def load_components():
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, 
                                               model="models/embedding-001")
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
    retriever = db_connection.as_retriever(search_kwargs={"k": 5})
    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-pro-latest")
    return retriever,chat_model

retriever, chat_model = load_components()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

def handle_query(query):
    # Assuming 'retriever' is properly configured to use GoogleGenerativeAIEmbeddings.
    retrieved_docs = rag_chain.invoke(query)
    if not retrieved_docs:
        return "No documents found."
    try:
            response = retrieved_docs  # Ensure this method exists or is properly implemented.
            return response
    except AttributeError:
        # Fallback or error handling if the method isn't implemented.
        return "An error occurred while generating the answer."

# Streamlit UI setup remains the same
# Apply custom style
set_custom_style()
# Take user's input

st.markdown("<h1 style='color:  #4d0404;'>👩‍💻RAG-based Question Answering System</h1>", unsafe_allow_html=True)

user_query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if user_query:
        answer = handle_query(user_query)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")


