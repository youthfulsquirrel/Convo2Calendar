import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
from tika import parser
import langchain
langchain.verbose = False


# don't need to use, replaced with get_text_from_docs function, which can read non-PDF docs also
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def generate_output_filename():
    # Find the next available file number
    file_number = 1
    while True:
        filename = f"../main3/text{file_number:03d}.txt"
        if not os.path.exists(filename):
            return filename
        file_number += 1

def get_text_from_docs(pdf_docs):
    # Generate the output text file name
    output_text_file = generate_output_filename()

    content_string = ""  # Initialize an empty string to accumulate content
    meta_data_fields = {'dc:creator': 'file owner',
                        'dcterms:created': 'date created',
                        'dcterms:modified': 'last modified date'}
    for pdf in pdf_docs:
        results = parser.from_file(pdf)
        # Extract text content and append to content_string
        content_list = results['content'].strip().split('/')
        content = ''.join(content_list[:-2]).strip()
        content_string += f"file title: {results['metadata']['resourceName'][0][2:-1]}\n"
        for field in meta_data_fields:
            if field in results['metadata']:
                content_string += f"{meta_data_fields[field]}: {results['metadata'][field]}\n"
        content_string += f"{content}\n\nend of text for '{results['metadata']['resourceName'][0][2:-1]}'\n\n"

    # Write content_string to the output text file
    with open(output_text_file, 'w') as file:
        file.write(content_string)

    return content_string

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'][::-1]
    print ('______________________________________________________________')
    print (response['chat_history'])
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++===')

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    user_question = None
    with open(r'../keys/archive_note.txt', 'r') as fp:
        # read all lines using readline()
        lines = fp.readlines()
        for line in lines:
            os.environ['OPENAI_API_KEY'] = line
    st.set_page_config(page_title="Chat with your files",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple files (e.g .pdf, .pptx) :books:")
    with st.form("myform", clear_on_submit=True):
        user_question = st.text_input("Ask your question here:", key = 'user_input')
        submit = st.form_submit_button(label="Submit")
    if submit:
        handle_userinput(user_question)
 

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            " ", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_text_from_docs(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.text(get_conversation_chain(
                    vectorstore))


if __name__ == '__main__':
    main()