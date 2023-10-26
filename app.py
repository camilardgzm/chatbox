import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import openai
from langchain.llms import OpenAI

from langchain.vectorstores import FAISS
import pickle
import os
from langchain.chains.question_answering import load_qa_chain
# simulamos una conexión con dirve para interacción con documentos

with st.sidebar:
    st.title("Internal Document Chat")
    st.markdown('''
Qr gente como andan 
''')
add_vertical_space(5)


def main():
    load_dotenv()
    st.header("Chat with Internal Docuemnts from DB")
    #Subir Pdfs
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        st.write(pdf.name)
        pdfReader = PdfReader(pdf)
        #st.write(pdfReader)
        text = ""
        for page in pdfReader.pages:
            text += page.extract_text()
        
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200, #overlap between chunks  
            length_function  = len
            )
        chunks = textSplitter.split_text(text=text)
        




        #embedings
        storeName = pdf.name[:-4]

# Reads file from storage, es decir ya se habia cargado este pdf antes 
        if os.path.exists(f"{storeName}.pkl"):
            with open(f"{storeName}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings Loaded from disk ")
        else:# si nunca se habi adetectado ese pdf, cargamos los embeddings
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{storeName}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
            st.write("Embeding computation completed")
# User input prompts
        query = st.text_input("Chings tu madre, preguntame algo")
        st.write(query)
# Perform semantic search, top 3 chunks more similar 
        if query:
            docs = VectorStore.similarity_search(query=query, k = 3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type="stuff")
            response = chain.run(input_documents = docs, question=query)
            st.write(response)
       





if __name__ == "__main__":
    main()