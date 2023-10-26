# Streamlite para front
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu
from streamlit_login_auth_ui.widgets import __login__


# Lectura de Pdf
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Open Ai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import openai
from langchain.llms import OpenAI

from langchain.vectorstores import FAISS
import pickle
import os
from langchain.chains.question_answering import load_qa_chain

#Menu de lado y horizontal
with st.sidebar:
    selected = option_menu(None, ["Chat Box", "Internal Chat", "Mail + Invoice Automation", 'Settings'], 
    icons=['house', 'chat', "envelope", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected
selected2 = option_menu(None, ["Home", "Internal Chat", "Mail + Invoice Automation", 'Settings'], 
    icons=['house', 'chat', "envelope", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2
# Autenticación
__login__obj = __login__(auth_token = "courier_auth_token", 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()
# Si esta loggeado entonces se inicia la funciionalidad de chatbox 
if LOGGED_IN == True:


    def main():
        load_dotenv()
        st.header("Chat with Internal Docuemnts")
        st.write("Training Personal")
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
            query = st.text_input("Ask me something about the doc")
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