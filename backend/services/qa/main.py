from fastapi import FastAPI, UploadFile
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

app = FastAPI()
llm = OpenAI(temperature=0)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    loader = PyPDFLoader(file.file)
    docs = loader.load()
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )
    return {"message": "PDF processed, ready for questions."}

@app.post("/ask")
async def ask_question(question: str):
    # retrieve QA chain from previous step (simple demo)
    answer = qa_chain.run(question)
    return {"answer": answer}
