from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

app = FastAPI()

llm = OpenAI(temperature=0)

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text concisely:\n{text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(request.text)
    return {"summary": summary}
