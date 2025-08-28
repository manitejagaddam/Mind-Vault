from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = FastAPI()
llm = OpenAI(temperature=0.3)

class LearningPathRequest(BaseModel):
    skill: str
    level: str  # beginner/intermediate/advanced

@app.post("/learning-path")
def generate_path(request: LearningPathRequest):
    prompt = PromptTemplate(
        input_variables=["skill", "level"],
        template="Create a structured {level} learning path for {skill} with resources and steps."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    path = chain.run({"skill": request.skill, "level": request.level})
    return {"learning_path": path}
