from langchain_groq import ChatGroq
import os


GROQ_API_KEY= os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = "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"

class LLMModel:
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.groq_model=ChatGroq(model=self.model_name)
        
    def get_model(self):
        return self.groq_model
  


