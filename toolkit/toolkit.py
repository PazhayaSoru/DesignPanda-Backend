from langchain.tools import tool
import re

@tool
def architect_retrieval(question : str) -> str:
  """
  Retrieve Software Architecture related information, which is primarily used for generating initial architecture design
  """

@tool
def critic_retrieval(question: str) -> str:
  pass

@tool
def optimizer_retrieval(question : str) -> str:
  pass

