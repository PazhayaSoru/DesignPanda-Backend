from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain.pydantic_v1 import Field


class AgentState(TypedDict):
  requirements : str
  messages : Annotated[list[Any],add_messages]
  next : Literal["requirements_node","architect_node","critic_node","optimizer_node","arbiter_node","__end__"] 
  architecture : str
  criticism : str
  optimizer_visits : int
  optimization_details : str

class CriticGrade(TypedDict):
  critic_score : Literal["yes","no"] = Field(description="Used to criticise a software architecture."
  " 'yes' if it needs optimization/improvement, 'no' if the current architecture is optimum ")
  criticism : str = Field(description="Describe the parts where the software architecture needs improvement")



class SadAgent:
  def __init__(self,):
    pass
