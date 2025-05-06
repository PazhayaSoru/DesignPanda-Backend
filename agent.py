#Third Party Imports
from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import Field
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from dotenv import load_dotenv
#Custom library imports
from utils.llms import LLMModel

load_dotenv()

#Setting environment variables
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")
print(GROQ_API_KEY)
os.environ["GROQ_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] =  LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

architect_vectorstore = FAISS.load_local("./vectorstores/architect_faiss", embedding, allow_dangerous_deserialization=True)
architect_faiss = architect_vectorstore.as_retriever(search_kwargs={"k": 3})


with open("./vectorstores/architect_chunks/architect_chunks.pkl", "rb") as f:
    architect_chunks = pickle.load(f)
architect_bm25 = BM25Retriever.from_documents(architect_chunks)
architect_bm25.k = 3


architect_retriever = EnsembleRetriever(retrievers=[architect_bm25, architect_faiss],weights=[0.3, 0.7])

critic_vectorstore = FAISS.load_local("./vectorstores/critic_faiss", embedding, allow_dangerous_deserialization=True)
critic_faiss = critic_vectorstore.as_retriever(search_kwargs={"k": 3})


with open("./vectorstores/critic_chunks/critic_chunks.pkl", "rb") as f:
    critic_chunks = pickle.load(f)
critic_bm25 = BM25Retriever.from_documents(critic_chunks)
critic_bm25.k = 3


critic_retriever = EnsembleRetriever(retrievers=[critic_bm25, critic_faiss],weights=[0.3, 0.7])

#State of the graph
class AgentState(TypedDict):
  requirements : str
  messages : Annotated[list[Any],add_messages]
  next : Literal["requirements_node","architect_node","critic_node","optimizer_node","arbiter_node","__end__"] 
  architecture : str
  criticism : str
  optimizer_visits : int
  optimization_details : str
  uml_code : str


class CriticGrade(TypedDict):
  critic_score : Literal["yes","no"] = Field(description="Used to criticise a software architecture."
  " 'yes' if it needs optimization/improvement, 'no' if the current architecture is optimum ")
  criticism : str = Field(description="Describe the parts where the software architecture needs improvement")





class SadAgent:
  def __init__(self):
    architect_model = LLMModel("deepseek-r1-distill-llama-70b")
    optimizer_critic_model = LLMModel('meta-llama/llama-4-maverick-17b-128e-instruct')
    arbiter_requirements_model = LLMModel("meta-llama/llama-4-scout-17b-16e-instruct")

    self.architect_model = architect_model.get_model()
    self.optimizer_model = optimizer_critic_model.get_model()
    self.critic_model = optimizer_critic_model.get_model()
    self.arbiter_model = arbiter_requirements_model.get_model()
    self.requirements_model =arbiter_requirements_model.get_model()

    


  def supervisor_node(self,state : AgentState) -> Command[Literal["requirements_node","architect_node","critic_node","optimizer_node","arbiter_node","__end__"]]:
    print("****************Entered supervisor****************")
    
    
    if state['optimizer_visits'] == 3:
        
        state["messages"].append({
            "role": "system",
            "content": "Exceeded maximum optimizer visits. Proceeding to Arbiter."
        })
        optimization_details = "Skipped Optimatization due to excessive optimization cycles."
        goto = "arbiter_node"

        return Command(update={"optimization_details":optimization_details,"next":goto},goto=goto)
    
    else:
      if len(state['messages']) == 1:
        query = state['messages'][0].content
        print(query)
        state['messages'].append({
          "role":"system",
          "content":"Transitioning to requirements engineer to generate the system requirements based on the User's input"
          })
        
        goto = 'requirements_node'
        return Command(update={"next":goto},goto=goto)
      else:
         goto=state['next']
         state['messages'].append({
            "role":"system",
            "content":f"Transitioning to {state["next"]} to proceed further"
         })
         return Command(goto=goto)

  def requirements_node(self,state : AgentState) -> Command[Literal["supervisor"]]:
     print("****************Entered requirements node****************")
     #requirements_agent = create_react_agent(self.llm_model,tools=[], prompt="You are a requirements engineer. You are supposed to break down the user's software idea and identify the system requirements. Give accurate, appropriate system requirements")
     
     requirements_prompt = "You are a requirements engineer. You are supposed to break down the user's software idea and identify the system requirements. Give accurate, appropriate system requirements"
     #result = requirements_agent.invoke({"input":state['messages'][0].content})

     result = self.requirements_model.invoke(f"{requirements_prompt}\n\n{state['messages'][0].content}")

     return Command(
        update={"messages":state['messages'] + [HumanMessage(content="[REQUIREMENTS ENGINEER] System Requirements has been generated by requirements engineer, Transitioning to Architect",name='requirements_engineer')]
                , "requirements":result.content,"next":"architect_node"},goto="supervisor"
     )
  

  def architect_node(self, state : AgentState) -> Command[Literal["supervisor"]]:
     print("****************Entered architect node****************")
     #architect_agent = create_react_agent(self.llm_model,tools=[],prompt="Based on the system requirements, generate a complete,neat, scalable software architecture design along with appropriate frameworks, libraries, deployment options")

     architect_prompt = "Based on the system requirements, generate a complete,neat, scalable software architecture design along with appropriate frameworks, libraries, deployment options"
     architect_rag = [i.page_content for i in architect_retriever.invoke(state['requirements'])]
     #result = architect_agent.invoke({"input":f"System Requirements : {state['requirements']}"})
     result = self.architect_model.invoke(f"{architect_prompt}\n\nRequirements : {state['requirements']} \n\nRelevant data extracted from books and articles are given below. Use the below data if it is relevant or else ignore it \n\n DATA : {architect_rag}")

     #result = architect_agent.invoke({"input":f"System Requirements : {state['requirements']}"})
    #  print(result.content.split('</think>')[1])
     return Command(
        update={"messages": state['messages'] + [HumanMessage(content="[ARCHITECT] Initial Design has been generated by architect, Transitioning to Critic for further evaluation",name="architect")],
                "architecture":result.content.split('</think>')[1],"next":"critic_node"},goto="supervisor"
     )


  def critic_node(self,state : AgentState) -> Command[Literal["supervisor","arbiter_node"]]:
     print("****************Entered critic node****************")
     #critic_agent = create_react_agent(self.llm_model.with_structured_output(CriticGrade),tools=[],prompt="You are a software architecture critic in a multi-agent system.Your job is to carefully examine a proposed system architecture and identify any flaws, anti-patterns, or weaknesses. These may include violations of software " \
     #"design principles, overengineering, security concerns, lack of scalability, or poor modularity.If you find the architecture to be optimal and meets the system requirements, then respond")

     critic_prompt = ("You are a software architecture critic in a multi-agent system.Your job is to carefully examine a proposed system architecture and identify any flaws, anti-patterns, or weaknesses. These may include violations of software " \
     "design principles, overengineering, security concerns, lack of scalability, or poor modularity.If you find that the architecture needs improvement/modification return critic_score as 'yes' and describe the areas where it needs improvement/modification. If the architecture is already optimal and doesn't need any improvement/modification return critic_score as 'no'")
     critic_rag = [i.page_content for i in critic_retriever.invoke(state['architecture'])]
     result= self.critic_model.with_structured_output(CriticGrade).invoke(f"{critic_prompt} \n\n software architecture: {state['architecture']} \n\nRelevant data extracted from books and articles are given below. Use the below data if it is relevant or else ignore it \n\n DATA : {critic_rag}")
     #result = critic_agent.invoke({"input":f"Software architecture: {state['architecture']}"})

     if result['critic_score'] == 'yes':
        goto = "optimizer_node"
        visits = state['optimizer_visits']+1
        #print(visits)
        msg = [HumanMessage(content="[CRITIC] Criticism has been generated. Transitioning to Optimizer",name="critic")]

        return Command(
        update={"messages":state['messages'] + msg,"next":goto,'criticism':result['criticism'],'optimizer_visits':visits},goto="supervisor"
     )

     else:
        goto = 'arbiter_node'
        msg = [HumanMessage(content="[CRITIC] Criticism has been generated. Transitioning to Arbiter",name="critic")]
        return Command(
        update={"messages":state['messages'] + msg,"next":goto,'criticism':result['criticism']},goto="arbiter_node"
     )
  
     
  
  def optimizer_node(self,state : AgentState) -> Command[Literal["supervisor","arbiter_node"]]:
     print("****************Entered optimizer node****************")
     print(f"Optimizer Visits: {state['optimizer_visits']}")
     if state['optimizer_visits'] == 3:
        return Command(goto="arbiter_node")
     #optimizer_agent = create_react_agent(self.llm_model,tools=[],prompt="You are a software architecture optimizer based on the criticism mentioned. Optimize the given architecture keeping the requirements in mind as well.Give only the final software architecture design after optimization")
     
     optimizer_prompt = "You are a software architecture optimizer based on the criticism mentioned. Optimize the given architecture keeping the requirements in mind as well.Give only the final software architecture design after optimization"
     
     #result = optimizer_agent.invoke({"input":f"Software Architecture: {state['architecture']}\n\n Criticism: {state['criticism']} \n\n Requirements: {state['requirements']}"})

     result = self.optimizer_model.invoke(f"{optimizer_prompt}\n\n Software Architecture: {state['architecture']}\n\n Requirements: {state['requirements']}\n\n Criticism: {state['criticism']}")
     print(result)
     return Command(
        update={"messages":state['messages']+[HumanMessage(content="[OPTIMIZER] The architecture has been optimized by the optimzer. Transitioning to the Critic",name="optimizer")],"next":"critic_node","architecture":result.content},goto="supervisor"
     )
  
  
  def arbiter_node(self, state: AgentState) -> Command[Literal["uml_node"]]:
    print("****************Entered arbiter node****************")
    
    sys_prompt = ("You are the Arbiter in a multi-agent architecture design system. "
        "Your role is to verify whether the proposed software architecture aligns with the user's business requirements. "
        "Carefully assess if the architecture supports the intended goals, use cases, and constraints. "
        "If the architecture is fine,then strictly do not change the current architecture. If not, modify it accordingly."
        "Give only the final, complete software architecture design,but dont say that this is the final version given by you, strictly without any special characterr (newline, '*',etc)")

    result = self.arbiter_model.invoke(f"{sys_prompt}\n\nSoftware Architecture: {state['architecture']}\n\n Requirements: {state['requirements']}")

    msg = [HumanMessage(content="[ARBITER] Final decision made. Sending architecture to user.", name="arbiter")]

    return Command(
        update={"architecture": result.content, "messages": state["messages"] + msg,"next":"FINISH"},
        goto="uml_node"
    )
  
  def uml_node(self,state : AgentState) -> Command[Literal["__end__"]]:
     print("****************Entered uml node****************")
     uml_prompt = (
        """
        Analyze the given software architecture and provide an appropriate plantuml code 
        based on the architecture, generate a clean, accurate UML class component diagram. Give only the plantUML code for it and not anything extra.  
        """
     )

     result = self.arbiter_model.invoke(f"{uml_prompt}\n\nSoftware Architecture: {state['architecture']}")
     msg = [HumanMessage(content="[UML GENERATOR] Generating UML code for the architecture")]

     return Command(
        update={"uml_code":result.content,"next":"__end__"},
        goto="__end__"
     )
 
  
  def workflow(self):
     self.graph = StateGraph(AgentState)
     self.graph.add_node("architect_node",self.architect_node)
     self.graph.add_node("arbiter_node",self.arbiter_node)
     self.graph.add_node("optimizer_node",self.optimizer_node)
     self.graph.add_node("critic_node",self.critic_node)
     self.graph.add_node("requirements_node",self.requirements_node)
     self.graph.add_node("supervisor",self.supervisor_node)
     self.graph.add_node("uml_node",self.uml_node)
     self.graph.add_edge(START,"supervisor")

     self.app = self.graph.compile()
     return self.app
