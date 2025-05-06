from fastapi import FastAPI
from prompt_library.prompts import initial_prompt
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agent import SadAgent
from data_models.models import FinalOutput,UserQuery
from utils.uml_gen import uml_gen
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = SadAgent()

@app.get('/')

@app.post("/execute")
async def execute_agent(user_query : UserQuery):
  app_graph = agent.workflow()

  formatted_user_query = initial_prompt.format(
    user_idea=user_query.user_idea,
    project_type=user_query.project_type,
    project_description=user_query.project_description,
    scale=user_query.scale,
    budget=user_query.budget,
    project_duration=user_query.project_duration,
    security_requirements=user_query.security_requirements,
    key_features="\n- " + "\n- ".join(user_query.key_features),
    additional_requirements=user_query.additional_requirements
  )

  query_data = {
  "requirements":"",
  "messages" : [HumanMessage(content=formatted_user_query)],
  "next" : "requirements_node",
  "architecture" : "",
  "criticism" : "",
  "optimizer_visits" : 0,
  "optimization_details" :"",
  "uml_code":""
  }

  response =  app_graph.invoke(query_data)

  # uml_code = response['uml_code']
  uml_gen(response['uml_code'])

  with open("example.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

  return {"architecture":response['architecture'],
          "uml_code":response['uml_code'],
          "image_data":encoded_image,
          "mime_type":"image/png"}


  


