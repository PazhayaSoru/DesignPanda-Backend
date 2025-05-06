from pydantic import BaseModel
from typing import List

class UserQuery(BaseModel):
    user_idea: str
    project_name: str
    project_type: str
    project_description: str
    scale: str
    budget: str
    project_duration: int
    security_requirements: str
    key_features: List[str]
    additional_requirements: str

class FinalOutput(BaseModel):
  project_name : str
  architecture : str
  uml_code : str

  

