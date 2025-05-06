from langchain.prompts import PromptTemplate


initial_prompt = PromptTemplate.from_template(
    """User's Idea: {user_idea}\n

Project Type: {project_type}\n

Project Description: {project_description}\n

Expected Scale (in terms of users): {scale}\n

Budget (in rupees): {budget}\n

Project Duration (in months): {project_duration}
Security Requirements: {security_requirements}\n

Key Features:
{key_features}\n

Additional Requirements: {additional_requirements}\n
"""
)

