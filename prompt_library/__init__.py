members_dict = {"requirements_node":"To understand the user's query and generate system requirements so that is easy for an LLM to understand and generate complete software architecture design"
                ,"architect_node":"To provide the complete software architecture design and recommend frameworks, libraries, and deployment options, refining it for accuracy and practicality"
                ,"critic_node":"To analyze the software architecture design provided and identify Anti-patterns. If identified, point out the parts where optimization is required, with appropriate justification"
                ,"optimizer_node":"To refine the current software architecture design according to the critic's response. Should not over engineer."
                ,"arbiter_node":"To perform business alignment validation"
                }


options = list(members_dict.keys()) + ["FINISH"]

worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in members_dict.items()]) + '\n\nWORKER: FINISH \nDESCRIPTION: If User Query is answered and route to Finished'


