"""

@author: Vijay B
"""

import os
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from vertexai.generative_models import GenerationConfig
import vertexai.preview.generative_models as generative_models
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_community.document_loaders import PyPDFLoader
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from functools import reduce
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
# Load configuration
APPLICATION_CONFIG = json.loads(open('D:\\Work\\2025\\Agentic AI Simulator\\dev_config.json', "r").read())
SERVICE_ACCOUNT_FILE = 'D:\\Work\\2025\\Casanova-EBS\\BE5\\be5\\config\\service_account.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_FILE
# Initialize VertexAI
vertexai.init(project=APPLICATION_CONFIG["PROJECT"], location="us-central1")
memory = MemorySaver()  # In production this should be changed to PostgreSaver
config = {"configurable": {"thread_id": "1"}}
# Initialize LLM
llm = init_chat_model("gemini-1.5-pro-002")

class OverallState(TypedDict):
    start_con: bool
    user: str
    topic_selection: str
    current_context: str
    generated_incidence: str
    language: str
    messages: Annotated[list, add_messages]
    cur_user_question: str
    


def chatbot(state: OverallState):
   system_prompt = (
         "You are a helpful assistant."
         "Your taks is to Guide the user for the Root Cause analysis they are doing , below is the given Scenario they are working on and the response from user "
         "Be polite with them always and explain them like 5 year old"
     )
   snapshot = graph.get_state(config)
   print(snapshot.values['messages'])
   messages = [SystemMessage(content=system_prompt)]+state['messages'] #Adding Historical Chat
   response = llm.invoke(messages,config)
   
   return {"messages": response}#{"messages": [HumanMessage(content=f'{final_result_dicts}')]}#reduce(lambda d1, d2: {**d1, **d2}, final_result_dicts)}

def node_1(state: OverallState):
    # Get user details from the Hosted Endpoints
    context = '#####'
    page_content = []
    for p in pages:
        page_content.append(p.page_content)
    context = '#####'.join(page_content)
   # print(context)
    return {'user': state['user'], 'topic_selection': state['topic_selection'], 'current_context': context, 'language': state['language']}

def node_cond_chat(state: OverallState):
    print('Checking condictions')
    print(state['start_con'])
    if state['start_con'] == True:
        print('Starting Conversation')
        return 'chat'
    else:
        print('Generating the Incidence...')
        return 'gen_inc'

def node_2(state: OverallState):
    print('Im creating new incidence')
    generation_config = {
        "max_output_tokens": 500,
        "temperature": 0.1,
        "top_p": 0.3,
    }

    vertexai.init(project=APPLICATION_CONFIG["PROJECT"], location="us-central1")
    prompt = f"""You are a Maintaiance Expert at Cement manufacturing industry 
    ##
    Create an incident based in the given SOP in triple backticks : '''{state['current_context']}''' 
    ##
    You will create a hypothetical incident happened at Cement plant , this incident should be based on given SOP 
    ##
    You will only generate the description of the given incident , and start the conversation with the user who is given a task to find RCA for the simulated incident , basically you need to judge him/her 
    ##
    The incidence should be generated to teach user Root cause analysis ,After generating the incidence ask user basic details like Name , Occupation 
    ##
    the final output should be in JSON format e.g. {{'incident':Description of incident}}
    ##
    """
    model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=["""Simulate an incident based on the given context 
                            ##
                            at the end of simulated incident ask user basic details like name , occupation and ask them "How would they approach this scenario?"
                            ##
                            Make sure the generated incidence is in JSON format and has only one key 'incident' as key and no other keys """]
    )

    responses = model.generate_content([prompt], generation_config=GenerationConfig(response_mime_type="application/json",max_output_tokens=500,temperature=0.1), stream=True)
    t = []
    for response in responses:
        t.append(response.text)
    z = ''.join(t)
    z = z.strip()
    try:
        question = eval(z)
    except:
        question = {"incident": None}
    try:
        #state['messages']= 
        return {'generated_incidence': question,'messages':[AIMessage(f"The Simulated Incidence is : /n {question}")]}
    except:
        return {"question_generation_error": True}

def check_incidence_generation_error(state: OverallState):
    if state.get('generated_incidence') is not None:
        return 'True'
    else:
        return 'False'

def node_4(state: OverallState):
    # Return the entire session state to the client/server
    
    return state

def add_human_msg(state:OverallState):
    return {"messages":[HumanMessage(content=f"{state['cur_user_question']}")]}

# --- FastAPI and Pydantic Models ---

class OverallStateRequest(BaseModel):
    start_con: bool
    user: str
    topic_selection: str
    language: str
    cur_user_question: str

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for PDF and Graph ---
pages = []
graph = None

#graph_chat = None
#graph_chat_comp=None

@app.on_event("startup")
def startup_event():
    global pages, graph , graph_chat,graph_chat_comp
    # Load PDF pages synchronously
    loader = PyPDFLoader('D:\\Work\\2025\\Agentic AI Simulator\\Remplacement blindages dans le tube broyeur BK4, BK5.pdf')
    pages = loader.load()
    print('Pages loaded successfully')
    # Compile the graph
    graph_builder = StateGraph(OverallState)
    graph_builder.add_node("node_1", node_1)
    graph_builder.add_node("node_2", node_2)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("node_4", node_4)
    graph_builder.add_node("add_human_msg",add_human_msg)
    
    graph_builder.add_edge(START, 'node_1')
    graph_builder.add_conditional_edges("node_1", node_cond_chat, {'chat': 'add_human_msg', 'gen_inc': "node_2"})
    graph_builder.add_conditional_edges("node_2", check_incidence_generation_error, {'True': 'node_4', 'False': 'node_2'})
    graph_builder.add_edge('add_human_msg', 'chatbot')
    graph_builder.add_edge('chatbot', 'node_4')
    graph_builder.add_edge("node_4", END)
    print('Compiling the graph')
    graph = graph_builder.compile(checkpointer=memory)
    
    #graph_chat=StateGraph(State_chat)
    #graph_chat.add_edge(START, "chatbot")
   # graph_chat_comp = graph_chat.compile(checkpointer=memory)
    #
    

@app.post("/run-incident-graph")
def run_incident_graph(payload: OverallStateRequest):
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized.")
    
    # Prepare initial state for the graph
    state = {
        'start_con': payload.start_con,
        'user': payload.user,
        'topic_selection': payload.topic_selection,
        'language': payload.language,
        'cur_user_question': payload.cur_user_question,
    }
    
    try:
        final_result_dicts=[]
        for event in graph.stream(state,{"configurable": {"thread_id": "1"}}):
            for value in event.values():
                final_result_dicts.append(value)
        result = reduce(lambda d1, d2: {**d1, **d2}, final_result_dicts)
        keys_to_include = ['user', 'topic_selection','generated_incidence','messages']

        sub_dict = {key: result[key] for key in keys_to_include if key in result}
        #print("Returning : ",sub_dict)
        
        return sub_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#http://localhost:8000/
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
