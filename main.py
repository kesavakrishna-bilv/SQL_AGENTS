from tempfile import NamedTemporaryFile
import os
import base64
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from a1_df_agent import SQLQueryTool
from a2_txt_agent import SQLAgentTool
from a3_viz_agent import DataVisualizationTool
# Create a prompt template
template = '''Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question and answer must always be well explained
Begin!
Question: {input}
Thought:{agent_scratchpad}'''

prompt_t = PromptTemplate.from_template(template)

# Initialize the tools and agent
tools = [SQLAgentTool(), SQLQueryTool(), DataVisualizationTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model_name="gpt-4o"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate',
    prompt=prompt_t
)

# Prompt user to upload an image file
file_path = '/home/kesava/Downloads/maxresdefault.jpg'

if os.path.exists(file_path):
    # Read the image content (no need for base64 encoding as we are not in a web environment)
    with open(file_path, "rb") as file:
        image_data = file.read()

    # Prompt the user to ask a question about the image
    user_question = input("Ask a question about your image: ")

    if user_question:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(image_data)

            # Run the agent with the question and image path
            response = agent.run(f"{user_question}, this is the image path give a 2 line answer: {temp_filename}")
            print("Agent's Response:", response)

    # Clean up temporary file
    os.remove(temp_filename)
else:
    print("The file path provided does not exist. Please try again.")
