import warnings
warnings.filterwarnings("ignore")
from langchain.chat_models import LangChainDeprecationWarning

# Ignore the specific LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
from logging_setup import log
from langchain.prompts import PromptTemplate
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
import os
from a2_agent import a2_agent_function
from a1_agent import a1_agent_function
from a3_agent import a3_agent_function
from dotenv import load_dotenv
from agentdecider import agent_decider
from langchain import hub

logger=log()
# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



def a_agent(question):
    logger.info("Initializing Agent Decider tool for question: %s", question)

    # List of tools available to the agent
    tools = [
        Tool(
            name="agent_decider",  # Name of the tool
            func=agent_decider,  # Function that the tool will execute
            description="Useful for deciding which agent is appropriate for which question but it does not invoke that agent",
        ),
    ]

    template = '''
    Your task is to decide which agent should handle the given question and return only the name of the agent.

    Agents:
    - **A1_Agent**: Use this for questions requiring structured or tabular data, like retrieving specific rows or columns from a database.
    - **A2_Agent**: Use this for questions that require a single value, summary, or a simple answer.
    - **A3_Agent**: Use this for questions that require a graph or mapping or drawing.
    To make the decision, you have access to a tool in {tools} which will help you identify the appropriate agent without executing it.

    Return Format:
    - Thought: Briefly consider the question and choose the appropriate agent by using {tool_names}.
    - Action: Select the "agent_decider" tool and provide the question.
    - Final Answer: The output should be only the agent name, either "A1_Agent" or "A2_Agent" or "A3_Agent", with no additional text.

    Question: {input}
    Thought: {agent_scratchpad}
    Final Answer: [Output should be exactly "A1_Agent" or "A2_Agent" or "A3_Agent" without additional details]
    '''

    # Initialize the PromptTemplate
    prompt_t = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template=template
    )

    # Update the prompt template with this refined version
    prompt_t = PromptTemplate.from_template(template)

    # Initialize ChatGroq with specified parameters
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info("ChatGroq model initialized.")

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_t,
        stop_sequence=True,
    )

    logger.info("ReAct agent created successfully.")

    # Create an agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    try:
        # Run the agent with the provided question
        logger.info("Invoking Agent Decider agent with question: %s", question)
        response = agent_executor.invoke({"input": question})
        logger.info("Received response from Agent Decider Agent: %s", response)
        return response
    except Exception as e:
        # logger.error("Error invoking A2_Agent: %s", e)
        response = e

def a2agent(question):
    logger.info("Initializing A2_Agent tool for question: %s", question)

    # List of tools available to the agent
    tools = [
        Tool(
            name="A2_Agent",  # Name of the tool
            func=a2_agent_function,  # Function that the tool will execute
            description="Useful for questions that can be answered only by A2 Agent",
        ),
    ]

    # Pull the prompt template from the hub
    prompt = hub.pull("hwchase17/react")
    logger.info("Prompt template pulled successfully from hub.")

    # Initialize ChatGroq with specified parameters
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info("ChatGroq model initialized.")

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence=True,
    )
    logger.info("ReAct agent created successfully.")

    # Create an agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    try:
        # Run the agent with the provided question
        logger.info("Invoking agent with question: %s", question)
        response = agent_executor.invoke({"input": question})
        logger.info("Received response from A2_Agent: %s", response)
    except Exception as e:
        logger.error("Error invoking A2_Agent: %s", e)
        response = None

    return response

def a1agent(question):
    logger.info("Initializing A1_Agent tool for question: %s", question)

    # List of tools available to the agent
    tools = [
        Tool(
            name="A1_Agent",  # Name of the tool
            func=a1_agent_function,  # Function that the tool will execute
            description="Useful for questions that can be answered only by A1 Agent.",
        ),
    ]

    # Pull the prompt template from the hub
    prompt_t = hub.pull("hwchase17/react")
    logger.info("Prompt template pulled successfully from hub.")

    # Initialize ChatGroq with specified parameters
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info("ChatGroq model initialized.")
    logger.info("Creating ReAct agent with prompt: %s", prompt_t)
    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_t,
        stop_sequence=True,
    )
    logger.info("ReAct agent created successfully.")

    # Create an agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    try:
        # Run the agent with the provided question
        logger.info("Invoking agent with question: %s", question)
        response = agent_executor.invoke({"input": question})
        logger.info("Received response from A1_Agent: %s", response)
    except Exception as e:
        logger.error("Error invoking A1_Agent: %s", e)
        response = None

    return response

def a3agent(question):
    logger.info("Initializing A3_Agent tool for question: %s", question)

    # List of tools available to the agent
    tools = [
        Tool(
            name="A3_Agent",  # Name of the tool
            func=a3_agent_function,  # Function that the tool will execute
            description="Useful for questions that can be answered only by A3 Agent",
        ),
    ]

    # Pull the prompt template from the hub
    prompt = hub.pull("hwchase17/react")
    logger.info("Prompt template pulled successfully from hub.")

    # Initialize ChatGroq with specified parameters
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info("ChatGroq model initialized.")

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence=True,
    )
    logger.info("ReAct agent created successfully.")

    # Create an agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    try:
        # Run the agent with the provided question
        logger.info("Invoking agent with question: %s", question)
        response = agent_executor.invoke({"input": question})
        logger.info("Received response from A3_Agent: %s", response)
    except Exception as e:
        logger.error("Error invoking A3_Agent: %s", e)
        response = None

    return response

def main(question):
    logger.info(f"Got question: {question}")    
    logger.info("Determining which agent to use to handle the question: %s", question)
    response = a_agent(question)
    logger.info("Selected Agent: %s", response)

    if response['output'] == 'A1_Agent':
        agent_response = a1agent(question)
        if agent_response:
            logger.info(f"Response:, {agent_response}")
            logger.info(type(agent_response))

            directory = '/home/krishhindocha/Desktop/SQL AI Agent (Executor)/output_csv'
            # Get all files in the directory
            files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            
            # If no files are found, return None
            if not files:
                return None

            # Get the file with the latest modification time
            latest_file = max(files, key=os.path.getmtime)
            if latest_file:
                print(f"The latest file is: {latest_file}")
                import pandas as pd
                # Replace 'your_file.csv' with the path to your CSV file
                df = pd.read_csv(latest_file)

                # Print the DataFrame
                print(df) 
                print(type(df))
                return latest_file
            else:
                print("No files found in the directory.")

        else:
            logger.warning("No response returned from A1_Agent.")

    elif response['output'] == 'A2_Agent':
        agent_response = a2agent(question)
        if agent_response:
            logger.info(f"Response:, {agent_response}")
            return agent_response
        else:
            logger.warning("No response returned from A2_Agent.")
    else:
        agent_response = a1agent(question)
        if agent_response:
            logger.info(f"Response from A1 Agent:, {agent_response}")
            logger.info(type(agent_response))

            directory = '/home/krishhindocha/Desktop/SQL AI Agent (Executor)/output_csv'
            # Get all files in the directory
            files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            
            # If no files are found, return None
            if not files:
                return None

            # Get the file with the latest modification time
            latest_file = max(files, key=os.path.getmtime)
            if latest_file:
                print(f"The latest file is: {latest_file}")
                agent_response = a3agent(question)
                if agent_response:
                    logger.info(f"Response from A3 Agent:, {agent_response}")
                    logger.info(type(agent_response))
                else:
                    logger.warning("No response returned from A1_Agent.")


                return latest_file
            else:
                print("No files found in the directory.")

        else:
            logger.warning("No response returned from A1_Agent.")


# Main section
if __name__ == "__main__":
    # # question = "What is the name of the database?"
    # question = "get the first 5 records from FactInternetSales table?"
    question = "get graph for the first 5 records from FactInternetSales table?"
    main(question)

"""
What is the name of the database?

how many tables are there in the database

get the first 5 records from FactInternetSales table?
"""
