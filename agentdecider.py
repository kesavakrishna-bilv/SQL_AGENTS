# from langchain_openai import ChatOpenAI  # Updated import
# from langchain.schema import SystemMessage
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env
# load_dotenv()

# # Load API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize the ChatOpenAI model
# llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# # Function to determine which agent to invoke based on the question
# def classify_and_invoke(question):
#     # Construct the prompt dynamically using the provided tool names
#     prompt = f"""
#     You are an agent capable of answering questions related to SQL queries. You have two agents available:

#     - **A1_Agent**: Handles questions that require structured, tabular results. A1_Agent takes a question, generates an appropriate SQL query, executes it, and returns the results as a DataFrame. Example questions for A1_Agent:
#       - "Get the first 5 records from the FactInternetSales table."
#       - "What are the first records in the FactInternetSales table?"

#     - **A2_Agent**: Handles questions that require simple answers, such as summaries, single values, or formatted results. A2_Agent processes the question and returns the output as a string. Example questions for A2_Agent:
#       - "What is the name of the database?"
#       - "How many rows are there in the FactInternetSales table?"

#     When you receive a question, determine the appropriate output format:
#     - If the question requires a DataFrame (structured or tabular data), invoke **A1_Agent**.
#     - If the question requires a single value, summary, or string result, invoke **A2_Agent**.

#     Answer the query by invoking the appropriate agent based on the required output format.

#     Tools available: "A1_Agent", "A2_Agent"

#     Agent's scratchpad: ""

#     Current tools: SQL execution, Summary generation

#     Question: {question}
#     """
    
#     # Use the LangChain ChatOpenAI model to generate the response
#     response = llm.invoke([SystemMessage(content=prompt)])

#     # Extract the response from the model and identify the agent
#     answer = response.content.strip()
#     if "A1_Agent" in answer:
#         return "A1_Agent"
#     elif "A2_Agent" in answer:
#         return "A2_Agent"
#     else:
#         return "Unknown Agent"

# def agent_decider(question):
#     # Run the function
#     response = classify_and_invoke(question)
#     return response


# if __name__ == "__main__":
#     question = "What is the name of the database"
#     answer = agent_decider(question)
#     print(answer)

from langchain_openai import ChatOpenAI  # Updated import
from langchain.schema import SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Function to determine which agent to invoke based on the question
def classify_and_invoke(question):
    # Construct the prompt dynamically using the provided tool names
    prompt = f"""
        You are an intelligent agent selector. Your role is to determine which agent—A1_Agent, A2_Agent, or A3_Agent—would be best suited to answer a given question. 
        You will NOT execute any agents; your task is solely to identify the most appropriate one.

        You have three agents available:

        - **A1_Agent**: Handles questions that need structured or tabular data meaning dataframe, typically involving SQL queries on databases. Choose A1_Agent for questions that:
            - Request specific records, columns, summaries in a table.
            - Require returning data in a structured, DataFrame format.
            **Example questions for A1_Agent**:
                - "Get the first 5 records from the FactInternetSales table."
                - "What are the first records in the FactInternetSales table?"
                - "Show sales data for each product category."

        - **A2_Agent**: Handles questions that need a straightforward, single-answer response, such as summaries or simple statistics. Choose A2_Agent for questions that:
            - Ask for a single value, description, or summary.
            - Expect a result in plain text (e.g., a sentence, single number, or paragraph).
            **Example questions for A2_Agent**:
                - "What is the name of the database?"
                - "How many rows are there in the FactInternetSales table?"
                - "Give a brief summary of the sales performance."

        - **A3_Agent**: Handles questions that require visualizations, such as graphs, charts, or mappings of values. Choose A3_Agent for questions that:
            - Involve drawing graphs, plotting data points, or creating visual summaries.
            - Seek to display data relationships or trends in a visual format.
            **Example questions for A3_Agent**:
                - "Generate a bar chart of monthly sales."
                - "Map the total sales by region."
                - "Draw a trend line for quarterly revenue."

        To decide which agent to assign, consider these guidelines:
        - If the question requires a DataFrame (structured/tabular data), return **A1_Agent**.
        - If the question requires a summary, single value, or textual response, return **A2_Agent**.
        - If the question requires a graph, chart, or other visualization, return **A3_Agent**.

        **Return only the name of the appropriate agent** (e.g., "A1_Agent", "A2_Agent", or "A3_Agent").

        Tools available: "A1_Agent", "A2_Agent", "A3_Agent"

        Question: {question}
        """    
    # Use the LangChain ChatOpenAI model to generate the response
    response = llm.invoke([SystemMessage(content=prompt)])

    # Extract the response from the model and identify the agent
    answer = response.content.strip()
    if "A1_Agent" in answer:
        return "A1_Agent"
    elif "A2_Agent" in answer:
        return "A2_Agent"
    elif "A3_Agent" in answer:
        return "A3_Agent"
    else:
        return "Unknown Agent"


def agent_decider(question):
    # Run the function
    response = classify_and_invoke(question)
    return response


if __name__ == "__main__":
    # question = "What is the name of the database"
    # question = "get graph for the first 5 records from FactInternetSales table?"
    # question = "How many tables are there in the database"
    # question = "List the tables in database"
    # question = "What are the tables in the database"
    question = "what tables are there in the database"
    answer = agent_decider(question)
    print(answer)