import warnings
warnings.filterwarnings("ignore")
from langchain.chat_models import LangChainDeprecationWarning
 
# Ignore the specific LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
 
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import FewShotPromptTemplate
import pandas as pd
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain import hub
from a1_agent import a1_agent_function
 
# Load environment variables
load_dotenv()
 
class SQLQueryTool:
    def __init__(self, db_uri: str, llm_model: str = "gpt-4o"):
        # Initialize LLM and Database connection
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.db_uri = db_uri
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.db = SQLDatabase.from_uri(db_uri)
        self.engine = create_engine(db_uri)
 
        # Set table_info and top_k
        table_info = list(['table'])  # Example table info, adjust as needed
        top_k = 10  # Example, adjust based on your requirement
 
        examples = [
            {"input": "top 10 employees", "query": "select top 10 emp from emp_table"}
        ]
        example_prompt = PromptTemplate.from_template("Questions: {input}\nSQL query: {query}")
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Given an input question, first create a syntactically correct mssql query to run, then look at the results of the query and return the answer. Only return SQL Query not anything else like ```sql ... `. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. Never query for all the columns from a specific table; only ask for the few relevant columns given the question. Pay attention to use only the column names that you can see in the schema description. Be careful not to query for columns that do not exist. Also, pay attention to which column is in which table. \n\n Only use the following tables: \n {table_info}\n Question: {input}",
            suffix="User Question: {input}\n SQL query: ",
            input_variables=["input", "top_k", "table_info"],
        )
 
        # Create the LLM chain with the modified prompt
        self.llm_chain = create_sql_query_chain(llm=self.llm, db=self.db, prompt=prompt, k=top_k)
 
    def generate_sql_query(self, question: str) -> str:
        """Generate SQL query from a natural language question without executing."""
        sql_query = self.llm_chain.invoke({
            "question": question,  # Change "input" to "question"
            "top_k": 5,  # or another suitable default for the number of results
            "table_info": self.db.get_table_info()
        })
        
        # Clean up any extraneous formatting
        return sql_query.strip().replace("```sql", "").replace("```", "")
 
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute the generated SQL query and return results as a DataFrame."""
        with self.engine.connect() as connection:
            df = pd.read_sql(sql_query, connection)
        return df
 
    def save_to_csv(self, df: pd.DataFrame, file_name: str = "output.csv") -> None:
        """Save the DataFrame to a CSV file."""
        df.to_csv(file_name, encoding='utf-8', index=False)
 
    def run_and_exe(self, question) -> pd.DataFrame:
        """Fetch all records from a specified table."""
        sql_query = self.generate_sql_query(question)
        df = self.execute_query(sql_query)
        self.save_to_csv(df)
        return df
 
    def list_tables(self) -> list:
        """List usable table names in the connected database."""
        return self.db.get_usable_table_names()
 
 
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
 
# Load environment variables
load_dotenv()
 
class SQLAgentTool:
    def __init__(self, db_uri: str, llm_model: str = "gpt-4o"):
        """
        Initialize the SQLAgentTool with a database connection and language model.
 
        Parameters:
            db_uri (str): The URI for connecting to the database.
            llm_model (str): The name of the language model to use. Default is "gpt-3.5-turbo".
        """
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Initialize the SQL agent executor
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=True
        )
 
    def list_tables(self) -> list:
        """List usable table names in the connected database."""
        return self.db.get_usable_table_names()
 
    def query_database(self, question: str) -> str:
        """
        Use the agent to process a natural language question and execute it on the database.
 
        Parameters:
            question (str): The natural language question to interpret and query the database.
 
        Returns:
            str: The result of the executed query.
        """
        response = self.agent_executor.invoke(question)
        return response
 
import os
import pandas as pd
import json
import plotly.express as px
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import FewShotPromptTemplate
# Load environment variables
load_dotenv()
 
class DataVisualizationTool(SQLQueryTool):
    def __init__(self, db_uri:str, llm_model: str = "gpt-4o"):
        """
        Initializes the DataVisualizationTool with a language model and settings.
        
        Parameters:
            api_key (str): OpenAI API key.
            model (str): Model to use for generating responses. Default is 'gpt-3.5-turbo'.
            temperature (float): Temperature for response generation. Default is 0.0.
        """
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.db_uri = db_uri
        self.db = SQLDatabase.from_uri(db_uri)
        self.engine = create_engine(db_uri)
        top_k = 10  # Example, adjust based on your requirement
 
        examples = [
            {"input": "top 10 employees", "query": "select top 10 emp from emp_table"}
        ]
        example_prompt = PromptTemplate.from_template("Questions: {input}\nSQL query: {query}")
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Given an input question, first create a syntactically correct mssql query to run, then look at the results of the query and return the answer. Only return SQL Query not anything else like ```sql ... `. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. Never query for all the columns from a specific table; only ask for the few relevant columns given the question. Pay attention to use only the column names that you can see in the schema description. Be careful not to query for columns that do not exist. Also, pay attention to which column is in which table. \n\n Only use the following tables: \n {table_info}\n Question: {input}",
            suffix="User Question: {input}\n SQL query: ",
            input_variables=["input", "top_k", "table_info"],
        )
 
        # Create the LLM chain with the modified prompt
        self.llm_chain = create_sql_query_chain(llm=self.llm, db=self.db, prompt=prompt, k=top_k)
 
 
    def load_data(self, dataset: str) -> tuple:
        """
        Loads dataset from a .csv file and returns both the DataFrame and a 10-row sample as a CSV string.
 
        Parameters:
            dataset (str): Path to the CSV dataset.
 
        Returns:
            tuple: The full DataFrame and a sample as a CSV string.
        """
        try:
            df = pd.read_csv(dataset)
        except FileNotFoundError:
            df = pd.read_csv('./data/car_sales.csv')
        
        data_sample = df.head(10).to_csv(index=False)
        return df, data_sample
 
    def generate_prompt(self, num_charts: int, data_sample: str) -> list:
        """
        Generates a prompt for the LLM to create data visualizations.
 
        Parameters:
            num_charts (int): Number of charts to generate.
            data_sample (str): A sample of the data in CSV format.
 
        Returns:
            list: The prompt formatted for ChatOpenAI.
        """
        system_template = """
        The following is a conversation between a Human and an AI assistant expert on data visualization with perfect Python 3 syntax. The human will provide a sample dataset for the AI to use as the source. The real dataset that the human will use with the response of the AI is going to have several more rows. The AI assistant will only reply in the following JSON format:
 
        {{
        "charts": [{{'title': string, 'chartType': string, 'parameters': {{...}}}}, ... ]
        }}
 
        Instructions:
 
        1. chartType must only contain method names from plotly.express, e.g., 'line', 'bar', 'scatter'.
        2. For each chartType, parameters must contain the value to be used for all parameters of that plotly.express method.
        3. There should be 4 parameters for each chart.
        4. Do not include "data_frame" in the parameters.
        5. There should be {num_charts} charts in total.
        """
 
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "This is the dataset:\n\n{data}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
 
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        prompt = chat_prompt.format_prompt(num_charts=str(num_charts), data=data_sample).to_messages()
 
        return prompt
 
    def get_visualization_instructions(self, num_charts: int, data_sample: str) -> dict:
        """
        Calls the language model to generate visualization instructions.
 
        Parameters:
            num_charts (int): Number of charts to generate.
            data_sample (str): A sample of the data in CSV format.
 
        Returns:
            dict: JSON response with chart details.
        """
        prompt = self.generate_prompt(num_charts, data_sample)
        
        # with get_openai_callback() as cb:
        response = self.llm.invoke(prompt)  # Use .invoke() instead of calling directly
        #     total_token = cb.total_tokens
        #     total_cost = cb.total_cost
        
        # print(f"#### Tokens: {total_token}")
        # print(f"#### Cost (USD): {total_cost}")
 
        print("LLM Response:", response.content)  # Print response to verify
 
        # Remove code block markers if they exist
        cleaned_content = response.content.strip("```json").strip("```")
 
        # Parse JSON content from the cleaned response
        try:
            output = json.loads(cleaned_content)
            return output
        except json.JSONDecodeError:
            print("Error parsing JSON from LLM response")
            return {}
 
 
    def create_charts(self, output: dict, df: pd.DataFrame):
            """
            Generates and displays charts based on the LLM response within Streamlit.
    
            Parameters:
                output (dict): JSON response with chart details.
                df (pd.DataFrame): The full dataset as a DataFrame.
            """
            st.header("Generated Charts")
            
            for chart_info in output.get('charts', []):
                try:
                    # Extract chart parameters
                    params = chart_info['parameters']
                    params['data_frame'] = df
                    params['title'] = chart_info['title']
                    
                    # Generate chart based on specified chart type
                    chart_type = chart_info['chartType']
                    fig = getattr(px, chart_type)(**params)  # e.g., px.line, px.bar, etc.
                    
                    # Display the chart in Streamlit
                    st.subheader(chart_info['title'])
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Visualization for '{chart_info['title']}' failed: {str(e)}")
                    st.write(chart_info)
                    
    def a1agent(self, question):
        # logger.info("Initializing A1_Agent tool for question: %s", question)
 
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
 
        # logger.info("Prompt template pulled successfully from hub.")
 
        # logger.info("Creating ReAct agent with prompt: %s", prompt_t)
        
        # Create the ReAct agent
        agent = create_react_agent(
            llm=self.llm,  # Use self.llm if this ChatGroq model is meant to be shared across methods
            tools=tools,
            prompt=prompt_t,
            stop_sequence=True,
        )
        # logger.info("ReAct agent created successfully.")
 
        # Create an agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
 
        try:
            # Run the agent with the provided question
            # logger.info("Invoking agent with question: %s", question)
            response = agent_executor.invoke({"input": question})
            # logger.info("Received response from A1_Agent: %s", response)
        except Exception as e:
            # self.logger.error("Error invoking A1_Agent: %s", e)
            response = None
 
        return response
 
    def total_viz(self, query: str):
        # self.run_and_exe(query)
        agent_response = self.a1agent(query)
        if agent_response:
            # logger.info(f"Response:, {agent_response}")
            # logger.info(type(agent_response))
 
            self.directory = '/home/krishhindocha/Desktop/SQL AI Agent (Executor)/output_csv'
            # Get all files in the directory
            self.files = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))]
            
            # If no files are found, return None
            if not self.files:
                return None
 
            # Get the file with the latest modification time
            self.latest_file = max(self.files, key=os.path.getmtime)
            if self.latest_file:
                print(f"The latest file is: {self.latest_file}")
 
            df, data_sample = self.load_data(self.latest_file)
            num_charts = 5
            visualization_instructions = self.get_visualization_instructions(num_charts, data_sample)
            self.create_charts(visualization_instructions, df)
        else:
            print("A3 Agent failed")
 
 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
 
class CSVAgentTool:
    def __init__(self, csv_file_path, model="gpt-4o", temperature=0, verbose=True, allow_dangerous_code=True):
        """
        Initializes the CSVAgentTool.
 
        Parameters:
            csv_file_path (str): Path to the CSV file for data analysis.
            model (str): Model name to use for the language model (default "gpt-4o").
            temperature (float): Temperature setting for the model (default 0).
            verbose (bool): Whether to print verbose output (default True).
            allow_dangerous_code (bool): Whether to allow dangerous code execution (default True).
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("OPENAI_API_KEY")
 
        # Initialize the language model
        self.llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=self.api_key)
        
        # Create the CSV agent with the specified configuration
        self.agent = create_csv_agent(
            llm=self.llm,
            path=csv_file_path,
            verbose=verbose,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=allow_dangerous_code
        )
    
    def run_query(self, query):
        """
        Runs a specified query on the CSV agent.
 
        Parameters:
            query (str): The query to execute on the CSV data.
 
        Returns:
            The response from the CSV agent.
        """
        try:
            response = self.agent.run(query)
            return response
        except Exception as e:
            print(f"An error occurred while running the query: {e}")
            return None