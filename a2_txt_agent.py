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

if __name__ == "__main__":
        
    # Usage example:
    # Initialize the SQLAgentTool
    db_uri = "mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server"
    sql_agent_tool = SQLAgentTool(db_uri)

    # List usable tables
    print("Available Tables:", sql_agent_tool.list_tables())

    # Query the database
    question = "what is the database about?"
    response = sql_agent_tool.query_database(question)
    print("Query Response:", response)
