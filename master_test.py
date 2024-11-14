from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from tools import SQLQueryTool, SQLAgentTool, DataVisualizationTool, CSVAgentTool
# Assuming CSVAgentTool and SQLAgentTool classes are already defined

class MasterAgent:
    def __init__(self, csv_file_path, db_uri, model="gpt-4o", temperature=0):
        """
        Initializes the MasterAgent with various tools and decision-making logic.

        Parameters:
            csv_file_path (str): Path to the CSV file for data analysis.
            db_uri (str): URI for the SQL database.
            model (str): Model name for the language model (default "gpt-4o").
            temperature (float): Temperature setting for the language model (default 0).
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize the base LLM
        self.llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
        
        # Initialize tools
        self.csv_agent_tool = CSVAgentTool(csv_file_path=csv_file_path, model=model, temperature=temperature)
        self.sql_agent_tool = SQLAgentTool(db_uri=db_uri)
        self.sql_query_tool = SQLQueryTool(db_uri=db_uri)
        self.visualize_tool = DataVisualizationTool(db_uri=db_uri)
        # Define available tools
        self.tools = {
            "CSVAgent": Tool(
                name="CSVAgent",
                func=self.csv_agent_tool.run_query,
                description="Useful for answering questions about CSV data."
            ),
            "SQLAgent": Tool(
                name="SQLAgent",
                func=self.sql_agent_tool.query_database,
                description="Useful for answering questions about SQL data."
            ),
            "SQLQuery": Tool(
                name = "SQLQuery",
                func = self.sql_query_tool.run_and_exe,
                description = "Useful for answering quantitative questions about SQL data and giving them as a dataframe."
            ),
            "Visualize": Tool(
                name = "Visualize",
                func = self.visualize_tool.total_viz,
                description = "Useful for answering quantitative questions about SQL data and giving them as a dataframe."
            )
        }

    def decide_and_execute(self, query):
        """
        Uses the language model to decide which tool to use based on the query and executes the query with the chosen tool.

        Parameters:
            query (str): The input query.

        Returns:
            The response from the chosen tool.
        """
        # Create a prompt for the language model to decide which tool to use
        decision_prompt = f"""
        You are an intelligent assistant with access to two tools:
        
        - CSVAgent: Used for questions related to CSV files or dataframes.
        - SQLAgent: Used for questions related to SQL databases.
        - SQLQuery: Used for questions related to SQL databases which require a quantitative output as a dataframe.
        - Visualize: Used for visualising the question related to SQL databases in graphs
        
        Given the user's question, decide which tool to use by outputting either "CSVAgent" or "SQLAgent", or "SQLQuery" or "Visualize".
        Then, execute the question using the selected tool.
        
        User's question: "{query}"
        
        Please reply with only the name of the chosen tool.
        """

        # Run the decision prompt with the language model
        decision = self.llm(decision_prompt).content.strip()
        
        # Validate and choose the tool
        chosen_tool_name = decision if decision in self.tools else "CSVAgent"
        # chosen_tool_name = "Visualize"
        chosen_tool = self.tools[chosen_tool_name]
        
        print(f"Using tool: {chosen_tool_name}")
        
        # Execute the query using the chosen tool
        return chosen_tool.func(query)

# Usage example
if __name__ == "__main__":
    # Initialize the MasterAgent with paths to CSV and SQL data sources
    csv_file_path = "output.csv"
    db_uri = "mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server"
    master_agent = MasterAgent(csv_file_path=csv_file_path, db_uri=db_uri)

    # Run a sample query
    query = "i want you to visualize and give the contents of the fact reseller sales table present in the database in graphs"
    response = master_agent.decide_and_execute(query)
    print(response)
