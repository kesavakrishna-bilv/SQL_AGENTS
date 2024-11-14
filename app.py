import streamlit as st
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import os
from tools import SQLQueryTool, SQLAgentTool, DataVisualizationTool, CSVAgentTool
from sqlalchemy import create_engine

class MasterAgent:
    def __init__(self, csv_file_path, db_uri, model="gpt-4o", temperature=0):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

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
                name="SQLQuery",
                func=self.sql_query_tool.run_and_exe,
                description="Useful for answering quantitative questions about SQL data and giving them as a dataframe."
            ),
            "Visualize": Tool(
                name="Visualize",
                func=self.visualize_tool.total_viz,
                description="Useful for visualizing SQL data in graphs."
            )
        }

    def decide_and_execute(self, query):
        decision_prompt = f"""
        You are an intelligent assistant with access to tools:
        
        - CSVAgent: For questions related to CSV files or dataframes.
        - SQLAgent: For questions related to SQL databases.
        - SQLQuery: For quantitative SQL database queries, returning data as a dataframe.
        - Visualize: For visualizing SQL database queries as graphs.
        
        Given the user's question, decide the appropriate tool:
        "{query}"
        
        Please reply with only the name of the chosen tool.
        """
        decision = self.llm(decision_prompt).content.strip()
        chosen_tool_name = decision if decision in self.tools else "CSVAgent"
        chosen_tool = self.tools[chosen_tool_name]
        st.write(f"Using tool: {chosen_tool_name}")

        return chosen_tool.func(query)

# Streamlit UI
def main():
    st.title("MasterAgent Query Interface")
    
    # Configuration
    csv_file_path = st.sidebar.text_input("CSV File Path", "output.csv")
    # db_uri = st.sidebar.text_input("Database URI", "mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server")
    db_uri = "mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine("mssql+pyodbc://username:password@your_server/database?driver=ODBC+Driver+17+for+SQL+Server", connect_args={'timeout': 30})

    # Instantiate the MasterAgent
    if "agent" not in st.session_state:
        st.session_state["agent"] = MasterAgent(csv_file_path=csv_file_path, db_uri=db_uri)
    agent = st.session_state["agent"]
    
    # User query input
    query = st.text_area("Enter your query:", "e.g., Show a visualization of reseller sales from the database")
    
    if st.button("Run Query"):
        with st.spinner("Processing..."):
            response = agent.decide_and_execute(query)
            st.write(response)
            if isinstance(response, dict) and "charts" in response:  # Check if response contains charts
                agent.create_charts(response, df=agent.some_dataframe_method())  # Pass the relevant DataFrame
            elif isinstance(response, str):
                st.write(response)
            elif isinstance(response, pd.DataFrame):
                st.write(response)


if __name__ == "__main__":
    main()
