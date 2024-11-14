import warnings
warnings.filterwarnings("ignore")
from langchain.chat_models import LangChainDeprecationWarning

# Ignore the specific LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from datetime import datetime
import re
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
# from logging_setup import log  # Assuming logging_setup.py has been set up correctly
# logger = log()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    db = SQLDatabase.from_uri("mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server")
    # logger.info("Connected to database successfully.")
except Exception as e:
    print(e)
    # logger.error(f"Database connection failed: {e}")
    raise

# logger.info("This is from A1 Agent")
# logger.info(f"Database dialect: {db.dialect}")
# logger.info(f"Usable table names: {db.get_usable_table_names()}")

llm = ChatOpenAI(model="gpt-4o")

# Define a custom prompt template to generate only the SQL query
prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate this question into an SQL server SQL query only. Do not execute it: {question}"
)

from langchain.chains import create_sql_query_chain

llm = ChatOpenAI(model="gpt-4o", temperature=0)

chain = create_sql_query_chain(llm, db)

# Define a function to generate the SQL query
def generate_sql_query_only(question):
    # Use the LLM chain to generate SQL without executing it
    try:
        sql_query = chain.invoke({"question": question})
        # logger.info(f"Generated Sql query: {sql_query}")
        return sql_query
    except Exception as e:
        print(e)
        # logger.error(f"Error in generating SQL query: {e}")
        raise

# Execute the query and convert to DataFrame
engine = create_engine("mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server")
def query_to_dataframe(sql_query, engine):
    try:
        with engine.connect() as connection:
            df = pd.read_sql(sql_query, connection)
            # logger.info("Query executed successfully and data loaded into DataFrame.")
            # logger.info(f"Dataframe: {df}")
            return df
    except Exception as e:
        print(e)
        # logger.error(f"Error in executing SQL query or loading data: {e}")
        raise

def get_keywords_from_llm(question):
    # Craft the prompt to get relevant keywords
    n_prompt = f"Extract the main keywords that summarize this question for use in a CSV filename: '{question}'"

    # LLM call with the updated format
    response = llm.invoke([{"role": "system", "content": n_prompt}])
    
    # Access the content directly
    keywords = response.content.strip()
    keywords = re.sub(r'[^a-zA-Z0-9_]', '_', keywords)  # Sanitize keywords for filename

    return keywords

def create_csv_filename_with_llm(question):
    keywords = get_keywords_from_llm(question)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{keywords}_{timestamp}.csv"
    return filename

# Example usage
def a1_agent_function(question):
    try:
        # logger.info(f"Received question: {question}")
        sql_query = generate_sql_query_only(question)
        
        # Clean the query by removing unwanted markdown formatting, if any
        if "```" in sql_query:
            # Use regex to extract content between triple backticks, optionally starting with "sql"
            cleaned_query = re.search(r"```(?i)(sql\s*)?(.*?)\s*```", sql_query, re.DOTALL)

            # If a match is found, cleaned_query will contain the SQL query
            if cleaned_query:
                cleaned_query = cleaned_query.group(2).strip()  # Group 2 contains the actual SQL query
                # logger.info(f"Cleaned SQL query: {cleaned_query}")
            else:
                print("No SQL query found within '''sql''' markers.")
                # logger.error("No SQL query found within '''sql''' markers.")
        else:
            # If the query is already clean (no triple backticks), log it directly
            cleaned_query = sql_query.strip()
            # logger.info(f"SQL query is already clean: {cleaned_query}")

        df = query_to_dataframe(cleaned_query, engine)
        # logger.info(f"DataFrame generated from query result: {df}")
        
        # Define the output directory and ensure it exists
        output_dir = "/home/krishhindocha/Desktop/SQL AI Agent (Executor)/output_csv"
        os.makedirs(output_dir, exist_ok=True)

        # Create the full file path with the specified directory
        file_path = os.path.join(output_dir, create_csv_filename_with_llm(question))
        df.to_csv(file_path, encoding='utf-8', index=False)
        # logger.info(f"Data saved to {file_path}")
        
        return file_path
    except Exception as e:
        print(f"Error in a1_agent_function: {e}")
        # logger.error(f"Error in a1_agent_function: {e}")
        raise

# if __name__ == "__main__":
#     question = "are there any common columns in FactResellerSales and FactInternetSales table?"
#     # question = "i want the first 5 rows of the FactResellerSales table?"
#     # question = "How many tables are there in database?"
#     a1_agent_function(question)