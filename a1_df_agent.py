import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import FewShotPromptTemplate
import pandas as pd

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

if __name__ == "__main__":
    # Usage
    db_uri = "mssql+pyodbc://sa:btcde%40123@10.0.0.200/AdventureWorksDW2019?driver=ODBC+Driver+17+for+SQL+Server"
    sql_tool = SQLQueryTool(db_uri)

    # Generate SQL query
    question = "i want each table name with how many records it has."
    sql_query = sql_tool.generate_sql_query(question)
    print("Generated SQL Query:", sql_query)
    output = sql_tool.execute_query(sql_query)
    
    print("output of query", output)
    print("dtype of query", type(output))
    
    # Save DataFrame to CSV
    sql_tool.save_to_csv(output)

    # List usable tables
    print("Available Tables:", sql_tool.list_tables())
