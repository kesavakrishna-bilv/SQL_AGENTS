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

from a1_df_agent import SQLQueryTool
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
        Generates charts based on the LLM response.

        Parameters:
            output (dict): JSON response with chart details.
            df (pd.DataFrame): The full dataset as a DataFrame.
        """
        for chart_info in output.get('charts', []):
            try:
                # Extract chart parameters
                params = chart_info['parameters']
                params['data_frame'] = df
                params['title'] = chart_info['title']
                
                # Generate chart based on specified chart type
                chart_type = chart_info['chartType']
                fig = getattr(px, chart_type)(**params)  # e.g., px.line, px.bar, etc.
                fig.show()
                
            except Exception as e:
                print(f"Visualization for '{chart_info['title']}' failed: {str(e)}")
                print(chart_info)

    def total_viz(self, query: str):
        self.run_and_exe(query)
        df, data_sample = self.load_data("output.csv")
        num_charts = 5
        visualization_instructions = self.get_visualization_instructions(num_charts, data_sample)
        self.create_charts(visualization_instructions, df)



if __name__ == "__main__":
    
    # Initialize tool with API key and temperature
    api_key = os.getenv("OPENAI_API_KEY")
    tool = DataVisualizationTool(api_key, temperature=0.0)

    # Load dataset and get a data sample
    df, data_sample = tool.load_data("output.csv")

    # Get chart generation instructions
    num_charts = 5
    visualization_instructions = tool.get_visualization_instructions(num_charts, data_sample)

    # Generate charts
    tool.create_charts(visualization_instructions, df)
