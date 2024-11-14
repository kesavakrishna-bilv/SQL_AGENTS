from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

class MultiCSVAgentTool:
    def __init__(self, csv_file_paths, model="gpt-4o", temperature=0, verbose=True, allow_dangerous_code=True):
        """
        Initializes the MultiCSVAgentTool for handling multiple CSV files.

        Parameters:
            csv_file_paths (list): List of CSV file paths for data analysis.
            model (str): Model name to use for the language model (default "gpt-4o").
            temperature (float): Temperature setting for the model (default 0).
            verbose (bool): Whether to print verbose output (default True).
            allow_dangerous_code (bool): Whether to allow dangerous code execution (default True).
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("OPENAI_API_KEY")

        # Initialize the language model
        self.llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=self.api_key)

        # Create a dictionary to store agents for each CSV file
        self.agents = {}
        for file_path in csv_file_paths:
            agent = create_csv_agent(
                llm=self.llm,
                path=file_path,
                verbose=verbose,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                allow_dangerous_code=allow_dangerous_code
            )
            self.agents[file_path] = agent  # Store agent with file path as key

    def run_query(self, query, file_identifier=None):
        """
        Runs a specified query on the CSV agent(s).

        Parameters:
            query (str): The query to execute on the CSV data.
            file_identifier (str): The file path or identifier to select a specific agent.

        Returns:
            The response(s) from the CSV agent(s).
        """
        if file_identifier:
            # Run query on a specific agent
            agent = self.agents.get(file_identifier)
            if agent:
                return agent.run(query)
            else:
                print(f"No agent found for identifier {file_identifier}")
                return None
        else:
            # Run query on all agents if no identifier is specified
            return {file_path: agent.run(query) for file_path, agent in self.agents.items()}

# Usage example:
if __name__ == "__main__":
    # Initialize the MultiCSVAgentTool with multiple CSV file paths
    csv_agent_tool = MultiCSVAgentTool(csv_file_paths=["output.csv", "test_cases.csv"])
    
    # Run a query on all agents
    query = "is there any relation between both dataframes"
    response = csv_agent_tool.run_query(query)
    print(response)

    # Run a query on a specific file
    # specific_response = csv_agent_tool.run_query(query, file_identifier="output.csv")
    # print(specific_response)
