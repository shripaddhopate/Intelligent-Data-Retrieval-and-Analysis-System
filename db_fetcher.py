import pandas as pd
from tabulate import tabulate
from urllib.parse import quote_plus
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import sqlalchemy
from config.settings import API_KEY, model_name
from config.db_config import DB_CONFIG

class DBFetcher:
    def __init__(self):
        self.api_key = API_KEY
        encoded_password = quote_plus(DB_CONFIG['password'])
        self.uri = f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        self.db = SQLDatabase.from_uri(self.uri)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=API_KEY
        )
        self.sql_tool = Tool.from_function(
            func=self.CleanQueryTool(self.db).run,
            name="SQLQueryTool",
            description="""
                You are an assistant that writes SQL queries and retrieves data from a MySQL database.
                
                Database schema:
                
                Table: claims
                - id (INTEGER): Primary key
                - claim_number (TEXT): Claim number
                - bodytype (TEXT): Vehicle body type
                - client_id (INTEGER): Foreign key → clients.id
                - make (TEXT): Vehicle make
                - model (TEXT): Vehicle model
                
                Table: claim_images
                - id (INTEGER): Primary key
                - claim_id (INTEGER): Foreign key → claims.id
                - url (TEXT): URL of the image
                
                Assume fully qualified table names like client_51.claims and client_51.claim_images.
                Use SQL syntax compatible with MySQL.
            """
        )
        self.agent = initialize_agent(
            tools=[self.sql_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True
        )

    class CleanQueryTool(QuerySQLDataBaseTool):
        def __init__(self, db):
            super().__init__(db=db)
        
        def run(self, query: str) -> pd.DataFrame:
            query = query.strip().strip("```sql").strip("```").strip()
            try:
                result = pd.read_sql(query, self.db._engine)
                return result
            except Exception as e:
                return pd.DataFrame({"error": [f"Query failed: {str(e)}"]})

    def run_agent_query(self, query: str) -> pd.DataFrame:
        try:
            agent_output = self.agent({"input": query})
            intermediate_steps = agent_output.get("intermediate_steps", [])
            
            print("Intermediate Steps:", intermediate_steps)
            
            for action, output in intermediate_steps:
                if action.tool == "SQLQueryTool":
                    if isinstance(output, pd.DataFrame):
                        return output
                    sql_query = action.tool_input
                    print("Re-running SQL Query:", sql_query)
                    df = self.sql_tool.func(sql_query)
                    return df
            
            if isinstance(agent_output.get("output"), pd.DataFrame):
                return agent_output["output"]
            return pd.DataFrame({"error": ["No valid SQL query executed"]})
        
        except Exception as e:
            return pd.DataFrame({"error": [f"Agent execution failed: {str(e)}"]})

def fetch_and_save_data(query: str, output_csv: str) -> bool:
    """
    Fetch data from the database using the provided query and save to a CSV file.

    Args:
        query (str): The database query to execute.
        output_csv (str): Path to save the output CSV file.

    Returns:
        bool: True if successful, False otherwise.
    """
    db_fetcher = DBFetcher()
    df = db_fetcher.run_agent_query(query)
    
    if 'error' in df.columns:
        print(df['error'].iloc[0])
        return False

    
    print("\nClaims Data:")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    try:
        df.to_csv(output_csv, index=False)
        print(f"\nData saved to {output_csv}")
        return True
    except Exception as e:
        print(f"❌ Error saving CSV: {str(e)}")
        return False