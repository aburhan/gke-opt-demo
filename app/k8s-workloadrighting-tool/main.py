import os
import json
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from metrics_retriever import MetricsRetriever

def get_env_variable(var_name):
    """Fetches an environment variable and raises an error if not found."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required but not set.")
    return value

def construct_query():
    """Constructs the SQL query string to fetch the most recent record using environment variables."""
    project_id = get_env_variable('PROJECT_ID')
    location = get_env_variable('LOCATION')
    cluster_name = get_env_variable('CLUSTER_NAME')
    namespace = get_env_variable('NAMESPACE')
    controller_name = get_env_variable('CONTROLLER_NAME')
    controller_type = get_env_variable('CONTROLLER_TYPE')
    container_name = get_env_variable('CONTAINER_NAME')
    
    return f"""
    WITH RankedRecords AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY run_date DESC) AS rn
        FROM `gke-opt-demo.gke_metrics_dataset.workload_recommendations`
        WHERE 
            project_id = '{project_id}' AND
            location = '{location}' AND
            cluster_name = '{cluster_name}' AND
            namespace_name = '{namespace}' AND
            controller_name = '{controller_name}' AND
            controller_type = '{controller_type}' AND
            container_name = '{container_name}'
    )
    SELECT * FROM RankedRecords WHERE rn = 1
    """

def execute_query(query):
    """Executes the provided SQL query, converts the results to JSON, and prints the JSON string."""
    try:
        # Initialize a BigQuery client
        client = bigquery.Client()

        # Execute the query
        query_job = client.query(query)

        # Initialize an empty list to hold query results
        results = []

        # Iterate over the query results and append each row to the results list as a dict
        for row in query_job:
            result_dict = dict(row)
            results.append(result_dict)

        # Convert the list of dicts to a JSON string
        json_results = json.dumps(results, default=str)  # default=str to handle non-serializable types like datetime

        # Print the JSON string
        print(json_results)

    except GoogleAPIError as e:
        print(f"BigQuery error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """Main function to orchestrate the workflow."""
    try:
        query = construct_query()
        execute_query(query)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
