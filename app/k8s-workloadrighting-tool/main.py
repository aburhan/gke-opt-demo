import os
import json
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import yaml
import sys



def get_env_variable(var_name):
    """Fetches an environment variable and raises an error if not found."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required but not set.")
    return value

def query_bigquery(project_id, location, cluster_name, namespace_name, container_name):
    """Constructs the SQL query string to fetch the most recent record using environment variables."""
    
    
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

def analyze_workload(file_path):
    with open(file_path, 'r') as file:
        workload_config = yaml.safe_load(file)
    
    # Simplified parsing logic; actual implementation will vary based on the YAML structure
    project_id = workload_config['project_id']
    location = workload_config['location']
    cluster_name = workload_config['cluster_name']
    namespace_name = workload_config['namespace_name']
    container_name = workload_config['container_name']

    recommendations = query_bigquery(project_id, location, cluster_name, namespace_name, container_name)
    if not recommendations:
        print("No recommendations found.")
        sys.exit(1)
    
    recommendation = recommendations[0]  # Assuming one match for simplicity
    response = {
        'container_name': container_name,
        'cpu_recommendation': recommendation['cpu_requested_recommendation'],
        'memory_recommendation': recommendation['memory_requested_recommendation'],
        'over_or_under_provisioned': 'under' if (workload_config['cpu_requested'] < recommendation['cpu_requested_recommendation'] or workload_config['memory_requested'] < recommendation['memory_requested_recommendation']) else 'over',
        # Simplify instance group recommendation logic
        'recommended_instance_group': 'gcp-instance-group-based-on-usage'  # Placeholder logic
    }
    
    # Determine if pipeline should fail
    if response['over_or_under_provisioned'] == 'under':
        response['error'] = 'Resource requests are less than the recommendations. Failing the pipeline.'
        print(json.dumps(response))
        sys.exit(1)
    
    print(json.dumps(response))

def main():
    """Main function to orchestrate the workflow."""
    try:
        query = construct_query()
        execute_query(query)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage: python script.py path/to/your/workload.yaml
    
    project_id = get_env_variable('PROJECT_ID')
    location = get_env_variable('LOCATION')
    cluster_name = get_env_variable('CLUSTER_NAME')
    namespace = get_env_variable('NAMESPACE')
    controller_name = get_env_variable('CONTROLLER_NAME')
    controller_type = get_env_variable('CONTROLLER_TYPE')
    container_name = get_env_variable('CONTAINER_NAME')

    print(construct_query())
    


