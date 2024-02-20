import os
import json
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import sys
import yaml
from google.cloud import asset_v1


def get_env_variable(var_name):
    """Fetches an environment variable and raises an error if not found."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required but not set.")
    return value

def read_k8s_workload_file(file_path):
    """
    Reads a Kubernetes workload file in YAML format and returns the data as a Python dictionary.

    :param file_path: Path to the YAML file containing the Kubernetes workload definition.
    :return: A dictionary representation of the Kubernetes workload.
    """
    with open(file_path, 'r') as file:
        # Load the YAML file content into a Python dictionary
        workload_data = yaml.safe_load(file)
    
    return workload_data
    
def query_bigquery(project_id, location, cluster_name, namespace_name, container_name):
    
    # Initialize a BigQuery client
    client = bigquery.Client()
    
    # Initialize a dictionary to hold the results
    results_dict = {}
    
    """Constructs the SQL query string to fetch the most recent record using environment variables."""
    query = f"""
    WITH RankedRecords AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY run_date DESC) AS rn
        FROM `gke-opt-demo.gke_metrics_dataset.workload_recommendations`
        WHERE 
            project_id = '{project_id}' AND
            location = '{location}' AND
            cluster_name = '{cluster_name}' AND
            namespace_name = '{namespace_name}' AND
            controller_name = '{controller_name}' AND
            controller_type = '{controller_type}' AND
            container_name = '{container_name}'
    )
    SELECT * FROM RankedRecords WHERE rn = 1 LIMIT 1
    """
 
    # Execute the query
    results = client.query(query)

    for row in results:
        for column, value in row.items():
            if column not in results_dict:
                results_dict[column] = value
            else:
                results_dict[column]=value
    
    return results_dict

def asset_inventory():
    from google.cloud import asset_v1
    from google.cloud.asset_v1.types import asset_service

    # Initialize the Asset Service Client
    client = asset_v1.AssetServiceClient()

    # Define the scope, query, asset types, and order by parameters
    scope = "projects/1034414536999"
    query = "adservice"
    asset_types = ["apps.k8s.io/Deployment"]
    order_by = "displayName,createTime"

    # Prepare the request
    request = asset_service.SearchAllResourcesRequest(
        scope=scope,
        query=query,
        asset_types=asset_types,
        order_by=order_by
    )

    # Execute the search and process the response
    for resource in client.search_all_resources(request=request):
        print(resource)

def asset_inventory_full():
    # TODO project_id = 'Your Google Cloud Project ID'
    # TODO asset_types = 'Your asset type list, e.g.,
    # ["storage.googleapis.com/Bucket","bigquery.googleapis.com/Table"]'
    # TODO page_size = 'Num of assets in one page, which must be between 1 and
    # 1000 (both inclusively)'
    # TODO content_type ="Content type to list"
    project_resource = f"projects/gke-opt-demo"
    client = asset_v1.AssetServiceClient()

    # Call ListAssets v1 to list assets.
    response = client.list_assets(
        request={
            "parent": project_resource,
            "read_time": None,
            "asset_types": ["apps.k8s.io/Deployment"],
            "content_type": asset_v1.ContentType.RESOURCE,
            "page_size": 1,
        }
    )

    for asset in response:
        print(asset)


def generate():
    vertexai.init(project="gke-opt-demo", location="us-central1")
    model = GenerativeModel("gemini-pro")
    responses = model.generate_content(
"""Act as a Kubernetes expert, giving  the following {workload} structure which represents a workload running in GKE determine if the workload is over or under-provisioned and suggest a compute instance based on the image. then update the yaml with the workload resource recommendations
    workload: {}

    yaml: {}
    """,
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.9,
        "top_p": 1
    },
    safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    stream=True,
  )
    for response in responses:
        print(response.text, end="")


    generate()

if __name__ == "__main__":
    project_id = get_env_variable('PROJECT_ID')
    location = get_env_variable('LOCATION')
    cluster_name = get_env_variable('CLUSTER_NAME')
    namespace_name = get_env_variable('NAMESPACE')
    controller_name = get_env_variable('CONTROLLER_NAME')
    controller_type = get_env_variable('CONTROLLER_TYPE')
    container_name = get_env_variable('CONTAINER_NAME')

    workload_info = (query_bigquery(project_id, location, cluster_name, namespace_name, container_name))
    

    print(workload_info)



