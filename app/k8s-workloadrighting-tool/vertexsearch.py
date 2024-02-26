from langchain_community.retrievers import (
    GoogleVertexAIMultiTurnSearchRetriever,
    GoogleVertexAISearchRetriever,
)
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch
import yaml
import os
import json
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from google.cloud import bigquery

        
# TODO ADD COST
def bigquery_vector_search():
    BIGQUERY_REGION="US"
    BIGQUERY_COST_DATASET = "all_billing_data"  # @param {type: "string"}
    BIGQUERY_COST_TABLE = "gcp_billing_export_resource_v1_0175E8_5CAB7B_91ED1D"
    
    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )
    store = BigQueryVectorSearch(
        project_id=PROJECT_ID,
        dataset_name=BIGQUERY_COST_DATASET,
        table_name=BIGQUERY_COST_TABLE,
        location=BIGQUERY_REGION,
        embedding=embedding,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    docs = store.similarity_search(query)
    print(docs)

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 



# Get project id, cluster name and cluster location
PROJECT_ID = os.getenv('PROJECT_ID', None)  # Use None as default if not found
LOCATION = os.getenv('LOCATION', None)  # Use None as default if not found
CLUSTER_NAME = os.getenv('CLUSTER_NAME', None)  # Use None as default if not found
    
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")
print(f"CLUSTER_NAME: {CLUSTER_NAME}")
    
# Read the K8 YAML Manifest
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

def calculate_qos(requests, limits):
    if requests == limits and "cpu" in requests and "memory" in requests:
        return "Guaranteed"
    elif "cpu" in requests or "memory" in requests:
        return "Burstable"
    else:
        return "BestEffort"

def extract_details_and_qos_from_yaml(parsed_yaml):
    try:
        kind = parsed_yaml.get("kind", "Kind not specified")
        metadata_name = parsed_yaml.get("metadata", {}).get("name", "Name not specified")
        
        details = {
            "kind": kind,
            "metadata_name": metadata_name,
            "containers": {}
        }
        
        containers = parsed_yaml.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        
        for container in containers:
            name = container.get("name", "Unnamed container")
            image = container.get("image", {})
            resources = container.get("resources", {})
            requests = resources.get("requests", {})
            limits = resources.get("limits", {})
            
            cpu_request = requests.get("cpu", "Not specified")
            memory_request = requests.get("memory", "Not specified")
            cpu_limit = limits.get("cpu", "Not specified")
            memory_limit = limits.get("memory", "Not specified")
            
            qos_class = calculate_qos(requests, limits)
            
            details["containers"][name] = {
                "image": image,
                "cpu_request": cpu_request,
                "memory_request": memory_request,
                "cpu_limit": cpu_limit,
                "memory_limit": memory_limit,
                "QoS": qos_class
            }
        
        return details
    except yaml.YAMLError as e:
        return f"Error parsing YAML: {str(e)}"


def vertexai_evaluate_generate_summary(k8_details):
    STATE_OF_KUBERNETES_COST_OPT_DATA_STORE_ID = "sunday-funday_1708890177344" 
    k8_details_str = json.dumps(k8_details)
    vertexai.init(project="gke-opt-demo", location="us-central1")
    model = GenerativeModel("gemini-1.0-pro-001")
    instruction = f"""
    You are a Kubernetes expert who specializes in workload optimization, given the following {{k8object}} which represents a Kubernetes object provide an analysis and return the output as a json in the following format

    example format: 
    {{
      "metadata": {{
        "name": "loadgenerator",
        "kind": "Deployment"
      }},
      "analysis": {{
        "cpu_request": "Not specified",
        "memory_request": "Not specified",
        "cpu_limit": "Not specified",
        "memory_limit": "Not specified",
        "QoS": "BestEffort",
        "recommendations": [
          {{
            "type": "resources",
            "message": "Consider specifying CPU and memory requests and limits to ensure consistent resource allocation and prevent performance issues."
          }},
          {{
            "type": "QoS",
            "message": "Consider using a higher QoS class, such as Guaranteed, to ensure consistent performance and resource allocation."
          }}
        ]
      }}
    }}

    k8object:{k8_details_str}"""
    responses = model.generate_content(instruction,
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.2,
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

    return(responses)


def build_query_workload_recommendation(project_id, location, cluster_name, namespace_name, container_name, controller_type, controller_name):
    
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
 
    return query

def build_image_and_node_metadata( location, cluster_name, namespace_name, container_name, controller_type, controller_name):
    query = f"""
    SELECT 
    container.update_time,
    JSON_VALUE(container.system_labels.top_level_controller_type) AS controller_type,
    JSON_VALUE(container.system_labels.container_image) AS container_image,
    JSON_VALUE(container.system_labels.node_name) AS node_name,
    container.user_labels,
    node.user_labels AS node_details
    FROM 
    `gke-opt-demo.1034414536999_Metrics._Metadata` AS container
    JOIN
    `gke-opt-demo.1034414536999_Metrics._Metadata` AS node
    ON  JSON_VALUE(container.system_labels.node_name) = JSON_VALUE(node.resource.labels.node_name)
    WHERE container.resource.type = "k8s_container" 
    AND JSON_VALUE(container.resource.labels.cluster_name) = '{cluster_name}'
    AND JSON_VALUE(container.resource.labels.location) = '{location}'
    AND JSON_VALUE(container.resource.labels.namespace_name) = '{namespace_name}'
    AND JSON_VALUE(container.system_labels.top_level_controller_name) = '{controller_name}'
    AND JSON_VALUE(container.system_labels.top_level_controller_type) = '{controller_type}'
    AND JSON_VALUE(container.resource.labels.container_name) = '{container_name}'
    AND node.resource.type= "k8s_node"
    ORDER BY container.update_time DESC
    LIMIT 1;
    """
    return query

def query_bigquery(query):
    # Initialize a BigQuery client
    client = bigquery.Client()
    
    # Initialize a dictionary to hold the results
    results_dict = {}
    
    # Execute the query
    results = client.query(query)

    for row in results:
        for column, value in row.items():
            if column not in results_dict:
                results_dict[column] = value
            else:
                results_dict[column]=value
    
    return results_dict

# Vertex AI Datastore Search
def vertex_ai_search(project_id, location_id, data_store_id, query):
    output=[]
    retriever = GoogleVertexAISearchRetriever(
        project_id=project_id,
        location_id=location_id,
        data_store_id=data_store_id,
        max_documents=1,
        get_extractive_answers=False
    )

    result = retriever.get_relevant_documents(query)
    for doc in result:
        output.append(doc)
    return output

def get_node_recommendations(project_id, bq_results_node_metadata, bq_result_workload_recommedation):
    location_id = "global"
    print("##############")
    print(bq_results_node_metadata['container_image'])
    choose_compute_data_store= "gke-opt-demo-choose-comput_1708926376105"
    # Extracting the container image
    #container_image = bq_results_node_metadata['container_image']
    #machine_family = bq_results_node_metadata['cloud.google.com/machine-family']
    #instance_type = bq_results_node_metadata['node.kubernetes.io/instance-type']
    #cpu_mcore_recommendation_limit = bq_result_workload_recommedation['memory_limit_recommendation']
    #memory_MiB_recommendation_limit= bq_result_workload_recommedation['cpu_limit_recommendation']
    
    query = f"Given the following information about my workload{bq_results_node_metadata}, {bq_result_workload_recommedation}, provide a recommendations on what compute instance to use for cost savings?"

    return (vertex_ai_search(project_id, location_id, choose_compute_data_store, query))
    
if __name__ == "__main__":

    # Extract fields from yaml
    yaml_content = (read_k8s_workload_file('loadgenerator.yaml')) 
    k8s_details = (extract_details_and_qos_from_yaml(yaml_content))

    # Vertex AI RAG - Use messaging from State Of Kubernetes report 
    vertexai_evaluate_generate_summary(k8s_details)
    
    # Query BigQuery for workload recommenations
    
    for container in k8s_details['containers']:
        bq_workload_recommendation_query = build_query_workload_recommendation(PROJECT_ID, LOCATION, CLUSTER_NAME, k8s_details.get('namespace', 'default'), container, k8s_details['kind'],k8s_details['metadata_name'])  
        bq_result_workload_recommedation = query_bigquery(bq_workload_recommendation_query)
    
    # Query BigQuery for node, and image information
        bq_node_query = (build_image_and_node_metadata( LOCATION, CLUSTER_NAME, k8s_details.get('namespace', 'default'), container, k8s_details['kind'],k8s_details['metadata_name']))
        bq_results_node_metadata = query_bigquery(bq_node_query )
        
    # Vertex AI RAG - Using choose compute, node, image, cpu and memory usgage information recommend compute
        get_node_recommendations(PROJECT_ID, bq_results_node_metadata, bq_result_workload_recommedation)
         
    
    # Return summary