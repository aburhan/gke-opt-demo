# GEMINI imports
import os
from google.cloud import aiplatform as vertex_ai
# CHATGPT imports
from google.cloud import aiplatform
from google.oauth2 import service_account
#
def console_search_documents(project, location, query_string, filter_string, language_codes):
    """Search for documents based on a given query and filter string"""

    # Import the Vertex Search client library
    from google.cloud import vertexsearch

    # Create a client
    client = vertexsearch.VertexSearchServiceClient()

    # Construct a document search configuration
    document_search_config = vertexsearch.SearchDocumentsRequest.DocumentSearchConfig(
        enable_exact_phrase_matches=True,
        boost_spec(
            boost=10,
            query="important",
        ),
        boost_spec(
            boost=5,
            query="new",
        ))

    # Construct a query object
    query = vertexsearch.SearchDocumentsRequest.Query(
        query_string=query_string,
        document_search_config=document_search_config,
    )

    # Construct a search request
    request = vertexsearch.SearchDocumentsRequest(
        parent=f"projects/{project}/locations/{location}",
        query=query,
        # Filter results based on language
        filter=filter_string,
        # Set the language codes for search ranking
        language_codes=language_codes,
    )

    # Execute the search request and get the response
    response = client.search_documents(request=request)

    # Print the search results
    for result in response.results:
        print("Document title: {}".format(result.document.title))


def gemini_search_documents():
    # Project ID and location (region) where your Vertex AI Search datastore is located
    project_id = "YOUR_PROJECT_ID"
    location = "YOUR_LOCATION" 
    endpoint_id = 'YOUR_ENDPOINT_ID' # The Endpoint ID of your Vertex AI Search deployment

    # Initialize the Vertex AI client
    vertex_ai.init(project=project_id, location=location)

    # Construct the query you want to send
    query = "Your search term here" 

    # Define parameters for the Search request 
    search_request = vertex_ai.types.SearchDocumentsRequest(
        endpoint=vertex_ai.Endpoint(endpoint_id).resource_name,
        query=query,
        page_size=5   # Adjust this to control the number of results returned
    )

    # Send the search query to Vertex AI Search
    response = vertex_ai.EndpointService().SearchDocuments(search_request)

    # Print the search results
    print("Search Results:")
    for document in response.documents:
        print(f"Document ID: {document.document.name}")
        print(f"Document Content: {document.document.content}")
        print("-" * 30) 

def chatgpt_search_documents():
    # Replace the following variables with your project's information
    project_id = 'your-project-id'
    location = 'your-project-location'  # e.g., us-central1
    credentials_path = 'path/to/your/service-account-file.json'

    # Set the credentials and initialize the Vertex AI client
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    aiplatform.init(project=project_id, location=location, credentials=credentials)

    def search_datasets(query: str):
        client = aiplatform.gapic.DatasetServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
        
        # Construct the parent resource path
        location_path = client.common_location_path(project_id, location)
        
        # Execute the search request
        response = client.list_datasets(parent=location_path, filter=query)
        
        # Print the search results
        print("Search results:")
        for dataset in response:
            print(f"Dataset name: {dataset.name}, Display name: {dataset.display_name}, Labels: {dataset.labels}")

if __name__ == "__main__":
    # Example search query: search for all datasets
    search_query = ""
    search_datasets(search_query)