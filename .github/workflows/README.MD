
## Infrastructure setup requirements
- Create monitor project
- Enable billing exports
- Enable metrics exports
- Create a storage bucket for workload docs
    - How to select compute
    - State of kubernetes cost optimization
    - Best practices
- Create logging sinks for evicted pods
- Create BigQuery Views
- Create Bigquery embedding on views
- Create service accounts with access to services with projects in the monitoring scope 
    - Vertex AI
    - Asset Inventory
    - Active Assist
- Load the workload docs to vertex AI search
- Create an vertex AI app and set docs as a retriever

## Github Action
- Set flags 
    - IN_PLACE_UPDATE=false
    - AUTOPILOT_CLUSTER=false
    - HPA_ENABLED=false
    - SPOT_INSTANCE=false
    - POD_DISRUPTION_BUDGETS=false
- Call Asset Inventory to check
    - If cluster is autopilot set AUTOPILOT=true
    - Is node auto-provision enabled?

- Identify which files in the pr are k8 manifest object 
    - Using vertex ai to return a list of files which are k8s workloads 
- Using asset inventory determine if the deployment has HPA enabled set flag HPA_ENABLED= true
- Query BigQuery and locally store recommendations
- Use vertex AI app pass in the yaml and the recommendations(including node information) for each.
    - Using the image and the yaml to determine what kind of workloads is running (web app, database etc) 
        - Return - Is the node instance type a good fit?
    - If IN_PLACE_UPDATE=true
        - Update the manifest yml
    - Build note about workload rightsizing and node information fail pipeline
