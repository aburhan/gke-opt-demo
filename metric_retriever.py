from google.cloud import bigquery

class MetricsCalculator:
    def __init__(self, project_id, dataset_table, cluster_name, location, container_name, namespace_name, controller_name):
        self.client = bigquery.Client(project=project_id)
        self.dataset_table = dataset_table
        self.cluster_name = cluster_name
        self.location = location
        self.container_name = container_name
        self.namespace_name = namespace_name
        self.controller_name = controller_name
    
        


    def query_metric(self, metric_info, days=30):
        metric_name = metric_info['name']
        aggregate_function = metric_info['aggregate_function']
        duration = metric_info.get('duration', days)
        
        filters = f"""
        AND JSON_VALUE(resource.labels.cluster_name) = '{self.cluster_name}'
        AND JSON_VALUE(resource.labels.location) = '{self.location}'
        AND JSON_VALUE(resource.labels.container_name) = '{self.container_name}'
        AND JSON_VALUE(resource.labels.namespace_name) = '{self.namespace_name}'
        AND JSON_VALUE(resource.labels.pod_name) LIKE "{self.controller_name}%"
        """

        current_data_filter = f"""
        {filters}
         ORDER BY
            start_time DESC
            LIMIT 1
        """

        # Construct and execute the query using the specified aggregate function and filters
        query = f"""
        SELECT
            '{metric_name}' as metric_name,
            {aggregate_function} as aggregated_value
        FROM
            `{self.dataset_table}`
        WHERE
            name = '{metric_name}'
            AND start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {duration} DAY)
            {filters}
        """
        print(query)
        query_job = self.client.query(query)
        results = query_job.result()

        for row in results:
            return {metric_name: row.aggregated_value}

    def query_metrics(self, metrics_info, days=3):
        results = {}
        for metric_info in metrics_info:
            metric_result = self.query_metric(metric_info, days)
            results.update(metric_result)
        return results

# Example usage
if __name__ == "__main__":
    project_id = 'gke-opt-demo'
    dataset_table = 'gke-opt-demo.1034414536999_Metrics._AllMetrics'
    cluster_name = 'online-boutique'
    location = 'us-central1-f'
    container_name = 'server'
    namespace_name = 'default'
    controller_name = 'adservice'


    metrics_info = [
            {"name": "kubernetes.io/container/cpu/core_usage_time", "aggregate_function": "AVG(value.double_value)", "duration": 30},

    ]
    current_metric_info = [
            {"name": "kubernetes.io/container/cpu/limit_cores", "aggregate_function": "value.double_value", "duration": 30},


    ]

    calculator = MetricsCalculator(project_id, dataset_table, cluster_name, location, container_name, namespace_name, controller_name)
    aggregated_metrics = calculator.query_metrics(current_metric_info )
    print(aggregated_metrics)
