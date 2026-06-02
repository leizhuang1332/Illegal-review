from prometheus_client import Counter, Histogram, Gauge

input_requests_total = Counter(
    "input_requests_total", "Input layer request count",
    ["type", "status"],
)
input_request_duration_seconds = Histogram(
    "input_request_duration_seconds", "Processing duration in seconds",
    ["type"],
)
input_file_size_bytes = Histogram(
    "input_file_size_bytes", "File size distribution",
    ["type"],
)
input_upload_chunk_total = Counter(
    "input_upload_chunk_total", "Chunked upload chunk count",
    ["status"],
)
input_temp_dir_usage_ratio = Gauge(
    "input_temp_dir_usage_ratio", "Temp dir disk usage",
)
input_active_tasks = Gauge(
    "input_active_tasks", "Active task count",
    ["type"],
)
input_queue_depth = Gauge(
    "input_queue_depth", "Queue depth",
    ["queue"],
)
input_kafka_fallback_count = Gauge(
    "input_kafka_fallback_count", "Kafka fallback count",
)
