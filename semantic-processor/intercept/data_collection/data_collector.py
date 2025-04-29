import time
import json
import csv
import os
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import psutil 

OUTPUT_DIR = "./statistics"

class DataCollector:
    """
    Utility class to collect statitics and save them to a CSV file for evaluation.
    DataCollector is activated when the user sends the start request.
    """
    def __init__(self, output_dir: str = OUTPUT_DIR):
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.csv_file = os.path.join(self.output_dir, f"{self.batch_id}.csv")

        self.csv_headers = [
            "request_id",
            "timestamp",
            "prompt",
            "user_id",
            "cache_strategy",
            "cache_hit",
            "cache_source",
            "similarity_score",
            "processing_time_ms",
            "embedding_time_ms",
            "similarity_search_time_ms",
            "response_length",
            "cumulative_requests",
            "cumulative_hits",
            "cumulative_misses",
            "cumulative_hit_rate",
            "memory_usage_mb"
        ]

        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)


        # Basics
        self.start_time = time.time()
        self.is_active = True
        self.tracking_started = False
        
        self.requests = []
        self.hits = []
        self.misses = []
        
        self.hit_times = []
        self.miss_times = []
        self.all_times = []
        
        self.similarity_scores = []
        
        # Memory
        self.memory_usage = []
        self.initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


    def start_tracking(self):
        """
        Start tracking the statistics.
        """
        self.start_time = time.time()
        self.tracking_started = True
        self.is_active = True

        # Reset all statistics
        self.requests = []
        self.hits = []
        self.misses = []
        self.hit_times = []
        self.miss_times = []
        self.all_times = []
        self.similarity_scores = []
        self.memory_usage = []
        print(f"DATA_COLLECTOR - started tracking at {self.start_time} with BATCH_ID {self.batch_id}")
        return {"status": "Running", "benchmark_id": self.batch_id}

    def stop_tracking(self):
        """
        Stop tracking the statistics and generate summary.
        """
        self.is_active = False
        total_time = time.time() - self.start_time

        print(f"Collection stopped. Duration: {total_time:.2f} seconds")
        
        return {
            "status": "Stopped",
            "duration_seconds": total_time,
            "csv_file": self.csv_file,
        }

    def record_query(self, 
                    request_id: str,
                    prompt: str,
                    user_id: str,
                    cache_strategy: str,
                    cache_hit: bool,
                    cache_source: Optional[str],
                    similarity_score: Optional[float],
                    start_time: float,
                    end_time: float,
                    embedding_start_time: Optional[float] = None,
                    embedding_end_time: Optional[float] = None,
                    similarity_search_start_time: Optional[float] = None,
                    similarity_search_end_time: Optional[float] = None,
                    response_length: Optional[int] = None):
        """
        Record single query statistics.

        Args:
            request_id (str): _description_
            prompt (str): _description_
            user_id (str): _description_
            cache_strategy (str): _description_
            cache_hit (bool): _description_
            cache_source (Optional[str]): _description_
            similarity_score (Optional[float]): _description_
            start_time (float): _description_
            end_time (float): _description_
            embedding_start_time (Optional[float], optional): _description_. Defaults to None.
            embedding_end_time (Optional[float], optional): _description_. Defaults to None.
            similarity_search_start_time (Optional[float], optional): _description_. Defaults to None.
            similarity_search_end_time (Optional[float], optional): _description_. Defaults to None.
            response_length (Optional[int], optional): _description_. Defaults to None.
        """
        if not self.tracking_started or not self.is_active:
            print("WARN - Benchmark tracking not active. Start tracking first.")
            return
        
        # Processing Time
        processing_time_ms = (end_time - start_time) * 1000
        
        # EMbedding TIme
        embedding_time_ms = None
        if embedding_start_time and embedding_end_time:
            embedding_time_ms = (embedding_end_time - embedding_start_time) * 1000
        
        # Similarity Search Time
        similarity_search_time_ms = None
        if similarity_search_start_time and similarity_search_end_time:
            similarity_search_time_ms = (similarity_search_end_time - similarity_search_start_time) * 1000
        
        # Store request data
        request_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "user_id": user_id,
            "cache_strategy": cache_strategy,
            "cache_hit": cache_hit,
            "cache_source": cache_source if cache_hit else None,
            "similarity_score": similarity_score if cache_hit else None,
            "processing_time_ms": processing_time_ms,
            "embedding_time_ms": embedding_time_ms,
            "similarity_search_time_ms": similarity_search_time_ms,
            "response_length": response_length,
            "cumulative_requests": len(self.requests),
            "cumulative_hits": len(self.hits),
            "cumulative_misses": len(self.misses),
            "hit_rate_percent": (len(self.hits) / len(self.requests)) * 100 if len(self.requests) > 0 else 0
        }
        
        self.requests.append(request_data)
        self.all_times.append(processing_time_ms)
        
        # Cache Hit / Miss Rates
        if cache_hit:
            self.hits.append(request_data)
            self.hit_times.append(processing_time_ms)
            if similarity_score is not None:
                self.similarity_scores.append(float(similarity_score))
        else:
            self.misses.append(request_data)
            self.miss_times.append(processing_time_ms)
        
        # Memory Usage
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        memory_usage_mb = current_memory - self.initial_memory
        self.memory_usage.append(memory_usage_mb)
        
        
        row_data = {
            **request_data,
            "memory_usage_mb": memory_usage_mb
        }
        
        self._append_row(row_data)
        
        return row_data

    def _append_row(self, row_data: Dict[str, Any]):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [str(row_data.get(header, "")) for header in self.csv_headers]
            writer.writerow(row)

    