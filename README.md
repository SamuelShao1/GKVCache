# Semantic Proxy for Lambda Labs GPU with TGI Server (Docker)

This README outlines how to use the provided files to run a semantic caching proxy in conjunction with a Text Generation Inference (TGI) server on Lambda Labs GPUs, utilizing Docker for containerization. The "executable" referred to in this context is the TGI server, which we will run using Docker. The datasets used for this use case include `trivia.json`, `class.json`, and others in the repository.

## Overview

This setup implements a semantic caching layer (`semantic_proxy.py` and `semantic_cache.py`) that sits in front of your TGI server. The goal is to improve generation efficiency by identifying semantically similar past requests and reusing the cached KV (key-value) states. This can lead to faster response times and reduced computational cost for repetitive or similar trivia questions, classification prompts, and other types of queries defined in the datasets.

The system also includes data collection capabilities (`data_collection/data_collector.py`) to benchmark the performance of the semantic cache on these types of datasets.

## File Descriptions

Here's a breakdown of the key files in the repository:

**`datasets/`:**
* Contains various JSON and CSV files (e.g., `trivia.json`, `class.json`, `UseCase*.json`, `UseCase*.csv`). These are the datasets used for this use case, containing trivia questions, classification examples, and other types of prompts designed to test and evaluate the semantic caching proxy.

**`evaluation/`:**
* `analyze_prompts.py`: This script is used to analyze the prompts within the datasets, possibly to understand their semantic relationships or to prepare them for benchmarking the cache.

**`gkv-cache/`:**
* This directory and its contents (`inference/`, `class.json`, `client.py`, `customer.json`, `Dataset_client.py`, `Dockerfile`, `gpt.json`, `legal.json`, `requirements.txt`, `trivia.json`, `writing.json`) are related to a different or older inference setup and are **not directly used** in the semantic proxy setup described here. Note that `class.json` and `trivia.json` here are distinct from the dataset files in the `datasets/` directory.

**`tgi_inference/`:**
* `tgi_intercept/`:
    * `data_collection/`:
        * `data_collector.py`: This script implements a `DataCollector` class responsible for recording and managing benchmark data related to cache hits, misses, and other relevant metrics when processing the trivia, classification, and other datasets.
        * `diagnostic.py`: Contains utility functions for debugging or monitoring the semantic cache's performance on the specific use case datasets.
    * `Dockerfile`: A Dockerfile specifically for building the semantic proxy image, optimized for handling requests related to trivia, classification, and other tasks defined in the datasets.
    * `requirements.txt`: Lists the Python dependencies required to run the semantic proxy, ensuring compatibility with processing the data from `trivia.json`, `class.json`, etc.
    * `semantic_cache.py`: This file contains the core logic for the semantic cache. It includes classes for storing cache entries (`SemanticCacheEntry`) and managing the cache (`SemanticCacheManager`). It uses a sentence transformer model to generate embeddings for prompts (trivia questions, classification inputs, etc.) and calculates cosine similarity to find similar past requests from the datasets.
    * `semantic_proxy.py`: This is the main application file. It uses FastAPI to create an API endpoint (`/generate`) that receives generation requests (likely based on the trivia, classification, and other data). It interacts with the `SemanticCacheManager` to check for cache hits and forwards requests to the TGI server if needed. It also includes endpoints for managing the data collection process for benchmarking the cache's effectiveness on the specific use case.

**`tgi_stats/`:**
* Contains scripts or configurations for monitoring the TGI server itself while it processes requests related to the trivia, classification, and other tasks.

**`docker-compose.yml`:**
* This file defines and manages multi-container Docker applications. It will likely be used to orchestrate the TGI server container and the semantic proxy container, optimized for handling the trivia, classification, and other use cases.

## Running the Semantic Proxy with Docker

Follow the instructions in the previous README output to build and run the Docker containers for the semantic proxy and the TGI server. Ensure your `docker-compose.yml` is configured correctly to link the services.

## Interacting with the Semantic Proxy

Use a client (implemented in `client.py`) to send requests to the semantic proxy's API endpoint (e.g., `http://your_lambda_labs_ip:8000/generate`). Your requests can be based on the data in `trivia.json`, `class.json`, and other datasets to test the caching behavior for these specific use cases. The `user_id` and `cache_strategy` parameters in your requests will control how the semantic cache is utilized for these trivia and classification tasks. You can also use the `/data` endpoints to benchmark the performance of the cache when processing these specific datasets.
