# Semantic Proxy for Lambda Labs GPU with TGI Server (Docker)

This README outlines how to use the provided files to run a semantic caching proxy in conjunction with a Text Generation Inference (TGI) server on Lambda Labs GPUs, utilizing Docker for containerization. The "executable" referred to in this context is the TGI server, which we will run using Docker. The datasets used for this use case include `trivia.json`, `class.json`, and others in the repository.

## Overview

This setup implements a semantic caching layer (`semantic_proxy.py` and `semantic_cache.py`) that sits in front of your TGI server. The goal is to improve generation efficiency by identifying semantically similar past requests and reusing the cached KV (key-value) states. This can lead to faster response times and reduced computational cost for repetitive or similar trivia questions, classification prompts, and other types of queries defined in the datasets.

The **semantic KV cache** implemented here has the capability to be **global and shared between users**. This means that if one user makes a request that gets cached, and a subsequent request from a different user is semantically similar enough (based on the configured threshold and the `cache_strategy`), the cached KV states and generated output can be reused, potentially speeding up response times and reducing computational load across all users of the system.

The system also includes data collection capabilities (`data_collection/data_collector.py`) to benchmark the performance of the semantic cache on these types of datasets.

## File Descriptions

Here's a breakdown of the key files in the repository:

### `datasets/`
- Contains various JSON and CSV files (e.g., `trivia.json`, `class.json`, `UseCase*.json`, `UseCase*.csv`). These are the datasets used for this use case, containing trivia questions, classification examples, and other types of prompts designed to test and evaluate the semantic caching proxy.

### `evaluation/`
- `analyze_prompts.py`: Script to analyze the prompts within the datasets, possibly to understand their semantic relationships or to prepare them for benchmarking the cache.

### `gkv-cache/`
- Related to a different or older inference setup and are **not directly used** in the semantic proxy setup described here.
- Includes files like `inference/`, `class.json`, `client.py`, `customer.json`, `Dataset_client.py`, `Dockerfile`, `gpt.json`, `legal.json`, `requirements.txt`, `trivia.json`, `writing.json`.

### `tgi_inference/`
- **`tgi_intercept/`:**
  - `data_collection/`:
    - `data_collector.py`: Implements a `DataCollector` class responsible for recording and managing benchmark data related to cache hits, misses, and other relevant metrics.
    - `diagnostic.py`: Utility functions for debugging or monitoring the semantic cache's performance.
  - `Dockerfile`: Builds the semantic proxy image.
  - `requirements.txt`: Python dependencies for running the semantic proxy.
  - `semantic_cache.py`: Core logic for the semantic cache. Includes:
    - Classes for cache entries (`SemanticCacheEntry`) and cache management (`SemanticCacheManager`).
    - Sentence transformer model for prompt embeddings and cosine similarity calculations.
    - Support for global and per-user caches.
  - `semantic_proxy.py`: Main application file. 
    - FastAPI server with `/generate` endpoint.
    - Handles cache checking, request forwarding to the TGI server, and benchmark data collection.

### `tgi_stats/`
- Evaluation statistics and analysis.

### `docker-compose.yml`
- Defines and manages multi-container Docker applications.
- Orchestrates the TGI server container and semantic proxy container.

### `tgi_gkv_server`
- This is a subdirectory containing the modified source code from Hugging Face TGI. The following is an overview of the files that were modfied to support our project.
- `tgi-gkv-server/text-generation-inference/proto/generate.proto`: Protobuf definitions modified.
- `tgi-gkv-server/text-generation-inference/server/text_generation_server/models/causal_lm.py`: Added GKV alias operations as well as to pass previous key values based on prefix position. 
- `tgi-gkv-server/text-generation-inference/server/text_generation_server/models/model.py`: Modified root class model defintion to add GKV alias operations during generation requests.
- `tgi-gkv-server/text-generation-inference/server/text_generation_server/gkv_cache/gkv_cache.py`: Primary GKV Cache implementation to store key or value and corresponding previosuly computed attention values.
- Other parts of the source code to pass new parameters/variables as needed.
## Running the Semantic Proxy with Docker

Use the dockerfile provided to run the server on a server with NVIDIA GPU, install all necessary drivers as well.

# Interacting with the Semantic Proxy and Collecting Statistics
You can use curl to send requests to the semantic proxy and interact with its API. The cache_strategy parameter in your requests ("global_user", "global_only", "per_user", "no_cache") determines how the shared global semantic KV cache is utilized alongside any per-user caches.


```
echo "--- Starting Benchmark Data Collection ---"
curl -X POST "${PROXY_URL}/data/start"
echo -e "\n--- [Benchmark started] ---\n"
```

To send a generation request that utilize the global cache:
```
echo "Query 1: User 1 - Initial TGI Inference"
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "What are some quick recipes?",
    "parameters": { "max_new_tokens": 100 },
    "user_id": "user1",
    "cache_strategy": "global_only"
  }'
```

## Running the System
* The entire application is dockerized in order to make this process much simplier.
* There are hardware requirements in order for the application to run in a timely matter due to the high resource usage of inference. The model we use is Mistral-7B-Instruct-v0.1 (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) which recommends the following GPU.
   * Nvidia A100 40GB
 * However, during the completion of our project we utilized Lambda Labs cloud GPU instances. Typically we used either Nvidia A100 40GB or Nvidia A10 24GB depending on avaliability.
*  In addition, this tutorial assumes an Ubuntu-based environment.
### Step 1: Configure the Nvidia Runtime
Start by setting up the docker environment, as well as ensuring docker uses the Nvidia runtime toolkit. 
First install the nvidia container toolkit:
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
Next, configure docker to use the nvidia runtime and restart docker.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Test that the configuration has been correctly applied by running the following, which should display your GPU information.
```
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
### Step 2: Start the Docker Environment
First build the docker compose and images (this can take a very long time, so please be patient).
```
sudo docker compose build
```
This should build two container images for the semantic-proxy as well as the tgi_server. 
Start both containers, and include your HF token for access. 
```
sudo HUGGING_FACE_HUB_TOKEN=hf_XXXX docker compose up               
```
The services should now be running. In order to make a query, you can use a curl request to the API endpoint at `localhost:9000` or you can use the provided client-side sample code as well















