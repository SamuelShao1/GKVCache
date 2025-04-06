import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
import time
import uuid
import logging
from collections import defaultdict

# Using the huggingface TGI
from text_generation import Client, AsyncClient  

# Sementaic Embedding Model
from sentence_transformers import SentenceTransformer

# Diagonistic Script
import diagnostic

# Access tokenzizer model 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


EMBEDDINGG_MODEL = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.8

class SemanticCacheEntry:
    """
    A class representing a single entry in the semantic cache.
    The semantic cache stores the input text (prompt), its prompt embedding, and the corresponding KV cache reference.
    """
    def __init__(self, prompt: str, embedding: np.ndarray, req_id: str, prefix_pos, timestamp: float):
        """
        Initialize a SemanticCacheEntry instance.
        Args:
            prompt (str): Text prompt.
            embedding (np.ndarray): Vector representation of the prompt, using a semantic embedding model, MiniLM)
            req_id (int): req id assigned by vLLM to the original generation to reference cached blocks.
            block_tables (List[List[int]]): 2D list of block IDs used nu the request for each layer's KV cache. Allowing the reuse an existing KV cache for a new and similar prompt.
            timestamp (float): Timestamp of when the entry was created, for LRU eviction policy.
        """
        self.prompt = prompt
        self.embedding = embedding
        self.req_id = req_id
        self.prefix_pos = prefix_pos
        self.timestamp = timestamp
        self.last_accessed = self.timestamp
        self.access_count = 0

    def access(self):
        """
        Update the last accessed time and increment the access count.
        """
        self.last_accessed = time.time()
        self.access_count += 1

class SemanticCacheManager:
    """
    Manages the semantic cache for storing and retrieving KV cache entries based on semantic similarity computations.
    """
    def __init__(self, embedding_function=str, similarity_threshold=0.8, max_cache_entries: int = 1000):
        """
        Initialize the SemanticCacheManager.

        Args:
            embedding_function (str, optional): Embedding function model name string. Convert text to embeddings. Defaults to None.
            similarity_threshold (float, optional): Cosine similarity. Defaults to 0.8.
            max_cache_entries (int, optional): Max number of cache entries. Defaults to 1000.
        """
        if embedding_function is None:
            raise ValueError("Embedding function must be provided.")

        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.max_cache_entries = max_cache_entries

        # Maps cache_id to SemanticCacheEntry
        self.cache: Dict[str, SemanticCacheEntry] = {}

        # Maps rid to cid for fast lookups
        self.rid_cid: Dict[int, str] = {}

        # Set of rids that shouldn't be cached, such as if they use cache KV states themselves
        self.no_cache_rids: Set[int] = set()

        # Cache statistics
        self.stats = {
            "total_requests": 0,
            "semantic_hits": 0,
            "misses": 0,
            "cache_entries": 0
        }

        print(f"Initialized SemanticKVCacheManager with similarity threshold {similarity_threshold}")


    def find_similar_prompts(self, prompt: str) -> Tuple[Optional[str], float]:
        """
        Use cosine similarity to compute similarity score, and compare it with cache entries above the threshold.


        Args:
            prompt (str): Query prompt, textual.

        Returns:
            Tuple[Optional[str], float]: Representing (cache_id of most similar prompt, cosine similarity score).
            Returns (None, 0.0) if no similar prompt is found, or no cache entries exceed the threshold value.
        """

        if not self.cache:
            # Empty cache, no similar prompts
            return None, 0.0
        
        query_embedding = self.embedding_function(prompt)

        print(f"Query embedding: {query_embedding}")
        # Find the most similar prompt in the cache.
        best_cache_id = None
        best_similarity = 0.0

        print(f"Cache Items: {self.cache.items()}")
        for cache_id, entry in self.cache.items():

            cached_embedding = entry.embedding
            
            cos_sim = self._cos_sim(query_embedding, cached_embedding)
            
            print(f"Comparing with cache ID {cache_id} with embedding {cached_embedding}, similarity: {cos_sim}")

            if cos_sim > best_similarity:
                best_similarity = cos_sim
                best_cache_id = cache_id
                print(f"New best cache ID: {best_cache_id}, similarity: {best_similarity}")

        if best_similarity > self.similarity_threshold:
            return best_cache_id, best_similarity
        
        return None, best_similarity
    

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space.
        It is defined as the cosine of the angle between them, which is equivalent to the dot product of the normalized vectors.

        cos_sim(vector_a, vector_b) = (vector_a . vector_b) / (||vector_a|| * ||vector_b||)

        Args:
            a (np.ndarray): vector_a
            b (np.ndarray): vector_b

        Returns:
            float: cosine similarity score between vector_a and vector_b.
        """

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # If either vector has zero magnitude
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def on_req_start(self, prompt: str, req_id: str) -> Tuple[bool, Optional[str], float]:
        """
        When a new request(prompt) is started, check if there are similar prompts and return the cache hit if found.

        Args:
            prompt (str): Input prompt.
            req_id (str): Current req id.
        Returns:
            Tuple[bool, Optional[str], float]: Representation of (cache hit status, cache_id, similarity score).
            Returns (False, None, 0.0) if no similar prompt is found.
        """
        print(f"Processing request {req_id} with prompt: {prompt}")
        
        cache_id, similarity = self.find_similar_prompts(prompt)

        if cache_id:
            # Cache hit
            self.stats["semantic_hits"] += 1
            
            # Update access
            self.cache[cache_id].access()

            # New request shouldn't be cache since we used cached states itself
            self.no_cache_rids.add(req_id)

            prefix_pos = self.cache[cache_id].prefix_pos

            return True, cache_id, similarity, prefix_pos
        
        # No matches
        self.stats["misses"] += 1
        print(f"No cache hit for prompt: {prompt}")
        return False, None, 0.0, None
    
    def on_req_end(self, prompt: str, req_id: int, prefix_pos: int) -> Optional[str]:
        """
        When a request (promot) ends, store the KV cache information for future reuse if the current request is cachable.

        Args:
            prompt (str): Input prompt, textual.
            req_id(int): Current req id.
            block_tables (List[List[int]]): block tables used by the request for each layer's KV cache.


        Returns:
            Optional[str]: Cache ID of the stored entry, or None if the request is not cachable (contains cached states).
        """
        # If contains cached states, don't cache
        if req_id in self.no_cache_rids:
            self.no_cache_rids.remove(req_id)
            print(f"req id {req_id} is not cachable due to cached states.")
            return None
        
        # If already cached, don't cache again, just return mapped cache ID
        if req_id in self.rid_cid:
            print(f"req id {req_id} is already cached.")
            return self.rid_cid[req_id]
        
        # Otherwise, create a new cache entry

        cache_id = str(uuid.uuid4())
        embedding = self.embedding_function(prompt)

        self.cache[cache_id] = SemanticCacheEntry(
            prompt=prompt,
            embedding=embedding,
            req_id=req_id,
            prefix_pos=prefix_pos,
            timestamp=time.time()
        )

        # Update sid_cid mapping
        self.rid_cid[req_id] = cache_id

        # Update cache statistics
        self.stats["cache_entries"] = len(self.cache)

        # Check if we have exceeded the max cache entries, if so evict LRU
        if len(self.cache) > self.max_cache_entries:
            self._evict_lru()

        print(f"Stored cache entry for prompt: {prompt}, cache_id: {cache_id}")
        return cache_id
    
    def _evict_lru(self):
        """
        Remove the LRU cache entry.
        """

        if not self.cache:
            return
        
        # Find the least recently used entry 
        oldest_id = None
        oldest_time = float('inf')

        for cache_id, entry in self.cache.items():
            if entry.last_accessed < oldest_time:
                oldest_time = entry.last_accessed
                oldest_id = cache_id

        if oldest_id:
            # Remove the oldest entry
            entry = self.cache.pop(oldest_id)
            # Remove from mappings
            if entry.req_id in self.rid_cid:
                del self.rid_cid[entry.req_id]
            
            # Update cache statistics
            self.stats["cache_entries"] = len(self.cache)

            print(f"Evicted LRU cache entry: {oldest_id}, prompt: {entry.prompt}")


    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Dictionary containing cache statistics.
        """
        stats = self.stats.copy()
        total = stats["total_requests"]
        stats["hit_rate"] = stats["semantic_hits"] / total if total > 0 else 0
        return stats
    

    def clear_cache(self):
        """
        Clear the entire cache.
        """
        self.cache.clear()
        self.rid_cid.clear()
        self.no_cache_rids.clear()
        self.stats["cache_entries"] = 0
        print("Cleared all cache entries.")

class SemanticTGIRouter:
    """
    Custom router for TGI with semantic KV cache sharing. 
    Integraes with existing TGI framework to reuse similar KV cache entries.
    """

    def __init__(self, embedding_model: str = EMBEDDINGG_MODEL, similarity_threshold: float = SIMILARITY_THRESHOLD, max_cache_entries: int = 1000):
        """
        Initialize the SemanticTGIRouter.

        Args:
            embedding_model (str): Model name for the embedding function.
            similarity_threshold (float): Cosine similarity threshold for cache hits.
            max_cache_entries (int): Maximum number of cache entries.
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.cache_manager = SemanticCacheManager(
            embedding_function=self.embedding_model.encode,
            similarity_threshold=self.similarity_threshold,
            max_cache_entries=max_cache_entries
        )

        self.active_requests = {}

        print(f"Initialized SemanticTGIRouter with embedding model {embedding_model} and similarity threshold {similarity_threshold}")

    async def route_request(self, request, next_router):
        """
        Route the TGI requests with semantic cachng
        Intercept requests before sent to TGI framework, cehcks for semantically similar prompts, and mopdifoes requests to reuse KV caches when pssbile.


        Args:
            request (_type_): _description_
            next_router (_type_): _description_
        """

        prompt = request.inputs
        req_id = request.id or str(uuid.uuid4())

        # check for semantic cache hit
        cache_hit, cache_id, similarity, prefix_pos = self.cache_manager.on_req_start(prompt, req_id)

        if cache_hit and cache_id:
            # Reuse the KV cache state by setting `past+_key_values` in the request

            cache_entry = self.cache_manager.cache[cache_id]

            # Track request

            self.active_requests[req_id] = {
                "using_cache": True,
                "cache_id": cache_id,
                "original_prompt": prompt,
                "similarity": similarity,
                "prefix_pos": prefix_pos
            }

            # Modify the request to use the cached KV states
            request.past_key_values = cache_entry.req_id
            request.past_key_values_length = prefix_pos

            print(f"Cache hit for request {req_id}, using cache ID {cache_id} with similarity {similarity}")

        else:

            self.active_requests[req_id] = {
                "using_cache": False,
                "cache_id": None,
                "original_prompt": prompt,
                "similarity": similarity,
                "prefix_pos": 0
            }

        # Forward to next router

        response = await next_router.route_request(request)

        # Update cache when request completes if it is cachable (did not use cached states)
        if req_id in self.active_requests and not self.active_requests[req_id].get("using_cache", False):
            
            # extract prefix position from the response
            tokens = tokenizer.encode(prompt)
            prefix_pos = len(tokens)
            
            print(f"Token IDs: {tokens}")
            decoded_tokens = [tokenizer.decode([token_id]) for token_id in tokens]
            print(f"Decoded tokens: {decoded_tokens}")
            
            print(f"Calculated prefix_pos: {prefix_pos} tokens for prompt: {prompt}")

            # store in semantic cache
            self.cache_manager.on_req_end(prompt, req_id, prefix_pos)
            print(f"Stored cache entry for request {req_id} with prompt: {prompt}")

        
        # Clean up the active requests
        if req_id in self.active_requests:
            del self.active_requests[req_id]
            print(f"Cleaned up request {req_id} from active requests.")

        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache_manager.get_stats()
    
    def clear_cache(self):
        self.cache_manager.clear_cache()


# EXAMPLE #TODO DELETE
# Example of implementation with TGI


# Implementation with Text Generation Inference
from text_generation import Client, AsyncClient

def setup_semantic_tgi_server(model_name, port=8080, host="104.171.202.139"):
    """Setup connection to TGI server with semantic caching router"""
    import time
    
    # Create and return client - connecting to your Lambda instance
    client = Client(f"http://{host}:{port}")
    print(f"Connected to TGI server at http://{host}:{port}")

    return client, None



def main():
    print("Diagnostic Script - Check TGI Active -------------------------------------------------------------------------------------------------------------------------------------------------")
    diagnostic.diagnostic()
    print("-------------------------------------------------------------------------------------------------------------------------------------------------")
    
    print("Connecting to TGI server...")
    port = 8080
    host = "104.171.202.139"  # Your Lambda instance IP

    # Connect to existing server
    client, _ = setup_semantic_tgi_server("TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=port, host=host)

    try:
        prompts = [
            "What are some good study habits for university students?",
            "How can I improve my study habits in college?",
            "Suggest effective study strategies for college learners.",
            "What are the best ways to prepare for exams?",
            "Explain the theory of relativity.",
        ]

        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1}: {prompt}")
            result = client.generate(prompt, max_new_tokens=500)

             # Debug the raw result
            """print(f"Raw result type: {type(result)}")
            print(f"Raw result attributes: {dir(result)}")
            print(f"Raw result dict: {result.__dict__ if hasattr(result, '__dict__') else 'No __dict__'}")"""

            print(f"Output: {result.generated_text[:100]}...")

        print("\nCache Statistics:")
        import requests
        try:
            stats_response = requests.get(f"http://{host}:{port}/admin/router/stats")
            stats = stats_response.json()
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        except Exception as e:
            print(f"Could not retrieve cache stats: {e}")

    finally:
        print("Finished run.")


if __name__ == "__main__":
    main()