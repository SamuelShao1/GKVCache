import os
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
model_name = os.getenv("MODEL_ID")
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print(f"MODEL_NAME: {model_name}")
print(f"TOKEN: {token}")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True, use_fast=True, add_prefix_space=True)

# Data Collection
from data_collection.data_collector import DataCollector


EMBEDDINGG_MODEL = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.8

class SemanticCacheEntry:
    """
    A class representing a single entry in the semantic cache.
    The semantic cache stores the input text (prompt), its prompt embedding, and the corresponding KV cache reference.
    """
    def __init__(self, prompt: str, embedding: np.ndarray, req_id: str, prefix_pos, response: str, timestamp: float, gkv_cache_id: str):
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
        self.response = response
        self.timestamp = timestamp
        self.last_accessed = self.timestamp
        self.access_count = 0
        self.gkv_cache_id = gkv_cache_id

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

        # Global cache for all users. Maps cache_id to SemanticCacheEntry
        self.global_cache: Dict[str, SemanticCacheEntry] = {}

        # Per-user cache. Maps user_id to a dictionary of cache_id to SemanticCacheEntry
        self.user_caches: Dict[str, Dict[str, SemanticCacheEntry]] = defaultdict(dict)

        # Maps rid to cid for fast lookups
        self.rid_cid: Dict[int, str] = {}

        # Set of rids that shouldn't be cached, such as if they use cache KV states themselves
        self.no_cache_rids: Set[int] = set()

        # Cache statistics
        self.stats = {
            "total_requests": 0,
            "global_hits": 0,
            "user_hits": 0,
            "misses": 0,
            "global_cache_entries": 0,
            "user_cache_entries": 0
        }

        print(f"Initialized SemanticKVCacheManager with similarity threshold {similarity_threshold}")


    def find_similar_prompts(self, prompt: str, user_id: str, cache_strategy = "global_user") -> Tuple[Optional[str], float, str]:
        """
        Use cosine similarity to compute similarity score, and compare it with cache entries above the threshold.


        Args:
            prompt (str): Query prompt, textual.
            user_id (str): User ID for per-user cache.
            cache_strategy (str): Cache strategy. 
                - "global_user": global and per-user cache.
                - "global_only": Only use global cache, no per-user cache.
                - "per_user": per-user cache, allowing different users to have their own cache entries.
                - "no_cache" : will not use any cached states, even if they are available. Note: TGI still does KV caching within a single request, but "no-cache"
                               will not use any cached states from previous requests.

                This caching strategy parameter is also used to determined cache storage as well as cache retrieval.

        Returns:
            Tuple[Optional[str], float, str]: Representing (cache_id of most similar prompt, cosine similarity score, source ("global" or "per-user" pr "none")).
            Returns (None, 0.0) if no similar prompt is found, or no cache entries exceed the threshold value.
        """
        self.stats["total_requests"] += 1

        if cache_strategy == "no_cache":
            # Cache is disabled
            return None, 0.0, None, None
        
        if cache_strategy == "global_user" and (not self.global_cache and not self.user_caches.get(user_id, None)):
            # Cache strategy is global_user, but no cache entries exist in either global or user-specific cache
            return None, 0.0, None, None
        
        if cache_strategy == "global_only" and not self.global_cache:
            # Cache strategy is global_only, but no cache entries exist in global cache
            return None, 0.0, None, None
        
        if cache_strategy == "per_user" and not self.user_caches.get(user_id, None):
            # Cache strategy is per_user, but no cache entries exist in user-specific cache
            return None, 0.0, None, None
        
        query_embedding = self.embedding_function(prompt)

        # print(f"Query embedding: {query_embedding}")
        # Find the most similar prompt in the cache.
        best_cache_id = None
        best_similarity = 0.0
        best_source = None
        best_response = None

        print(f"--------------SEMANTIC FIND PRE-OP STATISTICS------------------")
        print(f"QUERY: {prompt}")
        print(f"STATS-TOTAL_REQUESTS: {self.stats['total_requests']}")
        print(f"STATS-GLOBAL_CACHE_ENTRIES: {self.stats['global_cache_entries']}")
        print(f"STATS-USER_CACHE_ENTRIES: {self.stats['user_cache_entries']}")
        print(f"STATS-GLOBAL_HITS: {self.stats['global_hits']}")
        print(f"STATS-USER_HITS: {self.stats['user_hits']}")
        print(f"STATS-MISSES: {self.stats['misses']}")
        print(f"GLOBAL_CACHE_TABLE: {self.global_cache.items()}")
        print(f"USER_CACHE_TABLE: {self.user_caches.items()}")
        print(f"CACHE_STRATEGY: {cache_strategy}")  
        print(f"--------------SEMANTIC FIND PRE-OP STATISTICS------------------")

        # Check user cache
        if cache_strategy in ["global_user", "per_user"]:
            print(f"Checking user cache for user ID: {user_id}")
            user_cache = self.user_caches.get(user_id, {})
            print(f"User cache items: {user_cache.items()}")
            for cache_id, entry in user_cache.items():
                cached_embedding = entry.embedding
            
                cos_sim = self._cos_sim(query_embedding, cached_embedding)
                
                print(f"Comparing with cache ID {cache_id}, similarity: {cos_sim}")

                if cos_sim > best_similarity:
                    best_similarity = cos_sim
                    best_cache_id = cache_id
                    best_source = "user"
                    best_response = entry.response
                    print(f"New best cache ID: {best_cache_id}, similarity: {best_similarity}")


        # Check global cache
        if cache_strategy in ["global_user", "global_only"]:
            print(f"Checking global cache")
            for cache_id, entry in self.global_cache.items():
                cached_embedding = entry.embedding
            
                cos_sim = self._cos_sim(query_embedding, cached_embedding)
                
                print(f"Comparing with cache ID {cache_id}, similarity: {cos_sim}")

                if cos_sim > best_similarity:
                    best_similarity = cos_sim
                    best_cache_id = cache_id
                    best_source = "global"
                    best_response = entry.response
                    print(f"New best cache ID: {best_cache_id}, similarity: {best_similarity}")


         # Return result if above threshold
        if best_similarity > self.similarity_threshold:
            if best_source == "global":
                print(f"HIT! - CACHE HIT-MATCH for prompt: {prompt}. \n    BEST WAS: similarity: {best_similarity} with cache ID: {best_cache_id} from {best_source}")
                self.stats["global_hits"] += 1
            else:
                print(f"HIT! - CACHE HIT-MATCH for prompt: {prompt}. \n    BEST WAS: similarity: {best_similarity} with cache ID: {best_cache_id} from {best_source}")
                self.stats["user_hits"] += 1
            return best_cache_id, best_similarity, best_source, best_response
        
        print(f"MISS! - CACHE MISS-MATCH for prompt: {prompt}. \n    BEST WAS: similarity: {best_similarity} with cache ID: {best_cache_id} from {best_source}")
        self.stats["misses"] += 1
        return None, best_similarity, None, None
    

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
    
    def on_req_start(self, prompt: str, req_id: str, user_id: str, cache_strategy = "global_user") -> Tuple[bool, Optional[str], float, str, str]:
        """
        When a new request(prompt) is started, check if there are similar prompts and return the cache hit if found.

        Args:
            prompt (str): Input prompt.
            req_id (str): Current req id.
            user_id (str): User ID for per-user cache.
            cache_strategy (str): Cache strategy. 
                - "global_user": global and per-user cache.
                - "global_only": Only use global cache, no per-user cache.
                - "per_user": per-user cache, allowing different users to have their own cache entries.
                - "no_cache" : will not use any cached states, even if they are available. Note: TGI still does KV caching within a single request, but "no-cache"
                               will not use any cached states from previous requests.

                This caching strategy parameter is also used to determined cache storage as well as cache retrieval.

        Returns:
            Tuple[bool, Optional[str], float, str, str]: Representation of (cache hit status, cache_id, similarity score, source, response)
            Returns (False, None, 0.0) if no similar prompt is found.
        """
       
        print(f"ON_REQ_START - Processing request {req_id} with prompt: {prompt}")

        cache_id, similarity, source, response = self.find_similar_prompts(prompt, user_id, cache_strategy)

        if cache_id:
            # Cache hit
            if source == "global":
                entry = self.global_cache[cache_id]
            else:
                entry = self.user_caches[user_id][cache_id]
            
            # Update access
            entry.access()

            # New request shouldn't be cache since we used cached states itself
            self.no_cache_rids.add(req_id)


            return True, cache_id, similarity, entry.prefix_pos, source, entry.response
        
        # No matches
        self.stats["misses"] += 1
        print(f"No cache hit for prompt: {prompt}")
        return False, None, 0.0, None, None, None
    
    def on_req_end(self, prompt: str, req_id: int, prefix_pos: int, response: str, user_id: str, cache_strategy = "global_user", gkv_cache_id = None) -> Optional[str]:
        """
        When a request (promot) ends, store the KV cache information for future reuse if the current request is cachable.

        Args:
            prompt (str): Input prompt, textual.
            req_id(int): Current req id.
            prefix_pos (int): Last token position of the prompt.
            user_id (str): User ID for per-user cache.
            cache_strategy (str): Cache strategy. 
                - "global_user": global and per-user cache.
                - "global_only": Only use global cache, no per-user cache.
                - "per_user": per-user cache, allowing different users to have their own cache entries.
                - "no_cache" : will not use any cached states, even if they are available. Note: TGI still does KV caching within a single request, but "no-cache"
                               will not use any cached states from previous requests.

                This caching strategy parameter is also used to determined cache storage as well as cache retrieval.

        Returns:
            Optional[str]: Cache ID of the stored entry, or None if the request is not cachable (contains cached states).
        """
        # If contains cached states, don't cache
        if cache_strategy == "no_cache":
            print(f"ON_REQ_END - req id {req_id} is not cachable due to no_cache strategy.")
            return None
        
        if req_id in self.no_cache_rids:
            self.no_cache_rids.remove(req_id)
            print(f"ON_REQ_END - req id {req_id} is not cachable due to cached states.")
            return None
        
        # If already cached, don't cache again, just return mapped cache ID
        if req_id in self.rid_cid:
            print(f"ON_REQ_END - req id {req_id} is already cached.")
            return self.rid_cid[req_id]
        
        # Otherwise, create a new cache entry

        cache_id = str(uuid.uuid4())
        embedding = self.embedding_function(prompt)

        entry = SemanticCacheEntry(
            prompt=prompt,
            embedding=embedding,
            req_id=req_id,
            prefix_pos=prefix_pos,
            response=response,
            timestamp=time.time(),
            gkv_cache_id=gkv_cache_id
        )

        # Store in appropriate cache(s)
        # Global cache
        if cache_strategy in ["global_user", "global_only"]:
            self.global_cache[cache_id] = entry
            self.stats["global_cache_entries"] = len(self.global_cache)
        # User ache
        if cache_strategy in ["global_user", "per_user"]:
            self.user_caches[user_id][cache_id] = entry
            self.stats["user_cache_entries"] = len(self.user_caches[user_id])

        # Update sid_cid mapping
        self.rid_cid[req_id] = cache_id

        # Update cache statistics
        self.stats["cache_entries"] = len(self.global_cache)

        # Check if we have exceeded the max cache entries, if so evict LRU
        self._check_cache_sizes()

        print(f"Stored cache entry for prompt: {prompt}, cache_id: {cache_id}")
        return cache_id
    
    def _check_cache_sizes(self):
        """
        Checks the global and user caches for size limits.
        If exceeded, evict the least recently used (LRU) entry.
        """
        print(f"EVICTION_CHECK - max cache entries: {self.max_cache_entries}")
        print(f"    EVICTION_CHECK - global cache size: {len(self.global_cache)}")
        print(f"    EVICTION_CHECK - user cache size: {len(self.user_caches)}")
         # Check global cache
        if len(self.global_cache) > self.max_cache_entries:
            self._evict_lru(self.global_cache, len(self.global_cache) - self.max_cache_entries)
        
        # Check each user cache
        for user_id, user_cache in self.user_caches.items():
            if len(user_cache) > self.max_cache_entries:
                self._evict_lru(user_cache, len(user_cache) - self.max_cache_entries)

    def _evict_lru(self, cache_dict: Dict[str, SemanticCacheEntry], num_to_evict: int):
        """
        Remove the LRU cache entry.
        """
        if not cache_dict or num_to_evict <= 0:
            # No entries to evict
            return
        
        entries_to_evict = sorted(cache_dict.items(), key=lambda item: item[1].last_accessed)[:num_to_evict]
        
        for cache_id, entry in entries_to_evict:
            del cache_dict[cache_id]
            if entry.req_id in self.rid_cid:
                del self.rid_cid[entry.req_id]
            
            print(f"EVICTION - Evicted LRU cache entry: {cache_id}, prompt: {entry.prompt}")


    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Dictionary containing cache statistics.
        """
        stats = self.stats.copy()
        return stats
    

    def clear_cache(self):
        """
        Clear the entire cache.
        """
        self.global_cache.clear()
        self.rid_cid.clear()
        self.no_cache_rids.clear()
        self.stats["cache_entries"] = 0
        print("Cleared all cache entries.")

class SemanticTGIRouter:
    """
    Custom router for TGI with semantic KV cache sharing. 
    Integraes with existing TGI framework to reuse similar KV cache entries.
    """

    def __init__(self, 
                 embedding_model: str = EMBEDDINGG_MODEL, 
                 similarity_threshold: float = SIMILARITY_THRESHOLD, 
                 max_cache_entries: int = 1000,
                 collect_data: bool = False):
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
        # Initialize benchmarking
        self.collect_data = collect_data
        if collect_data:
            self.data_collection = DataCollector()

        print(f"PROXY_INIT_ROUTER - SemanticTGIRouter with embedding model {embedding_model} and similarity threshold {similarity_threshold}")
        if collect_data:
            print(f"ROUTER - Data collection enabled.")

    async def route_request(self, request, next_router):
        """
        Route the TGI requests with semantic cachng
        Intercept requests before sent to TGI framework, cehcks for semantically similar prompts, and mopdifoes requests to reuse KV caches when pssbile.


        Args:
            request (_type_): _description_
            next_router (_type_): _description_
        """
        # For data collection
        request_start_time = time.time()
        embedding_start_time = None
        embedding_end_time = None
        similarity_search_start_time = None
        similarity_search_end_time = None

        prompt = request.inputs
        req_id = request.id or str(uuid.uuid4())

        # Get request parameters
        user_id = getattr(request, "user_id", None)
        cache_strategy = getattr(request, "cache_strategy", "global_user")
        print(f"-------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"PARAM:REQUEST_ID - request_id: {req_id}")
        print(f"PARAM:REQUEST_PROMPT - request_prompt: {prompt}")
        print(f"PARAM:USER_ID - user_id: {user_id}")
        print(f"PARAM:REQUEST_CACHE_STRATEGY - request_cache_strategy: {cache_strategy}")


        # Embedding time data collection
        if self.collect_data and self.data_collection.is_active:
            embedding_start_time = time.time()

        if self.collect_data and self.data_collection.is_active:
            embedding_end_time = time.time()
            similarity_search_start_time = time.time()


        # check for semantic cache hit
        cache_hit, cache_id, similarity, prefix_pos, source, cached_response = self.cache_manager.on_req_start(prompt, req_id, user_id, cache_strategy)
        
        if self.collect_data and self.data_collection.is_active:
            similarity_search_end_time = time.time()

        if cache_hit and cache_id:
            # Reuse the KV cache state by setting `past+_key_values` in the request

            if source == "global":
                entry = self.cache_manager.global_cache[cache_id]
            else:  # source == 'user'
                entry = self.cache_manager.user_caches[user_id][cache_id]

            if hasattr(entry, "gkv_cache_id") and entry.gkv_cache_id:
                #TODO
                if not hasattr(request, "parameters") or request.parameters is None:
                    request.parameters = {} 
                request.parameters["gkv_cache_id"] = entry.gkv_cache_id
                print(f"ROUTER - Using GKV cache ID: {entry.gkv_cache_id} for request {req_id}")
                response = await next_router.route_request(request)
            else:
                print("ROUTER ERROR - No GKV cache ID found in entry.")
                print(entry, entry.__dict__)
                raise ValueError("No GKV cached ID found in entry.")


            print(f"ROUTER - Cache HIT for request {req_id} \n\
                        Prompt:{prompt} \n\
                        Cache_ID {cache_id} \n\
                        Similarity {similarity} \n\
                        Source {source}.")

        else:
            print(f"ROUTER - Cache MISS for request {req_id} \n\
                        Prompt:{prompt} \n\
                        Cache_ID {cache_id} \n\
                        Similarity {similarity} \n\
                        Source {source}.")
            print("ROUTER - Forwarding to TGI")

            response = await next_router.route_request(request)
            
            gkv_cache_id = None
            if hasattr(response, "details") and isinstance(response.details, dict):
                gkv_cache_id = response.details.get("gkv_cache_id")

            tokens = tokenizer.encode(prompt)
            prefix_pos = len(tokens)


            self.cache_manager.on_req_end(prompt, req_id, prefix_pos, response.generated_text, user_id, cache_strategy, gkv_cache_id=gkv_cache_id)
        
        # End timing  request
        request_end_time = time.time()
        
        # Record data if enabled
        if self.collect_data and self.data_collection.is_active:
            # Get the response length
            response_length = len(response.generated_text) if hasattr(response, "generated_text") else 0
            
            self.data_collection.record_query(
                request_id=req_id,
                prompt=prompt,
                user_id=user_id,
                cache_strategy=cache_strategy,
                cache_hit=cache_hit,
                cache_source=source if cache_hit else None,
                similarity_score=similarity if cache_hit else None,
                start_time=request_start_time,
                end_time=request_end_time,
                embedding_start_time=embedding_start_time,
                embedding_end_time=embedding_end_time,
                similarity_search_start_time=similarity_search_start_time,
                similarity_search_end_time=similarity_search_end_time,
                response_length=response_length
            )
        
        return response
    def start_benchmark(self):
        if not self.collect_data:
            self.collect_data = True
            self.data_collection = DataCollector()
            
        return self.data_collection.start_tracking()
    
    def stop_benchmark(self):
        if not self.collect_data:
            return {"error": "Benchmarking is not enabled"}
            
        return self.data_collection.stop_tracking()
    

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache_manager.get_stats()
    
    def clear_cache(self):
        self.cache_manager.clear_cache()


