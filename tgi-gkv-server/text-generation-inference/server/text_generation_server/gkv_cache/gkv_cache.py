import time
import uuid
import pickle
from typing import Dict, Optional, Tuple, Any
import torch

class GKVCacheEntry:
    """
    Stores past key-value pairs for a specific prompt.
    """

    def __init__(self, entry_id: str, req_id: str, prefix_pos: int, past_kv_data: Dict[str, torch.Tensor], timestamp: float = None):
        """
        Initializes the GKVCacheEntry with the provided parameters.

        Args:
            entry_id (str): Unique identifier for the cache entry.
            req_id (str): Request (rid) identifier of originating request.
            prefix_pos (int): Position of the prefix in the input, aka the position in the sequence where the prompt ends.
            past_kv_data (Dict[str, torch.Tensor]): Key or value tensor entries. Represented by a dictionary where
                the dictionary key is a string identifying the tensor type (key/value) and layer_idx
                the dictionary value is the tensor itself.
                Example: {"key_0": torch.Tesnor(...)}, represents a entry for the key tensor at layer 0.
                Example: {"value_0": torch.Tensor(...)}, represents a entry for the value tensor at layer 0.
            timestamp (float, optional): _description_. Defaults to None.
        """
        self.id = entry_id
        self.req_id = req_id
        self.prefix_pos = prefix_pos
        self.past_kv_data = past_kv_data
        self.timestamp = timestamp if timestamp else time.time()
        self.last_accessed = self.timestamp
        self.access_count = 0

    def access(self):
        """
        Updates the last accessed time and increments the access count.
        """
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the cache entry to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the cache entry.
        """
        return {
            "id": self.id,
            "req_id": self.req_id,
            "prefix_pos": self.prefix_pos,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
    

class GKVCacheManager:
    """
    Manages the GKV cache entries for operations, reuse, and eviction.
    """

    def __init__(self, max_entries: int = 100, ttl: int = 3600): #TODO TTL, max entries
        """
        Initializes the GKVCacheManager with the provided parameters.

        Args:
            max_entries (int, optional): Max number of entries to store. Defaults to 100.
            ttl (int, optional): Cache entry TTL. Defaults to 3600.
        """

        self.entries: Dict[str, GKVCacheEntry] = {}
        self.max_entries = max_entries
        self.ttl = ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, cache_id: str) -> Optional[GKVCacheEntry]:
        """
        Retrieves a cache entry by its ID.

        Args:
            cache_id (str): Unique identifier for the cache entry.

        Returns:
            Optional[GKVCacheEntry]: The cache entry if found, otherwise None.
        """
        # Cache not found, Cache MISS
        if cache_id not in self.entries:
            self.stats["misses"] += 1
            print(f"Cache not found, Cache MISS for ID: {cache_id}")
            return None
        
        # Otherwise retrieve the entry
        entry = self.entries[cache_id]

        # Cache expired, Cache MISS
        if time.time() - entry.timestamp > self.ttl:
            print(f"Cache entry expired, Cache MISS: {cache_id}")
            del self.entries[cache_id]
            self.stats["misses"] += 1
            return None
        
        # Update access

        entry.access()
        self.stats["hits"] += 1
        print(f"Cache HIT for ID: {cache_id}")
        return entry
    
    def store(self, req_id: str, prefix_pos: int, past_kv_data: Dict[str, torch.Tensor]) -> str:
        """
        Stores a new cache entry.

        Args:
            req_id (str): Originating request's id.
            prefix_pos (int): Position of the prefix in the input, aka the position in the sequence where prompt ends.
            past_kv_data (Dict[str, torch.Tensor]): Key or value tensor entries. Represented by a dictionary where
                the dictionary key is a string identifying the tensor type (key/value) and layer_idx
                the dictionary value is the tensor itself.
                Example: {"key_0": torch.Tesnor(...)}, represents a entry for the key tensor at layer 0.
                Example: {"value_0": torch.Tensor(...)}, represents a entry for the value tensor at layer 0.

        Returns:
            str: ID of cache entry once stored.
        """
        print("PAST KV DATA", past_kv_data)
        
        cache_id = str(uuid.uuid4())    

        entry = GKVCacheEntry(
            entry_id = cache_id, 
            req_id = req_id,
            prefix_pos = prefix_pos, 
            past_kv_data = past_kv_data
        )

        # Evict check
        if len(self.entries) >= self.max_entries:
            self._evict_lru()
        
        # Store the entry
        self.entries[cache_id] = entry
        self.stats["stores"] += 1
        print(f"Stored cache entry with ID: {cache_id} for request ID: {req_id}")

        return cache_id
        
    def _evict_lru(self):
        """
        LRU scheme for cache eviction.
        """
        if not self.entries:
            return
        
        # Sort entries with oldest first
        sorted_entries = sorted(self.entries.items(), key=lambda x: x[1].last_accessed)

        oldest, _ = sorted_entries[0]
        del self.entries[oldest]
        self.stats["evictions"] += 1
        print(f"Evicted cache entry with ID: {oldest} due to LRU policy.")
        print(f"Cache stats: {self.stats}")
        print(f"Cache size: {len(self.entries)}")

    def clear(self):
        
        """
        Clears the cache.
        """
        self.entries.clear()
        print("Cache cleared.")

    def get_stats(self) -> Dict[str, int]:
        """
        Returns cache statistics.

        Returns:
            Dict[str, int]: Cache statistics.
        """
        stats = self.stats.copy()
        stats["entries"] = len(self.entries)
        return stats #TODO