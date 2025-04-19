# client_simulator.py
import asyncio
import httpx 
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataset_client import distribute_prompts


class LLMClient:

    def __init__(self, client_id: int, server_url: str, client: httpx.AsyncClient):
        """
        Initializes an LLM Client.

        Args:
            client_id: A unique identifier for the client.
            server_url: The base URL of the FastAPI server (e.g., "http://127.0.0.1:8000").
            client: An instance of httpx.AsyncClient for making requests.
        """
        self.client_id = client_id
        self._client = client 
        self.generate_url = f"{server_url}/generate"

    async def query(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        payload = {
            "inputs": prompt,
            "parameters": parameters or {},
            "user_id": f"user_{self.client_id}",
            "cache_strategy": "per_user"
        }
        print(f"Client {self.client_id}: Sending prompt: '{prompt[:50]}...'")
        try:
            headers = {"X-Client-ID": str(self.client_id)}
            response = await self._client.post(self.generate_url, json=payload, headers=headers)
            response.raise_for_status()

            response_data = response.json()
            generated_text = response_data.get("generated_text")
            print(f"Client {self.client_id}: Received response for '{prompt[:50]}...' in {elapsed:.2f}s. Result: '{str(generated_text)[:50]}...'")
            return prompt, generated_text
        except Exception as e:
            print(f"Client {self.client_id}: Request failed for '{prompt[:50]}...'")
            return prompt, None


# --- LLM Client Manager ---
class LLMClientManager:
    """Manages multiple LLMClient instances and orchestrates queries."""

    def __init__(self, server_url: str, num_clients: int, file:str ="class.json"):
        """
        Initializes the Client Manager.

        Args:
            server_url: The base URL of the FastAPI server.
            num_clients: The number of concurrent clients to simulate.
            prompts: A list of prompts that each client will send.
        """
        self.server_url = server_url
        self.num_clients = num_clients
        self.prompts = distribute_prompts(file, num_clients)
        self.clients: List[LLMClient] = []
        self.results: Dict[int, List[Tuple[str, Optional[str]]]] = {}

    async def run_queries(self):
        """
        Creates clients and runs all queries asynchronously, with start and stop data collection.
        """
        start_time = time.monotonic()
        print(f"Starting simulation with {self.num_clients} clients and {len(self.prompts)} prompts each.")

        async with httpx.AsyncClient() as http_client:
            # Start data collection
            try:
                print("Client Manager: Starting data collection...")
                response = await http_client.post(f"{self.server_url}/data/start", json={"output_dir": "./statistics"})
                print(f"Client Manager: Data collection start response: {response.status_code}, {response.json()}")
            except Exception as e:
                print(f"Client Manager: Failed to start data collection: {e}")

            self.clients = [LLMClient(i, self.server_url, http_client) for i in range(self.num_clients)]
            self.results = {client.client_id: [] for client in self.clients}

            tasks = []
            for c in range(self.num_clients):
                for prompt in self.prompts[c]:
                    async def query(c: LLMClient, p: str):
                        return c.client_id, await c.query(p)
                    tasks.append(asyncio.create_task(query(self.clients[c], prompt)))

            print(f"Dispatching {len(tasks)} tasks...")
            query_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in query_results:
                if isinstance(result, Exception):
                    print(f"MANAGER: A task failed: {result}")
                else:
                    client_id, (prompt, response_text) = result
                    if client_id in self.results:
                         self.results[client_id].append((prompt, response_text))

            try:
                print("Client Manager: Stopping data collection...")
                response = await http_client.post(f"{self.server_url}/data/stop")
                print(f"Client Manager: Data collection stop response: {response.status_code}, {response.json()}")
                response = await http_client.post(f"{self.server_url}/clear")
                print("CLEARING CACHE.")
            except Exception as e:
                print(f"Client Manager: Failed to stop data collection: {e}")

        self.report_results()




    def report_results(self):
        """Prints a summary of the results."""
        total_requests = 0
        successful_requests = 0
        failed_requests = 0

        for client_id, client_results in self.results.items():
            client_success = sum(1 for _, resp in client_results if resp is not None)
            client_fail = len(client_results) - client_success
            total_requests += len(client_results)
            successful_requests += client_success
            failed_requests += client_fail
            print(f"  Client {client_id}: {client_success} successful, {client_fail} failed requests.")
            for prompt, resp in client_results:
               print(f"    Prompt: '{prompt[:30]}...' -> Response: '{str(resp)[:50]}...'")

        print(f"\nOverall: {successful_requests}/{total_requests} successful requests ({failed_requests} failed).")


# --- Main execution for client simulation ---
if __name__ == "__main__":
    SERVER_ADDRESS = "http://localhost:9000"
    NUM_CLIENTS = 8

    print("--- LLM Client Simulation ---")
    print(f"Target Server: {SERVER_ADDRESS}")
    print(f"Simulating: {NUM_CLIENTS} clients")

    manager = LLMClientManager(
        server_url=SERVER_ADDRESS,
        num_clients=NUM_CLIENTS,
        file="class.json"
    )

    # Run the simulation using asyncio
    asyncio.run(manager.run_queries())

    print("--- Simulation Complete ---")