import os
import time
from fastapi.responses import FileResponse
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from text_generation import Client

from semantic_cache import SemanticTGIRouter

from data_collection.data_collector import DataCollector
app = FastAPI()
tgi_client = Client("http://tgi-server:80")
router = SemanticTGIRouter()

class GenerationRequest(BaseModel):
    inputs: str
    parameters: dict = {}
    user_id: str = None
    cache_strategy: str = "global_user"

class DataCollectionRequest(BaseModel):
    output_dir: str = "./statistics"


@app.post("/generate")
async def generate(request: GenerationRequest):
    if not request.user_id:
        return {"error": "user_id is required", "status_code": 400}
    
    valid_strategies = ["global_user", "global_only", "per_user", "no_cache"]
    if request.cache_strategy not in valid_strategies:
        print(f"WARN - Invalid cache strategy: {request.cache_strategy}. Defaulting to 'global_user'.")
        request.cache_strategy = "global_user"


    class DummyRequest:
        def __init__(self, inputs, user_id, cache_strategy):
            self.inputs = inputs
            self.id = None
            self.past_key_values = None
            self.past_key_values_length = 0
            self.user_id = user_id
            self.cache_strategy = cache_strategy

    req_obj = DummyRequest(inputs=request.inputs, user_id=request.user_id, cache_strategy=request.cache_strategy)

    # Route through semantic cache
    print(f"INFO - Using Semantic Cache")
    async def forward(r):
        gen_args = {
            "max_new_tokens": request.parameters.get("max_new_tokens", 100)
        }

        if hasattr(r, 'parameters') and r.parameters and 'gkv_cache_id' in r.parameters:
            gen_args["gkv_cache_id"] = r.parameters['gkv_cache_id']
            print(f"FORWARDING WITH KV CACHE - gkv_cache_id={r.parameters['gkv_cache_id']}")
        else:
            print("FORWARDING WITHOUT KV CACHE")

        response = tgi_client.generate(r.inputs, **gen_args)
        info = tgi_client.headers
        print(f"INFO - TGI Headers: {info}")
        if hasattr(response, 'details') and isinstance(response.details, dict) and 'gkv_cache_id' in response.details:
            response.gkv_cache_id = response.details['gkv_cache_id']

        return tgi_client.generate(r.inputs, **gen_args)

    response = await router.route_request(req_obj, type("Forwarder", (), {"route_request": forward}))

    cached = False
    similarity = None
    source = None
    gkv_cache_id = None

    if hasattr(response, "details"):
        # Check if it's our custom CachedResponse with a dict details
        if isinstance(response.details, dict):
            cached = response.details.get("cached", False)
            similarity = float(response.details.get("similarity")) if response.details.get("similarity") is not None else None
            source = response.details.get("source")
            gkv_cache_id = response.details.get("gkv_cache_id")

    result = {
        "generated_text": response.generated_text,
        "cached": cached
    }

    print(f"INFO - RESPONSE: {response}")

    if cached:
        result["similarity"] = similarity
        result["source"] = source
    
    return result

@app.post("/data/start")
async def start_benchmark(request: DataCollectionRequest = None):
    if request is None:
        request = DataCollectionRequest()
        
    result = router.start_benchmark()
    return result


@app.post("/data/stop")
async def stop_benchmark():
    result = router.stop_benchmark()
    return result

@app.get("/data/csv")
async def get_benchmark_csv():
    """
    Retrieve the current benchmark CSV file.
    """
    if not router.collect_data or not hasattr(router, 'data_collection'):
        return {"error": "Colleciton is not enabled"}
    
    try:
        csv_file = router.data_collection.csv_file
        
        if not os.path.exists(csv_file):
            return {"error": "CSV file not found", "details": f"File {csv_file} does not exist"}
        
        return FileResponse(
            path=csv_file, 
            filename=os.path.basename(csv_file),
            media_type="text/csv"
        )
    except Exception as e:
        return {"error": str(e)}

@app.get("/data/files/{filename}")
async def get_benchmark_file(filename: str):
    """
    Download a specific benchmark file by name.
    """

    if not router.collect_data:
        return {"error": "Benchmarking is not enabled"}
    
    try:
        output_dir = router.data_collection.output_dir
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return {"error": f"File {filename} not found"}
        
        media_type = "text/csv" if filename.endswith('.csv') else "application/json"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
    except Exception as e:
        return {"error": str(e)}