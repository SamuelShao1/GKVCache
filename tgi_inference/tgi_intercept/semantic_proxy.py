import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from text_generation import Client

from semantic_cache import SemanticTGIRouter

app = FastAPI()
tgi_client = Client("http://tgi-server:80")
router = SemanticTGIRouter()

class GenerationRequest(BaseModel):
    inputs: str
    parameters: dict = {}
    user_id: str = None
    cache_strategy: str = "global_user"

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

        """if r.past_key_values is not None:
            gen_args["past_key_values"] = r.past_key_values
            gen_args["past_key_values_length"] = r.past_key_values_length
            print(f"FORWARDING WITH KV CACHE - req_id={r.past_key_values} at prefix_pos={r.past_key_values_length}")
        else:
            print("FORWARDING WITHOUT KV CACHE")"""

        return tgi_client.generate(r.inputs, **gen_args)

    response = await router.route_request(req_obj, type("Forwarder", (), {"route_request": forward}))

    cached = False
    similarity = None
    source = None

    if hasattr(response, "details"):
        # Check if it's our custom CachedResponse with a dict details
        if isinstance(response.details, dict):
            cached = response.details.get("cached", False)
            similarity = float(response.details.get("similarity")) if response.details.get("similarity") is not None else None
            source = response.details.get("source")

    result = {
        "generated_text": response.generated_text,
        "cached": cached
    }

    if cached:
        result["similarity"] = similarity
        result["source"] = source
    
    return result