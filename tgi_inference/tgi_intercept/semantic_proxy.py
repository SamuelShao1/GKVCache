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

@app.post("/generate")
async def generate(request: GenerationRequest):
    class DummyRequest:
        def __init__(self, inputs):
            self.inputs = inputs
            self.id = None
            self.past_key_values = None
            self.past_key_values_length = 0

    req_obj = DummyRequest(inputs=request.inputs)

    # Route through semantic cache
    async def forward(req_obj):
        return tgi_client.generate(
            req_obj.inputs,
            **request.parameters
        )

    response = await router.route_request(req_obj, type("Forwarder", (), {"route_request": forward}))
    return {"generated_text": response.generated_text}