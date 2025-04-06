# dummy_tgi_server.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import time
import random # To add slight variability to response time

print("--- Dummy TGI Server ---")
print("This server mimics the /generate endpoint for client testing.")
print("It accepts POST requests with {'inputs': '...', 'parameters': {...}}")
print("And returns {'generated_text': '...'}")
print("Runs on http://localhost:9000")
print("------------------------")

app = FastAPI(title="Test TGI Server")


class GenerationRequest(BaseModel):
    inputs: str
    parameters: dict = {}

@app.post("/generate")
async def dummy_generate(request: GenerationRequest):
    """
    Handles POST requests to /generate.
    """
    request_received_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received request:")
    print(f"  Inputs: '{request.inputs}'")
    print(f"  Parameters: {request.parameters}")


    max_tokens = request.parameters.get('max_new_tokens', 'N/A')
    generated_text = (
        f"DUMMY RESPONSE for input: '{request.inputs[:80]}...' "
        f"(Params: max_new_tokens={max_tokens}, ...) "
    )
    response_payload = {"generated_text": generated_text}

    return response_payload

# Add a root endpoint for a basic health check via browser
@app.get("/")
async def root():
    """Provides a simple message indicating the server is running."""
    return {"message": "Dummy TGI server running. Send POST requests to /generate"}

if __name__ == "__main__":
    host = "localhost"
    port = 9000
    print(f"\nStarting dummy server on http://{host}:{port}")
    print("Press CTRL+C to stop.")
    # To run: python dummy_tgi_server.py
    # Requires: pip install fastapi uvicorn
    uvicorn.run(app, host=host, port=port)

    # If you want auto-reloading during development (install 'python-multipart' if needed):
    # uvicorn.run("dummy_tgi_server:app", host=host, port=port, reload=True)