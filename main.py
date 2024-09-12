import subprocess
import requests
import json
import time
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI()

# Enable CORS for all origins (you can limit this to specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base URL for OpenAI's API
OPENAI_API_BASE = 'https://api.openai.com'

# List of o1 models
O1_MODELS = ['o1-preview', 'o1-mini']

# Unsupported parameters for o1 models during beta
UNSUPPORTED_PARAMETERS = [
    'temperature', 'top_p', 'n', 'presence_penalty', 'frequency_penalty',
    'stream', 'functions', 'function_call', 'logit_bias', 'user', 'system_prompt'
]

@app.get("/")
async def root():
    return {"message": "Welcome to the OpenAI Proxy!"}


@app.post("/v1/{path:path}")
async def proxy_post(path: str, request: Request, authorization: Optional[str] = Header(None)):
    # Extract the Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")

    # Get the API key from the Authorization header
    openai_api_key = authorization.split("Bearer ")[-1]

    # Construct the full OpenAI API URL
    openai_url = f"{OPENAI_API_BASE}/v1/{path}"

    # Extract the incoming request body
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid JSON data"}, status_code=400)

    # Modify the request if the model is an o1 model
    model = body.get("model", "")
    
    if model in O1_MODELS:
        # Remove unsupported parameters
        for param in UNSUPPORTED_PARAMETERS:
            body.pop(param, None)

        # Handle max_tokens and max_completion_tokens
        if "max_tokens" in body:
            # For o1 models, use max_completion_tokens instead
            body["max_completion_tokens"] = body.pop("max_tokens")
        else:
            # If neither is specified, set a default max_completion_tokens
            body["max_completion_tokens"] = 25000  # Default value, adjust as needed

        # Remove system messages (not supported)
        body["messages"] = [
            msg for msg in body.get("messages", []) if msg.get("role") != "system"
        ]

    # For gpt-4o or other models, pass through the request without modification
    # Forward the modified request to OpenAI's API
    response = requests.post(
        openai_url,
        headers={"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"},
        json=body,
        stream=True
    )

    # Return the streamed response back to the client
    return StreamingResponse(response.iter_content(chunk_size=8192), status_code=response.status_code)


@app.get("/v1/{path:path}")
async def proxy_get(path: str, authorization: Optional[str] = Header(None)):
    # Extract the Authorization header for GET requests
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")

    # Get the API key from the Authorization header
    openai_api_key = authorization.split("Bearer ")[-1]

    # Construct the full OpenAI API URL
    openai_url = f"{OPENAI_API_BASE}/v1/{path}"

    # Forward the GET request to OpenAI's API
    response = requests.get(
        openai_url,
        headers={"Authorization": f"Bearer {openai_api_key}"},
        stream=True
    )

    # Return the streamed response back to the client
    return StreamingResponse(response.iter_content(chunk_size=8192), status_code=response.status_code)


def start_ngrok():
    """Starts ngrok and returns the public URL."""
    # Start ngrok with a tunnel on port 5000
    ngrok_command = ["ngrok", "http", "5000"]
    ngrok_process = subprocess.Popen(ngrok_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for ngrok to establish the tunnel
    time.sleep(2)  # Wait a moment to ensure ngrok starts up properly
    
    # Retrieve the public URL from ngrok's API
    try:
        ngrok_response = requests.get("http://127.0.0.1:4040/api/tunnels")
        ngrok_url = ngrok_response.json()["tunnels"][0]["public_url"]
        return ngrok_url
    except Exception as e:
        print(f"Error fetching ngrok URL: {e}")
        return None


if __name__ == "__main__":
    import uvicorn

    # Start FastAPI in the background
    uvicorn_server = subprocess.Popen(["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "5000"])

    # Start ngrok and print the URL
    ngrok_url = start_ngrok()
    if ngrok_url:
        print(f"ngrok tunnel started: {ngrok_url}")
        print(f"Use this URL in Cursor: {ngrok_url}")
    else:
        print("Failed to start ngrok.")

    try:
        # Keep the process alive to allow ngrok and FastAPI to run together
        uvicorn_server.wait()
    except KeyboardInterrupt:
        print("Shutting down...")

        # Terminate ngrok and FastAPI when done
        uvicorn_server.terminate()
        subprocess.run(["killall", "ngrok"])
