"""Runs the FastAPI detection backend via uvicorn."""
import uvicorn

from src.deploy.api_server import get_server_config

if __name__ == "__main__":
    config = get_server_config()
    uvicorn.run(
        "src.deploy.api_server:app",
        host=config["host"],
        port=config["port"],
        reload=config["reload"],
    )
