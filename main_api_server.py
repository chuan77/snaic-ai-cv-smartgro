"""Runs the FastAPI detection backend via uvicorn."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.deploy.api_server:app", host="0.0.0.0", port=8000, reload=True)
