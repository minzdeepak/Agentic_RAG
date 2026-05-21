"""Main entry point for RAG System"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app

if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT
    
    # Disable reload in Docker environment
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        reload=reload
    )
