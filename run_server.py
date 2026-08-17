"""Dynamic Oracle server launcher."""
import os
import sys
from pathlib import Path

# Set root directory
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
os.chdir(str(root_dir))

if __name__ == "__main__":
    import uvicorn
    print("==================================================================")
    print("⚡ DYNAMIC ORACLE — SOCCER MATCH PREDICTION & SIMULATION SERVER")
    print("📍 Running locally at: http://127.0.0.1:5100")
    print("==================================================================")
    uvicorn.run("src.service.server:app", host="127.0.0.1", port=5100, reload=False)
