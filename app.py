"""Launch the React + FastAPI application (no Streamlit required)."""
from pathlib import Path
import sys
import webbrowser
from threading import Timer
import uvicorn

if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    if not (root/'frontend'/'dist'/'index.html').exists():
        sys.exit('Build the frontend first: npm --prefix frontend ci && npm --prefix frontend run build')
    print('Causal Studio 2.0: http://127.0.0.1:8000')
    Timer(2,lambda:webbrowser.open('http://127.0.0.1:8000')).start()
    uvicorn.run('backend.main:app',host='127.0.0.1',port=8000)
