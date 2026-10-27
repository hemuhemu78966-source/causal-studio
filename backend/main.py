"""Local FastAPI application serving the React production build."""
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from backend.service import StudioService

app=FastAPI(title='Causal Studio API',version='2.0.0')
service=StudioService()
lock=RLock()
class TrainRequest(BaseModel):
    domain: str='fintech'
    algorithm: Literal['hill','pc','tan','reference']='hill'
    samples: int=Field(default=2000,ge=500,le=5000)
    seed: int=Field(default=42,ge=0,le=99999)
    records: list[dict[str,Any]] | None=Field(default=None,max_length=10000)
    target: str | None=None
class InterventionRequest(BaseModel):
    session_id:str
    treatment:str
    value:Any
    control:Any
    target:str
    desired:Any
class RecourseRequest(BaseModel):
    session_id:str
    profile:dict[str,Any]
    target:str
    desired:Any
    immutable:list[str]=[]
    threshold:float=Field(default=.65,ge=.01,le=.99)
    max_actions:int=Field(default=2,ge=1,le=3)
    costs:dict[str,float] | None=None

def execute(fn,*args,**kwargs):
    try:
        with lock: return fn(*args,**kwargs)
    except (ValueError,KeyError,TypeError) as exc:
        raise HTTPException(422,str(exc)) from exc

@app.get('/api/health')
def health(): return {'status':'ok','version':'2.0.0','frontend':'React'}
@app.post('/api/train')
def train(req:TrainRequest): return execute(service.train,**req.model_dump())
@app.post('/api/intervene')
def intervene(req:InterventionRequest):
    fields=req.model_dump(); sid=fields.pop('session_id')
    return execute(service.intervene,sid,**fields)
@app.post('/api/recourse')
def recourse(req:RecourseRequest):
    fields=req.model_dump(); sid=fields.pop('session_id')
    return execute(service.recourse,sid,**fields)
@app.get('/api/export/{sid}')
def export(sid:str):
    return execute(lambda: service.get(sid)['summary'])
@app.get('/api/dataset/{sid}')
def dataset(sid:str):
    csv=execute(lambda: service.get(sid)['raw'].to_csv(index=False))
    return Response(csv,media_type='text/csv',headers={'Content-Disposition':'attachment; filename=causal-studio-dataset.csv'})

build=Path(__file__).resolve().parents[1]/'frontend'/'dist'
if build.exists():
    app.mount('/assets',StaticFiles(directory=build/'assets'),name='assets')
    @app.get('/')
    def index(): return FileResponse(build/'index.html')
