from fastapi import FastAPI
from pydantic import BaseModel
import torch
from project_root.inference.translate import translate_sentence
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

class translation(BaseModel):
    sentence : str
    

app = FastAPI()
app.mount("/static", StaticFiles(directory="project_root/templates"), name="static")
@app.get("/")
def read_index():
    return FileResponse("project_root/templates/index.html")
@app.post("/translate/")
def translate(request:translation):
    trans = translate_sentence(request.sentence)
    return {'translated: ': trans}