from fastapi import FastAPI,Request
from pydantic import BaseModel
from project_root.inference.translate import translate_sentence
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

class translation(BaseModel):
    sentence : str
    

app = FastAPI()
app.mount("/static", StaticFiles(directory="project_root/static"), name="static")
templates = Jinja2Templates(directory="project_root/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate/")
def translate(request:translation):
    trans = translate_sentence(request.sentence)
    return {'translation': trans}