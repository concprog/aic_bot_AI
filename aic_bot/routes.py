import shutil
from fastapi import APIRouter

from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from haystack_integrations.components.retrievers.qdrant import retriever

from . import models, functions



router = APIRouter()

@router.get("/")
def status():
    
    return {"message": "Status OK"}


@router.post("/converse")
def converse(conversation: models.Conversation):
    try:
        response = pipelines.rqa_pipeline.run(conversation)
    except Exception as e:
        response = str(e)
    return models.BotMessage(content="")


@router.post("/send_data")
def send_data(data: list[models.DataMessage]):
    try:
        
        response = "Data loaded!"
    except Exception as e:
        response = str(e) 
    return models.BotMessage(content=response)

@router.post("/summarize")
def summarize(messages: list[models.Message]):
    try:
        response = "Data loaded!"
    except Exception as e:
        response = str(e) 
    return models.BotMessage(content=response)

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return models.BotMessage(content="There was an error uploading the file")
    finally:
        file.file.close()
        
    return models.BotMessage(content=f"Successfully uploaded {file.filename}")

