import shutil
from fastapi import APIRouter

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from . import models, functions



router = APIRouter()

@router.get("/")
def status():
    
    return {"message": "Status OK"}


@router.post("/converse")
def converse(conversation: models.Conversation):
    response = functions.respond_to_query(conversation.messages[0], conversation.messages[1:])

    return models.BotMessage(content=response)


@router.post("/send_data")
def send_data(conversation: models.Conversation):
    if conversation.channel != "data":
        response = "Incorrect channel for data!"
    else:
        try:
            functions.index_data(conversation)
            response = "Data loaded!"
        except Exception as e:
            response = str(e) 
    return models.BotMessage(content=response)

@router.post("/summarize")
def summarize(messages: list[models.DataMessage]):
        try:
            functions.summarize(messages)
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

