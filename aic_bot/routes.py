import shutil
from fastapi import APIRouter

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

import .models
import .functions



router = APIRouter()

@router.get("/")
def status():
    
    return {"message": "OK"}


@router.post("/converse")
def converse(conversation: models.Conversation):
    
    return {"message": response}


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
    return {"message": response}

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        
    return {"message": f"Successfully uploaded {file.filename}"}

