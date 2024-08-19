import shutil
from fastapi import APIRouter

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from . import functions, pipelines


router = APIRouter()
models = functions.get_models()


@router.get("/")
def status():

    return models.BotMessage(content="Status OK")


@router.post("/converse")
def converse(conversation: models.Conversation):
    try:
        response = pipelines.rqa_pipeline.run(
            functions.rqa_query(
                message=conversation.messages[0], context=conversation.messages[1:]
            )
        )
    except Exception as e:
        response = str(e)
    return models.BotMessage(content=response)


@router.post("/ingest_data")
def ingest_data(data: list[models.DataMessage]):
    try:
        docs = list(map(functions.ingest_query, data))
        docs = list(map(pipelines.to_document, docs))
        _ = pipelines.indexing_pipeline.run(docs)
        response = "Data loaded!"
    except Exception as e:
        response = str(e)
    return models.BotMessage(content=response)


@router.post("/summarize")
def summarize(messages: list[models.Message]):
    try:
        response = pipelines.summarizer_pipeline.run(functions.summ_query(messages))
    except Exception as e:
        response = str(e)
    return models.BotMessage(content=response)


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return models.BotMessage(content="There was an error uploading the file")
    finally:
        file.file.close()

    return models.BotMessage(content=f"Successfully uploaded {file.filename}")
