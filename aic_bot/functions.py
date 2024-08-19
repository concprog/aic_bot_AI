from . import models


def get_models():
    return models


def message_to_str(message: models.Message):
    return f"{message.author}: {message.content}"


def message_to_dict(message: models.Message):
    return message.model_dump()


def role_to_pri(
    role: str, clearances: list[models.ClearanceMapping] = models.clearances
):
    clearance = list(filter(lambda c: c.discord_role == role, clearances))
    clearance = min(clearance, key=lambda c: c.priority)
    return clearance.priority


def react_to_pri(
    reactions: set[str], clearances: list[models.ClearanceMapping] = models.clearances
):
    clearance = list(filter(lambda c: c.reaction in reactions, clearances))
    clearance = min(clearance, key=lambda c: c.priority)
    return clearance.priority


def rqa_query(message: models.Message, context: list[models.Message]):
    pipeline_inputs: dict[str, dict] = {}
    pipeline_inputs["embedder"] = {"text": message.content}
    pipeline_inputs["retriever"] = {
        "filters": {
            "field": "meta.clearance",
            "operator": "<=",
            "value": role_to_pri(message.discord_role),
        }
    }
    pipeline_inputs["prompt_builder"] = {"query": message.content, "messages": context}
    return pipeline_inputs


def ingest_query(data: models.DataMessage):
    pipeline_inputs: dict[str, dict] | dict[str, str] = {}
    pipeline_inputs["content"] = data.content
    pipeline_inputs["meta"] = {"clearance": react_to_pri(data.reactions)}
    return pipeline_inputs


def summ_query(messages: list[models.Message]):
    pipeline_inputs: dict[str, list[models.Message]] = {}
    pipeline_inputs["messages"] = list(map(message_to_dict, messages))
    return pipeline_inputs
