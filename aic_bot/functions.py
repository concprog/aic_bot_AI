from . import models
from typing import Callable


def role_to_pri(role: str, clearances: list[models.ClearanceMapping] = models.clearances):
    clearance = list(filter(lambda c: c.discord_role == role, clearances))
    clearance = min(clearance, key=lambda c: c.priority)
    return clearance.priority
    
def react_to_pri(reactions: set[str], clearances: list[models.ClearanceMapping] = models.clearances):
    clearance = list(filter(lambda c: c.reaction in reactions, clearances))
    clearance = min(clearance, key=lambda c: c.priority)
    return clearance.priority

def rqa_query(message: models.Message, context: list[models.Message]):
    pipeline_inputs: dict[str, dict] | dict[str, str] = {}
    pipeline_inputs["embedder"] = {"text": message.content}
    pipeline_inputs["retriever"]["filters"] = {"field": "meta.clearance", "operator": "<=", "value": role_to_pri(message.discord_role)}
    pipeline_inputs["prompt_builder"] = {"query": message.content, "messages": context}
    return pipeline_inputs

def ingest_query(data: models.DataMessage):
    pipeline_inputs: dict[str, dict] | dict[str, str] = {}
    pipeline_inputs["content"] = data.content
    pipeline_inputs["meta"] = react_to_pri(data.reactions)
    return pipeline_inputs

def summ_query(messages: list[models.Message]):
    pipeline_inputs: dict[str, list[models.Message]] = {}
    pipeline_inputs["messages"] = messages
    return pipeline_inputs

