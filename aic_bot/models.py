from pydantic import BaseModel


class User(BaseModel):
    name: str
    password: str


class Message(BaseModel):
    author: str
    discord_role: str
    content: str


class Conversation(BaseModel):
    messages: list[Message]
    channel: str


class DataMessage(Message):
    reactions: set[str] = set()
    discord_role: str = ""


class BotMessage(Message):
    author: str = "AIC_BOT"
    discord_role: str = "BOT"


class ClearanceMapping(BaseModel):
    name: str
    priority: int
    discord_role: str
    reaction: str


clearances = [
    ClearanceMapping(name="sensitive", discord_role="Core", reaction="A", priority=0),
    ClearanceMapping(name="internal", discord_role="Member", reaction="B", priority=1),
    ClearanceMapping(name="external", discord_role="Guest", reaction="C", priority=2),
    ClearanceMapping(name="excluded", discord_role="Dev", reaction="D", priority=-1),
]
