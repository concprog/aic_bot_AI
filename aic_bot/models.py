from pydantic import BaseModel


class Message(BaseModel):
    author: str
    content: str
    discord_role: str

class Conversation(BaseModel):
    messages: list[Message]
    channel: str

class DataMessage(Message):
    reactions: list[str]
    discord_role: str = "" 

class BotMessage(Message):
    author: str = "AIC_BOT"
    discord_role: str = "BOT"  
    
reaction_to_clearance_map: dict[str, str] = {
    "A":"sensitive",
    "B":"internal",
    "C":"public",
    "D":"excluded",
}
    