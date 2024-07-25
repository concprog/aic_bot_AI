from pydantic import BaseModel


class Message(BaseModel):
    author: str
    content: str
    discord_role: str
    reference: Message | None = None

class Conversation(BaseModel):
    messages: list[Message]
    channel: str
