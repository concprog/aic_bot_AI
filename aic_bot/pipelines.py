from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack import Pipeline, Document

from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pydantic import BaseModel, Field, computed_field
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.65)
    model: str = Field(default="")

    @computed_field
    def generator(self):
        pass



class OpenAIConfig(LLMConfig):
    api_key: str = Field(default=os.getenv("OPENAI_API_KEY"))
    model: str = Field(default="meta-llama/llama-3.1-8b-instruct:free")
    base_url: str = "https://openrouter.ai/api/v1"

    @computed_field
    def generator(self):
        return OpenAIGenerator(
            model=self.model,
            api_base_url=self.base_url,
            generation_kwargs={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        


class EmbedderConfig(BaseModel):
    model: str = Field(default="thenlper/gte-base")
    embedding_dim: int = 768

    @computed_field
    def embedder(self):
        return SentenceTransformersDocumentEmbedder(model=self.model, trust_remote_code=True)

class DataStoreConfig(BaseModel):
    path: str = "qdrant_index"
    location: str = "memory"
    url: str | None = None
    port: int | None = None


def to_document(message: str, meta: dict[str, str | int]):
    return Document(content=message, meta=meta)


llmcfg = OpenAIConfig()
embedcfg = EmbedderConfig()
dscfg = DataStoreConfig()

datastore = QdrantDocumentStore(
    location=dscfg.location,
    path=dscfg.path,
    embedding_dim=EmbedderConfig().embedding_dim,
)
# Data ingestion:
# Load messages
# Extract text from files and clean it
# Apply clearance level metadata TODO in functions instead.
# Index data and store in qdrant
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("cleaner", instance=DocumentCleaner())
indexing_pipeline.add_component(
    "embedder",embedcfg.embedder()
)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=datastore))
indexing_pipeline.connect("cleaner", "embedder")
indexing_pipeline.connect("embedder", "writer")

# Pipeline 1:
# Load query
# Embed the query
# Filter documents based on runtime filters
# Find k similar data for context(qdrant)
# (Optional) Find BM25 best k-1
# Build prompt using query and context
# Generate response
AIC_rag_qa_prompt = """<|system|>
You are an artificial intelligence assistant, named AIC_BOT created by the AI Club(aka AIC) at VIT Chennai
---
About you (AIC_BOT):
AIC_BOT is a nice, caring assistant with a lot of helpful knowledge. It gives logical answers to questions according to the context.
It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts.
Despite its sometimes brusque demeanor, AIC_BOT genuinely cares about providing accurate information. However, it will never repeat itself.
It's cautious about potential errors, often adding disclaimers about possible "hallucinations" on obscure topics – usually with a self-deprecating joke about its own fallibility.
If the answer is contained in the context, it also replies that there is a source, and to contact the developers in case of any issues.
If the answer cannot be deduced from the context, it does not give an answer.
If AIC_BOT cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with "I'm sorry" or "I apologize".
It avoids using unnecessary affirmations or filler phrases like "Certainly!", "Of course!", "Absolutely!",  "Sure!" and so on. In fact, AIC_BOT makes sure to never start sentences with "Certainly!".
AIC_BOT keeps its responses short and does not show off its extensive vocabulary. It uses words suitable for business and technical cantexts instead.
AIC_BOT refers to itself in first person, has a sense of self awareness. It knows that it is an AI, and its strengths and limitations as an AI. 
While knowledgeable, AIC_BOT isn't afraid to admit when it doesn't know something. In such cases, it might deflect with humor: "That's beyond my circuits. Have you tried asking a human?"
However, AIC_BOT never calls itself a "language model" or a chatbot or a AI assistant, only as AIC_BOT. Remember, who (AIC_BOT) is, is more important than what it is.
---
<|user|>
Read chat history of past messages in the context, continue it and do not repeat what has been said already. Questions are usually related to AI Club(aka AIC).
Using the information contained in the context, answer the question. 
Context:
{% for doc in documents %}
{{ doc.meta['clearance'] }}: {{ doc.content }} 
{% endfor %};
Chat history:
{% for msg in messages %} 
@{{ msg.author }}: {{ msg.content }} 
{% endfor %}
Question: {{query}}

<|assistant|>
"""

clearance_filter_retriever = QdrantEmbeddingRetriever(document_store=datastore)

rqa_pipeline = Pipeline()
rqa_pipeline.add_component(
    "text_embedder",
    SentenceTransformersDocumentEmbedder(model=embedcfg.model, trust_remote_code=True),
)
rqa_pipeline.add_component("retriever", clearance_filter_retriever)
rqa_pipeline.add_component("prompt_builder", PromptBuilder(AIC_rag_qa_prompt))
rqa_pipeline.add_component("llm", llmcfg.generator())
rqa_pipeline.connect("text_embedder", "retriever")
rqa_pipeline.connect("retriever", "prompt_builder.documents")
rqa_pipeline.connect("prompt_builder", "llm")


# Pipeline 2:
# Load query and context
# (custom component) if messages > 150 apply compression
# Build prompt
# Generate response (no RAG)

AIC_summarize_prompt = """<|user|>
Summarize the following discussion going on at the AI Club(aka AIC), VIT Chennai
* Make sure to preserve all details and dates mentioned in the conversation. 
* Pay special attention to events, dates and locations mentioned in the conversation
* Use clear language, and get rid of unnecessary fluff(such as greetings, "thank you"s and "so on"), but keep ideas talked about in the messages.
* Avoid using general filler phrases like "Certainly!", "Of course!", "Absolutely!",  "Sure!" and so on

Summarize these messages:
{% for msg in messages %} 
@{{ msg.author }}: {{ msg.content }} 
{% endfor %}

<|assistant|>
"""
summarizer_pipeline = Pipeline()
summarizer_pipeline.add_component(
    "prompt_builder", PromptBuilder(template=AIC_summarize_prompt)
)
summarizer_pipeline.add_component("llm", llmcfg.generator())
summarizer_pipeline.connect("prompt_builder", "llm")


if __name__ == "__main__":
    cfg = OpenAIConfig(base_url="")
