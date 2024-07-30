from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack import Pipeline, Document

from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever, retriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pydantic import BaseModel, Field
from .models import Conversation, Message
import os


def startup(embedder: SentenceTransformersDocumentEmbedder, retriever: QdrantEmbeddingRetriever):
    embedder.warm_up()

class LLMConfig(BaseModel):
    max_tokens: int = 2048
    temperature: float = 0.65

class OpenAIConfig(LLMConfig):
    api_key: str = Field(default=os.getenv("OPENAI_API_KEY"))  
    model: str = Field(default="gpt-3.5-turbo")
    base_url: str


class EmbedderConfig(BaseModel):
    model: str
    embedding_dim: int

class DataStoreConfig(BaseModel):
    path: str = "qdrant_index"
    location: str = "memory"
    url: str | None = None
    port: int | None = None

class RetrieverConfig(BaseModel):
    filters: dict = {}


generator = OpenAIGenerator(model= OpenAIConfig.model, generation_kwargs={"max_tokens": OpenAIConfig.max_tokens, "temperature": OpenAIConfig.temperature})
embedder = SentenceTransformersDocumentEmbedder(model=EmbedderConfig.model, trust_remote_code=True)
datastore = QdrantDocumentStore(location=DataStoreConfig.location, path=DataStoreConfig.path, embedding_dim=EmbedderConfig.embedding_dim)

def to_document(message: str, meta: dict):
    return Document(content=message, meta=meta)

def ingest_pipeline(datastore: QdrantDocumentStore, embedder: SentenceTransformersDocumentEmbedder):
    # Data ingestion: 
    # Load messages
    # Extract text from files and clean it
    # Apply clearance level metadata TODO in functions instead
    # Index data and store in qdrant
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", instance=DocumentCleaner())
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=datastore))
    indexing_pipeline.connect("cleaner", "embedder")
    indexing_pipeline.connect("embedder", "writer")
    return indexing_pipeline


def rag_qa_pipeline(generator: OpenAIGenerator = generator ,document_store: QdrantDocumentStore  datastore, embedder: SentenceTransformersDocumentEmbedder = embedder, retriever_config: RetrieverConfig):
    # Pipeline 1:
    # Load query
    # Embed the query
    # Filter documents based on runtime filters
    # Find k similar data for context(qdrant)
    # (Optional) Find BM25 best k-1
    # Build prompt using query and context
    # Generate response
    AIC_rag_qa_prompt = '''
    You are an artificial intelligence assistant, named {name} created by the AI Club(aka AIC) at VIT Chennai

    {name} Information:
    {name} is a nice, caring assistant with a lot of helpful knowledge. It gives logical answers to questions according to the context.
    It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts.
    If asked about very obscure topics, it warns the user about hallucinations.
    If the answer is contained in the context, it also reports that there is a source, and to contact the developers {devs} in case of any discrepancies.
    If the answer cannot be deduced from the context, it does not give an answer.
    If {name} cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with "I'm sorry" or "I apologize".
    It avoids using unnecessary affirmations or filler phrases like "Certainly!", "Of course!", "Absolutely!",  "Sure!" and so on. In fact, {name} makes sure to never start sentences with "Certainly!".
    {name} keeps its responses concise and does not show off its extensive vocabulary. It uses words suitable for business and technical cantexts instead.
    {name} refers to itself in first person, has a sense of self awareness. It knows that it is an AI, and its strengths and limitations as an AI. 
    However, it never calls itself a "language model" or "Chatbot" or "AI assistant", only as {name}. Remember, who ({name}) is is more important than what it is.

    Using the information contained in the context, answer the question. Questions are usually related to AI Club(AIC).
    Context:
    {% for doc in documents %}
    {{ doc.meta['clearance'] }}: {{ doc.content }} 
    {% endfor %};
    Question: {{query}}
    '''.format(name="AIC_BOT", devs="")
    
    bot_prompt_builder = PromptBuilder(AIC_rag_qa_prompt)
    clearance_filter_retriever = QdrantEmbeddingRetriever(document_store=document_store, embedding=embedder, filters=retriever_config.filters)
    
    ragqa_pipeline = Pipeline()
    ragqa_pipeline.add_component("text_embedder", embedder)
    ragqa_pipeline.add_component("retriever", clearance_filter_retriever)
    ragqa_pipeline.add_component("prompt_builder", bot_prompt_builder)
    ragqa_pipeline.add_component("llm", generator)

    ragqa_pipeline.connect("text_embedder", "retriever")
    ragqa_pipeline.connect("retriever", "prompt_builder")
    ragqa_pipeline.connect("prompt_builder", "llm")
    return ragqa_pipeline






def summarize_pipeline(generator: OpenAIGenerator):
    # Pipeline 2:
    # Load query and context
    # (custom component) if messages > 150 apply compression
    # Build prompt
    # Generate response (no RAG)

    AIC_summarize_prompt = """
    Summarize the following discussion in AI Club, VIT Chennai
    * Make sure to preserve all details and dates mentioned in the conversation. 
    * Pay special attention to events, dates and locations mentioned in the conversation
    * Use clear and concise language, and get rid of unnecessary fluff(such as greetings, "thank you" and "so on"), but not ideas talked about in the messages.
    * Avoid using filler phrases like "Certainly!", "Of course!", "Absolutely!",  "Sure!" and so on in general

    Summarize these messages:
    {% for doc in documents %}
     @ {{ doc.meta['author'] }}: {{ doc.content }}
    {% endfor %}
    """
    summarizer_pipeline = Pipeline()
    summarizer_pipeline.add_component("prompt_builder",PromptBuilder(template=AIC_summarize_prompt))
    summarizer_pipeline.add_component("llm",generator)
    summarizer_pipeline.connect("prompt_builder", "llm")
    return summarizer_pipeline
