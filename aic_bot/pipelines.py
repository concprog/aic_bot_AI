from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.embedders.jina import (
    JinaTextEmbedder,
    JinaDocumentEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder


from haystack import Pipeline, Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from .component_configs import *

from dotenv import load_dotenv

load_dotenv()

llmconfig = OpenAIConfig()
embedconfig = EmbedderConfig()
documentstoreconfig = QdrantDataStoreConfig()

datastore = QdrantDocumentStore(
    embedding_dim=embedconfig.embedding_dim, **documentstoreconfig.component_kwargs
)

# Data ingestion:
# Load messages
# Extract text from files and clean it
# Apply clearance level metadata DONE in functions instead.
# Index data and store in qdrant

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component(
    "splitter", DocumentSplitter(split_by="passage", split_length=1)
)
indexing_pipeline.add_component("embedder", JinaDocumentEmbedder())
indexing_pipeline.add_component(
    "writer", DocumentWriter(document_store=datastore, policy=DuplicatePolicy.OVERWRITE)
)
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
Read through the chat history of past messages in the context, talk and do not repeat what has been said already. Questions are usually related to AI Club(aka AIC).
Using the information contained in the context, answer the question. 
Context:
{% for doc in documents %}
{{ doc.meta['clearance'] }}: {{ doc.content }} 
{% endfor %};
Chat history:
{% for msg in messages %} 
@{{ msg.author }}: {{ msg.content }} 
{% endfor %}
Now, tell me: {{query}}
<|assistant|>
"""
rqa_pipeline = Pipeline()
rqa_pipeline.add_component("embedder", JinaTextEmbedder())
rqa_pipeline.add_component(
    "retriever", QdrantEmbeddingRetriever(document_store=datastore)
)
rqa_pipeline.add_component("prompt_builder", PromptBuilder(AIC_rag_qa_prompt))
rqa_pipeline.add_component("llm", OpenAIGenerator(**llmconfig.component_kwargs))
rqa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
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
summarizer_pipeline.add_component("llm", OpenAIGenerator(**llmconfig.component_kwargs))
summarizer_pipeline.connect("prompt_builder", "llm")
