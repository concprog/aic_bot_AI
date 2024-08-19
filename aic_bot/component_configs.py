from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing import override
from haystack.utils.auth import Secret


class GenerationConfig(BaseModel):
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.65)


class ComponentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field(return_type=dict)
    def component_kwargs(self):
        return {k: v for k, v in self.model_dump().items() if k != "init_component"}


class OpenAIConfig(ComponentConfig):
    api_key: Secret = Field(default=Secret.from_env_var("OPENROUTER_API_KEY"))
    model: str = Field(default="meta-llama/llama-3.1-8b-instruct:free")
    api_base_url: str = "https://openrouter.ai/api/v1"
    generation_kwargs: GenerationConfig = GenerationConfig()
    @computed_field(return_type=dict)
    @override
    def component_kwargs(self):
        return {
            "api_key": self.api_key,
            "api_base_url": self.api_base_url,
            "model": self.model,
            "generation_kwargs": {
                "max_tokens": self.generation_kwargs.max_tokens,
                "temperature": self.generation_kwargs.temperature,
            },
        }


class EmbedderConfig(ComponentConfig):
    model: str = Field(default="thenlper/gte-base")
    embedding_dim: int = 768

    @computed_field(return_type=dict)
    @override
    def component_kwargs(self):
        return {
            "model": self.model,
            "trust_remote_code": True,
        }


class QdrantDataStoreConfig(ComponentConfig):
    path: str = "qdrant_index"
    location: str = "memory"
    url: str | None = None
    port: int = 6333

    @computed_field(return_type=dict)
    @override
    def component_kwargs(self):
        return {
            "location": self.location,
        }
