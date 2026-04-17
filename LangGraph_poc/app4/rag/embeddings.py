import os
from typing import List
from langchain_core.embeddings import Embeddings
from models.ollama_client import embed as ollama_embed, OLLAMA_EMBED_MODEL


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = OLLAMA_EMBED_MODEL):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._embed_many(texts)
        )

    def embed_query(self, text: str) -> List[float]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            ollama_embed(text, model=self.model)
        )

    async def _embed_many(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            emb = await ollama_embed(text, model=self.model)
            results.append(emb)
        return results
