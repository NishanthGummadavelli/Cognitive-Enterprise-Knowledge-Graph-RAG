
"""Route RAG requests to the appropriate processing pipeline."""

from src.rag.knowledge_augmented_pipeline import KnowledgeAugmentedPipeline
from src.rag.models import (
    RAGRequest,
    RAGResponse,
    RequestMode,
)
from src.rag.rag_pipeline import RAGPipeline
from src.rag.summarization_pipeline import SummarizationPipeline


class RequestRouter:
    """Dispatch validated RAG requests to their corresponding pipelines."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        summarization_pipeline: SummarizationPipeline,
        knowledge_augmented_pipeline: KnowledgeAugmentedPipeline,
    ) -> None:
        """Initialize the router with the three processing pipelines."""

        self.rag_pipeline = rag_pipeline
        self.summarization_pipeline = summarization_pipeline
        self.knowledge_augmented_pipeline = knowledge_augmented_pipeline

    def process(self, request: RAGRequest) -> RAGResponse:
        """Route a request according to its selected processing mode."""

        if not isinstance(request, RAGRequest):
            raise TypeError("request must be a RAGRequest instance.")

        if request.mode == RequestMode.DOCUMENT_QA:
            response = self.rag_pipeline.answer(
                question=request.question,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
                top_k=request.top_k,
            )

        elif request.mode == RequestMode.DOCUMENT_SUMMARY:
            response = self.summarization_pipeline.summarize(
                document_ids=request.document_ids,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
                summary_style=request.summary_style,
            )

        elif request.mode == RequestMode.KNOWLEDGE_AUGMENTED_QA:
            response = self.knowledge_augmented_pipeline.answer(
                question=request.question,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
                top_k=request.top_k,
            )

        else:
            raise ValueError(f"Unsupported request mode: {request.mode}")

        if not isinstance(response, RAGResponse):
            raise TypeError(
                "The selected pipeline must return a RAGResponse instance."
            )

        return response
