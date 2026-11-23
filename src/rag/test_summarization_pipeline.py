
"""Unit tests for hierarchical document summarization."""

import pytest

from src.rag.context_builder import ContextBuilder
from src.rag.models import (
    EvidenceOrigin,
    RequestMode,
    ScopeType,
    SummaryStyle,
)
from src.rag.prompt_builder import PromptPackage
from src.rag.summarization_pipeline import SummarizationPipeline


class FakePromptBuilder:
    """Capture summary prompt requests for deterministic testing."""

    def __init__(self):
        """Initialize the prompt history."""

        self.calls = []

    def build_document_summary(self, context_package, summary_style):
        """Record the context and requested summary style."""

        self.calls.append(
            {
                "context_package": context_package,
                "summary_style": summary_style,
            }
        )

        return PromptPackage(
            system_prompt="Test summary instructions",
            user_prompt=context_package.context_text,
        )


class FakeLLMGenerator:
    """Return configured summaries without calling Ollama."""

    def __init__(self, answers=None):
        """Initialize the configured model responses."""

        self.answers = answers or [
            "The document describes its main policy. [S1]"
        ]
        self.calls = []

    def generate(self, prompt_package):
        """Record the prompt and return the next configured response."""

        self.calls.append(prompt_package)

        if not self.answers:
            raise AssertionError("No configured fake model response remains.")

        return self.answers.pop(0)


class FakeDocumentChunkLoader:
    """Return configured document chunks and record loader requests."""

    def __init__(self, chunks):
        """Initialize the loader with deterministic document chunks."""

        self.chunks = chunks
        self.calls = []

    def __call__(self, document_ids, scope_type, scope_id):
        """Record requested documents and return their chunks."""

        self.calls.append(
            {
                "document_ids": document_ids,
                "scope_type": scope_type,
                "scope_id": scope_id,
            }
        )

        return self.chunks


def make_chunk(
    chunk_id="chunk-1",
    document_id="document-1",
    text="Employees receive 20 days of annual leave.",
    chunk_index=0,
    scope_type="personal",
    scope_id="user-1",
):
    """Create a scope-aware document chunk for testing."""

    return {
        "chunk_id": chunk_id,
        "text": text,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source": f"{document_id}.pdf",
            "filename": f"{document_id}.pdf",
            "chunk_index": chunk_index,
            "page_numbers": [chunk_index + 1],
            "heading_path": ["Policy"],
            "scope_type": scope_type,
            "scope_id": scope_id,
        },
    }


def make_pipeline(
    chunks=None,
    answers=None,
    max_batch_chars=10000,
):
    """Create a summarization pipeline with fake external dependencies."""

    loader = FakeDocumentChunkLoader(
        chunks if chunks is not None else [make_chunk()]
    )
    prompt_builder = FakePromptBuilder()
    llm_generator = FakeLLMGenerator(answers=answers)

    pipeline = SummarizationPipeline(
        document_chunk_loader=loader,
        context_builder=ContextBuilder(),
        prompt_builder=prompt_builder,
        llm_generator=llm_generator,
        max_batch_chars=max_batch_chars,
    )

    return pipeline, loader, prompt_builder, llm_generator


def test_summarizes_one_document():
    """Verify that one selected document produces a cited summary."""

    pipeline, loader, _, _ = make_pipeline()

    response = pipeline.summarize(
        document_ids=["document-1"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert response.mode == RequestMode.DOCUMENT_SUMMARY
    assert response.general_knowledge_used is False
    assert response.insufficient_evidence is False
    assert "[S1]" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].document_id == "document-1"
    assert loader.calls[0]["document_ids"] == ["document-1"]



def test_summarizes_multiple_selected_documents():
    """Verify that chunks from all selected documents are included."""

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            document_id="document-1",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            document_id="document-2",
            chunk_index=0,
            text="Employees must obtain manager approval for leave.",
        ),
    ]

    pipeline, loader, prompt_builder, _ = make_pipeline(
        chunks=chunks,
        answers=[
            "The documents describe annual leave entitlement "
            "and manager approval requirements. [S1] [S2]"
        ],
    )

    response = pipeline.summarize(
        document_ids=["document-1", "document-2"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert len(loader.calls) == 1

    # Both selected documents must be present in the generated context.
    context = prompt_builder.calls[0]["context_package"].context_text

    assert "Employees receive 20 days of annual leave." in context
    assert "Employees must obtain manager approval for leave." in context

    # Both sources must be cited in the final summary.
    assert len(response.sources) == 2

    assert {
        source.document_id for source in response.sources
    } == {"document-1", "document-2"}


def test_loader_receives_requested_scope():
    """Verify that document loading receives the requested scope."""

    pipeline, loader, _, _ = make_pipeline()

    pipeline.summarize(
        document_ids=["document-1"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert loader.calls[0]["scope_type"] == "personal"
    assert loader.calls[0]["scope_id"] == "user-1"


@pytest.mark.parametrize(
    "summary_style",
    [
        SummaryStyle.COMPREHENSIVE,
        SummaryStyle.EXECUTIVE,
        SummaryStyle.SECTION_WISE,
        SummaryStyle.KEY_POINTS,
        SummaryStyle.ACTION_ITEMS,
    ],
)
def test_supports_all_summary_styles(summary_style):
    """Verify that all supported styles reach the prompt builder."""

    pipeline, _, prompt_builder, _ = make_pipeline()

    pipeline.summarize(
        document_ids=["document-1"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
        summary_style=summary_style,
    )

    assert prompt_builder.calls[0]["summary_style"] == summary_style


def test_uses_hierarchical_summarization_for_long_documents():
    """Verify that long documents are summarized in multiple batches."""

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            document_id="document-1",
            text="A" * 30,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            document_id="document-1",
            text="B" * 30,
            chunk_index=1,
        ),
    ]

    pipeline, _, prompt_builder, llm_generator = make_pipeline(
        chunks=chunks,
        answers=[
            "First section summary. [S1]",
            "Second section summary. [S2]",
            "Combined document summary. [S1] [S2]",
        ],
        max_batch_chars=40,
    )

    response = pipeline.summarize(
        document_ids=["document-1"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
        summary_style=SummaryStyle.EXECUTIVE,
    )

    assert len(llm_generator.calls) == 3
    assert len(prompt_builder.calls) == 3

    # Intermediate summaries preserve details.
    assert prompt_builder.calls[0]["summary_style"] == (
        SummaryStyle.COMPREHENSIVE
    )
    assert prompt_builder.calls[1]["summary_style"] == (
        SummaryStyle.COMPREHENSIVE
    )

    # Final synthesis applies the requested style.
    assert prompt_builder.calls[2]["summary_style"] == (
        SummaryStyle.EXECUTIVE
    )

    assert len(response.sources) == 2
    assert "[S1]" in response.answer
    assert "[S2]" in response.answer


def test_assigns_unique_citations_across_batches():
    """Verify that citation identifiers do not collide between batches."""

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            text="A" * 30,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            text="B" * 30,
            chunk_index=1,
        ),
    ]

    pipeline, _, prompt_builder, _ = make_pipeline(
        chunks=chunks,
        answers=[
            "First summary. [S1]",
            "Second summary. [S2]",
            "Final summary. [S1] [S2]",
        ],
        max_batch_chars=40,
    )

    pipeline.summarize(
        document_ids=["document-1"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    second_batch_context = (
        prompt_builder.calls[1]["context_package"].context_text
    )

    assert "[S2]" in second_batch_context
    assert "[S1]" not in second_batch_context


def test_rejects_unknown_intermediate_citation():
    """Verify that invalid citations in a batch summary are rejected."""

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            text="A" * 30,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            text="B" * 30,
            chunk_index=1,
        ),
    ]

    pipeline, _, _, _ = make_pipeline(
        chunks=chunks,
        answers=[
            "First summary. [S9]",
        ],
        max_batch_chars=40,
    )

    with pytest.raises(RuntimeError, match="unknown citations"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_intermediate_summary_without_citations():
    """Verify that intermediate summaries must retain source references."""

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            text="A" * 30,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            text="B" * 30,
            chunk_index=1,
        ),
    ]

    pipeline, _, _, _ = make_pipeline(
        chunks=chunks,
        answers=["An uncited intermediate summary."],
        max_batch_chars=40,
    )

    with pytest.raises(RuntimeError, match="no source citations"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_unknown_final_citation():
    """Verify that the final summary cannot cite an unknown source."""

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            text="A" * 30,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            text="B" * 30,
            chunk_index=1,
        ),
    ]

    pipeline, _, _, _ = make_pipeline(
        chunks=chunks,
        answers=[
            "First summary. [S1]",
            "Second summary. [S2]",
            "Final summary. [S9]",
        ],
        max_batch_chars=40,
    )

    with pytest.raises(RuntimeError, match="unknown citations"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_final_summary_without_citations():
    """Verify that the final generated summary must contain citations."""

    pipeline, _, _, llm_generator = make_pipeline(
        answers=["A summary without source references."]
    )

    with pytest.raises(RuntimeError, match="no source citations"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )

    assert len(llm_generator.calls) == 1


def test_rejects_chunks_from_unselected_documents():
    """Verify that chunks outside the selected document IDs are rejected."""

    pipeline, _, _, _ = make_pipeline(
        chunks=[
            make_chunk(
                chunk_id="chunk-1",
                document_id="different-document",
            )
        ]
    )

    with pytest.raises(ValueError, match="selected document"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_wrong_scope_type():
    """Verify that chunks from another scope type cannot be summarized."""

    pipeline, _, _, _ = make_pipeline(
        chunks=[
            make_chunk(
                scope_type="project",
                scope_id="user-1",
            )
        ]
    )

    with pytest.raises(ValueError, match="unexpected scope type"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_wrong_scope_id():
    """Verify that chunks from another scope ID cannot be summarized."""

    pipeline, _, _, _ = make_pipeline(
        chunks=[
            make_chunk(scope_id="another-user")
        ]
    )

    with pytest.raises(ValueError, match="unexpected scope ID"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_inconsistent_scope_metadata():
    """Verify that metadata cannot contradict the chunk's scope."""

    chunk = make_chunk()
    chunk["metadata"]["scope_id"] = "another-user"

    pipeline, _, _, _ = make_pipeline(chunks=[chunk])

    with pytest.raises(ValueError, match="inconsistent scope metadata"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_missing_document_chunks():
    """Verify that every selected document must have retrieved chunks."""

    pipeline, _, _, _ = make_pipeline(
        chunks=[
            make_chunk(document_id="document-1")
        ]
    )

    with pytest.raises(ValueError, match="document-2"):
        pipeline.summarize(
            document_ids=["document-1", "document-2"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_empty_loader_results():
    """Verify that an empty document selection result is rejected."""

    pipeline, _, _, _ = make_pipeline(chunks=[])

    with pytest.raises(ValueError, match="No chunks were found"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_duplicate_chunk_ids():
    """Verify that duplicate chunks cannot enter summarization."""

    chunk = make_chunk()

    pipeline, _, _, _ = make_pipeline(
        chunks=[chunk, chunk.copy()]
    )

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_rejects_chunk_larger_than_batch_limit():
    """Verify that an oversized chunk is not silently truncated."""

    pipeline, _, _, _ = make_pipeline(
        chunks=[
            make_chunk(text="A" * 100)
        ],
        max_batch_chars=50,
    )

    with pytest.raises(ValueError, match="exceeds the configured"):
        pipeline.summarize(
            document_ids=["document-1"],
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


@pytest.mark.parametrize(
    "document_ids,scope_type,scope_id,summary_style,error",
    [
        ([], ScopeType.PERSONAL, "user-1", SummaryStyle.COMPREHENSIVE, ValueError),
        ([""], ScopeType.PERSONAL, "user-1", SummaryStyle.COMPREHENSIVE, ValueError),
        (["document-1", "document-1"], ScopeType.PERSONAL, "user-1", SummaryStyle.COMPREHENSIVE, ValueError),
        (["document-1"], "personal", "user-1", SummaryStyle.COMPREHENSIVE, TypeError),
        (["document-1"], ScopeType.PERSONAL, "", SummaryStyle.COMPREHENSIVE, ValueError),
        (["document-1"], ScopeType.PERSONAL, "user-1", "invalid", TypeError),
    ],
)
def test_rejects_invalid_inputs(
    document_ids,
    scope_type,
    scope_id,
    summary_style,
    error,
):
    """Verify that invalid document, scope, and style inputs are rejected."""

    pipeline, _, _, _ = make_pipeline()

    with pytest.raises(error):
        pipeline.summarize(
            document_ids=document_ids,
            scope_type=scope_type,
            scope_id=scope_id,
            summary_style=summary_style,
        )


def test_summary_is_document_grounded():
    """Verify that the final response identifies document evidence only."""

    pipeline, _, _, _ = make_pipeline()

    response = pipeline.summarize(
        document_ids=["document-1"],
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert response.sections[0].origin == EvidenceOrigin.DOCUMENT
    assert response.general_knowledge_used is False
    assert response.sections[0].citations == ["S1"]


def test_rejects_non_callable_loader():
    """Verify that a callable document loader is required."""

    with pytest.raises(TypeError, match="must be callable"):
        SummarizationPipeline(document_chunk_loader=None)


def test_rejects_invalid_batch_character_limit():
    """Verify that the batch character limit must be positive."""

    with pytest.raises(ValueError, match="positive integer"):
        SummarizationPipeline(
            document_chunk_loader=lambda ids, scope, scope_id: [],
            max_batch_chars=0,
        )
