import os
import tempfile
from typing import List
from unittest.mock import patch

import pytest

# Bring src onto the path if tests are executed from repo root
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.tools.web_search import WebSearchTool, SearchResult  # noqa: E402
from src.tools.vector_db import VectorDBTool  # noqa: E402


class DummyModel:
    """Tiny stand-in that mimics the interface of ModelBuilder.build().run()."""

    def run(self, prompt: str):  # type: ignore[override]
        # For test purposes we just echo a static answer
        return "dummy insight"


# ---------------------------------------------------------------------------
# WebSearchTool tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def stubbed_web_tool(monkeypatch):
    """Return a WebSearchTool instance where external dependencies are stubbed."""
    # Instantiate without API key – avoids hitting Tavily.
    tool = WebSearchTool(api_key=None)

    # Patch the private _execute_search to avoid real HTTP calls
    dummy_results: List[SearchResult] = [
        SearchResult(
            title="Test Article 1",
            url="https://example.com/1",
            content="Quantum computing breakthrough in 2024 …",
            score=0.9,
            source_type="web",
        ),
        SearchResult(
            title="Test Article 2",
            url="https://example.com/2",
            content="Further advances in quantum hardware …",
            score=0.85,
            source_type="web",
        ),
    ]

    monkeypatch.setattr(
        WebSearchTool,
        "_execute_search",
        lambda self, sq: dummy_results,  # type: ignore[lambda-assignment]
    )

    # Replace LLM model with a minimal stub so extract_key_insights works offline
    tool.model = DummyModel()
    return tool


def test_web_search_returns_expected_results(stubbed_web_tool):
    """`search` should forward to `_execute_search` and return the dummy list."""
    results = stubbed_web_tool.search("quantum computing")
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)


def test_extract_key_insights_produces_dict(stubbed_web_tool):
    """`extract_key_insights` should include per-result insights and confidence score."""
    dummy_results = stubbed_web_tool.search("quantum computing")
    insights = stubbed_web_tool.extract_key_insights(dummy_results)

    # Expect one numbered key per result plus a confidence score field
    assert "1" in insights and "2" in insights
    assert insights["1"]["insights"] == "dummy insight"
    assert pytest.approx(insights["confidence_score"], rel=1e-3) == pytest.approx(
        (dummy_results[0].score + dummy_results[1].score) / 2,
        rel=1e-3,
    )


# ---------------------------------------------------------------------------
# VectorDBTool tests
# ---------------------------------------------------------------------------

def test_vector_db_add_and_similarity_search(tmp_path):
    """VectorDBTool with the dummy embedding provider should round-trip documents."""
    db_dir = tmp_path / "vectordb"
    vdb = VectorDBTool(
        persist_directory=str(db_dir),
        embedding_provider="dummy",  # avoids any external API
    )

    texts = [
        "Quantum computing is the future of high-performance computation.",
        "Malaria vaccines have reduced mortality in sub-Saharan Africa.",
    ]
    metadatas = [{"topic": "quantum"}, {"topic": "health"}]
    vdb.add_texts(texts, metadatas=metadatas)

    results = vdb.similarity_search("quantum computing", k=1)

    # We inserted two docs, should get a matching metadata back
    assert len(results) == 1
    assert any(r["metadata"]["topic"] == "quantum" for r in results)


# ---------------------------------------------------------------------------
# ArxivSearchTool tests (including PDF handling)
# ---------------------------------------------------------------------------

from src.tools.arxiv_search import ArxivSearchTool, ArxivResult  # noqa: E402
import src.tools.arxiv_search as arxiv_mod  # noqa: E402


def _dummy_arxiv_results() -> List[ArxivResult]:
    return [
        ArxivResult(
            title="Quantum Paper 1",
            url="https://arxiv.org/abs/1234.5678",
            abstract="A study on quantum supremacy …",
            authors=["A. Author"],
            published_date="2024-01-01",
            pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
            categories=["cs.QC"],
            score=0.95,
            metadata={},
        )
    ]


@pytest.fixture()
def stubbed_arxiv_tool(monkeypatch):
    tool = ArxivSearchTool()
    # Stub out the network-bound methods
    monkeypatch.setattr(ArxivSearchTool, "_execute_search", lambda self, q: _dummy_arxiv_results())
    monkeypatch.setattr(ArxivSearchTool, "_download_pdf_content", lambda self, url: "dummy pdf text")
    # Ensure the guard in _enhance_results_with_pdf_content passes
    monkeypatch.setattr(arxiv_mod, "PDF_AVAILABLE", True)
    return tool


def test_arxiv_search_returns_result_with_full_text(stubbed_arxiv_tool):
    results = stubbed_arxiv_tool.search("quantum", download_pdfs=True)
    assert len(results) == 1
    assert results[0].full_text == "dummy pdf text"


# ---------------------------------------------------------------------------
# CSVAnalysisTool tests
# ---------------------------------------------------------------------------

from src.tools.csv_analysis import CSVAnalysisTool  # noqa: E402
import pandas as pd  # CSV test relies on pandas which is already a dependency


def test_csv_analysis_basic(tmp_path):
    # Create a tiny CSV file
    df = pd.DataFrame({"value1": [1, 2, 3, 4], "value2": [10, 20, 30, 40]})
    csv_file = tmp_path / "sample.csv"
    df.to_csv(csv_file, index=False)

    # Create descriptions CSV expected by the tool
    desc_file = tmp_path / "descriptions.csv"
    pd.DataFrame({"filename": ["sample.csv"], "description": ["sample quantum data"]}).to_csv(desc_file, index=False)

    tool = CSVAnalysisTool(model_builder=None)
    # Redirect internal paths to the temporary directory
    tool.csv_dir = str(tmp_path)
    tool.descriptions_file = str(desc_file)
    tool.dataset_descriptions = {"sample.csv": "sample quantum data"}

    analysis = tool.analyze_for_query("quantum", max_datasets=1)
    assert analysis["datasets_analyzed"] == 1
    assert len(analysis["results"]) == 1
    # Ensure summary text mentions the dataset name
    assert "sample.csv" in analysis["results"][0]["dataset"]
