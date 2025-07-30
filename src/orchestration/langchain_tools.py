"""
LangChain Tool Wrappers for Research Assistant

This module wraps existing research tools as LangChain tools to demonstrate
LangChain integration without modifying the original tool implementations.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field

try:
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Create dummy classes for type hints
    class BaseTool:
        pass

    class CallbackManagerForToolRun:
        pass


from src.tools.web_search import WebSearchTool
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.csv_analysis import CSVAnalysisTool
from src.models.model_builder import ModelBuilder

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(description="The search query to execute")
    max_results: int = Field(
        default=5, description="Maximum number of results to return"
    )


class ArxivSearchInput(BaseModel):
    """Input schema for ArXiv search tool."""

    query: str = Field(
        description="The research query to search for in academic papers"
    )
    max_results: int = Field(
        default=5, description="Maximum number of papers to return"
    )


class CSVAnalysisInput(BaseModel):
    """Input schema for CSV analysis tool."""

    query: str = Field(description="The research question to analyze CSV data for")


class LangChainWebSearchTool(BaseTool):
    """LangChain wrapper for WebSearchTool."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information. Use this for general research questions, "
        "news, current events, and factual information from reliable web sources."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._web_tool = WebSearchTool()

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute web search."""
        try:
            results = self._web_tool.search(query, max_results=max_results)

            if not results:
                return f"No web results found for query: {query}"

            # Format results for LangChain
            formatted_results = []
            for i, result in enumerate(results[:max_results], 1):
                formatted_results.append(
                    f"{i}. **{result.title}**\n"
                    f"   Source: {result.url}\n"
                    f"   Content: {result.content[:200]}...\n"
                )

            return f"Web Search Results for '{query}':\n\n" + "\n".join(
                formatted_results
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Web search failed: {str(e)}"


class LangChainArxivSearchTool(BaseTool):
    """LangChain wrapper for ArxivSearchTool."""

    name: str = "arxiv_search"
    description: str = (
        "Search academic papers on ArXiv. Use this for scientific research questions "
        "that would benefit from peer-reviewed academic literature and recent research papers."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._arxiv_tool = ArxivSearchTool()

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute ArXiv search."""
        try:
            results = self._arxiv_tool.search(query, max_results=max_results)

            if not results:
                return f"No academic papers found for query: {query}"

            # Format results for LangChain
            formatted_results = []
            for i, result in enumerate(results[:max_results], 1):
                authors = ", ".join(result.authors[:3])  # Show first 3 authors
                if len(result.authors) > 3:
                    authors += " et al."

                formatted_results.append(
                    f"{i}. **{result.title}**\n"
                    f"   Authors: {authors}\n"
                    f"   Published: {result.published_date[:10]}\n"
                    f"   URL: {result.url}\n"
                    f"   Abstract: {result.abstract[:200]}...\n"
                )

            return f"ArXiv Search Results for '{query}':\n\n" + "\n".join(
                formatted_results
            )

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return f"ArXiv search failed: {str(e)}"


class LangChainCSVAnalysisTool(BaseTool):
    """LangChain wrapper for CSV Analysis Tool."""

    name: str = "csv_analysis"
    description: str = (
        "Analyze CSV datasets to find statistical insights and data relevant to research questions. "
        "Use this when you need quantitative data, statistics, or tabular analysis."
    )
    args_schema: Type[BaseModel] = CSVAnalysisInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            model_builder = ModelBuilder()
            self._csv_tool = CSVAnalysisTool(model_builder=model_builder)
        except Exception as e:
            logger.warning(f"CSV tool initialization failed: {e}")
            self._csv_tool = None

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute CSV analysis."""
        if not self._csv_tool:
            return "CSV analysis tool not available (missing dependencies or data)"

        try:
            analysis = self._csv_tool.analyze_for_query(query)

            if not analysis.get("results"):
                return f"No relevant CSV data found for query: {query}"

            # Format analysis results
            summary = analysis.get("summary", "No summary available")
            datasets_analyzed = analysis.get("datasets_analyzed", 0)

            formatted_results = [
                f"CSV Analysis Results for '{query}':",
                f"Datasets analyzed: {datasets_analyzed}",
                f"\nSummary:\n{summary}",
            ]

            # Add key insights from individual datasets
            for result in analysis.get("results", [])[:3]:  # Show top 3 datasets
                if result.get("key_insights"):
                    formatted_results.append(f"\nDataset: {result['dataset']}")
                    for insight in result["key_insights"][:3]:  # Top 3 insights
                        formatted_results.append(f"  • {insight}")

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"CSV analysis failed: {e}")
            return f"CSV analysis failed: {str(e)}"


def create_langchain_tools() -> List[BaseTool]:
    """Create and return all available LangChain tools."""
    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain not available")
        return []

    tools = []

    # Always include web search as it's most broadly applicable
    try:
        tools.append(LangChainWebSearchTool())
        logger.info("Web search tool added to LangChain toolkit")
    except Exception as e:
        logger.error(f"Failed to create web search tool: {e}")

    # Add ArXiv search
    try:
        tools.append(LangChainArxivSearchTool())
        logger.info("ArXiv search tool added to LangChain toolkit")
    except Exception as e:
        logger.error(f"Failed to create ArXiv search tool: {e}")

    # Add CSV analysis if available
    try:
        tools.append(LangChainCSVAnalysisTool())
        logger.info("CSV analysis tool added to LangChain toolkit")
    except Exception as e:
        logger.error(f"Failed to create CSV analysis tool: {e}")

    logger.info(f"Created {len(tools)} LangChain tools")
    return tools
