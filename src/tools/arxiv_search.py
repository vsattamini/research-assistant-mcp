"""
ArXiv Search Tool for Research Assistant

This module provides academic paper search capabilities using the ArXiv API
to find relevant research papers and publications.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import feedparser
import requests
import tempfile
import io

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from src.models.model_builder import ModelBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArxivResult:
    """Represents a single ArXiv search result."""

    title: str
    url: str
    abstract: str
    authors: List[str]
    published_date: str
    pdf_url: str
    categories: List[str]
    score: float = 0.0
    metadata: Dict[str, Any] = None
    full_text: Optional[str] = None  # Add field for full PDF content


@dataclass
class ArxivQuery:
    """Represents an ArXiv search query with parameters."""

    query: str
    search_query: str = "all"  # all, title, author, abstract, comment, etc.
    max_results: int = 10
    sort_by: str = "relevance"  # relevance, lastUpdatedDate, submittedDate
    sort_order: str = "descending"  # ascending, descending


class ArxivSearchTool:
    """
    ArXiv search tool for academic paper research.

    This tool provides:
    - Academic paper search using ArXiv API
    - Recent paper discovery
    - Field-specific searches
    - Paper metadata extraction
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        """Initialize the ArXiv search tool."""
        try:
            self.model = (
                ModelBuilder().with_provider("openai").with_model("gpt-4o-mini").build()
            )
            logger.info("ArXiv insight extraction model loaded.")
        except Exception as e:
            logger.error(f"Failed to load insight extraction model: {e}")
            self.model = None

    def search(
        self, query: str, download_pdfs: bool = True, **kwargs
    ) -> List[ArxivResult]:
        """
        Perform an ArXiv search.

        Args:
            query: The search query
            download_pdfs: Whether to download and extract PDF content
            **kwargs: Additional search parameters

        Returns:
            List of ArXiv results
        """
        search_query = ArxivQuery(query=query, **kwargs)
        results = self._execute_search(search_query)
        return self._enhance_results_with_pdf_content(results, download_pdfs)

    def search_recent_papers(
        self, topic: str, days_back: int = 30, download_pdfs: bool = True
    ) -> List[ArxivResult]:
        """
        Search for recent papers on a topic.

        Args:
            topic: The research topic
            days_back: Number of days to look back
            download_pdfs: Whether to download and extract PDF content

        Returns:
            List of recent ArXiv results
        """
        search_params = {
            "max_results": 15,
            "sort_by": "submittedDate",
            "sort_order": "descending",
        }

        return self.search(topic, download_pdfs=download_pdfs, **search_params)

    def search_by_category(
        self, query: str, categories: List[str], download_pdfs: bool = True
    ) -> List[ArxivResult]:
        """
        Search within specific ArXiv categories.

        Args:
            query: The search query
            categories: List of ArXiv category codes (e.g., ['cs.AI', 'cs.LG'])
            download_pdfs: Whether to download and extract PDF content

        Returns:
            List of ArXiv results
        """
        # Construct category filter
        cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
        combined_query = f"({query}) AND ({cat_filter})"

        return self.search(combined_query, download_pdfs=download_pdfs)

    def _download_pdf_content(self, pdf_url: str) -> Optional[str]:
        """
        Download and extract text content from a PDF.

        Args:
            pdf_url: URL to the PDF file

        Returns:
            Extracted text content or None if extraction fails
        """
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not available. Cannot extract PDF content.")
            return None

        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            # Extract text using PyPDF2
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page: {e}")
                    continue

            if len(text.strip()) == 0:
                logger.warning("No text extracted from PDF")
                return None

            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text.strip()

        except Exception as e:
            logger.error(f"Failed to download or extract PDF content: {e}")
            return None

    def _enhance_results_with_pdf_content(
        self, results: List[ArxivResult], download_pdfs: bool = True
    ) -> List[ArxivResult]:
        """
        Enhance ArXiv results with full PDF content.

        Args:
            results: List of ArXiv results
            download_pdfs: Whether to download and extract PDF content

        Returns:
            Enhanced results with PDF content
        """
        if not download_pdfs or not PDF_AVAILABLE:
            return results

        enhanced_results = []
        for result in results:
            if result.pdf_url:
                logger.info(f"Downloading PDF for: {result.title}")
                result.full_text = self._download_pdf_content(result.pdf_url)
            enhanced_results.append(result)

        return enhanced_results

    def _execute_search(self, search_query: ArxivQuery) -> List[ArxivResult]:
        """
        Execute the search using ArXiv API.

        Args:
            search_query: The search query configuration

        Returns:
            List of ArXiv results
        """
        try:
            # Prepare search parameters
            params = {
                "search_query": f"{search_query.search_query}:{search_query.query}",
                "max_results": search_query.max_results,
                "sortBy": search_query.sort_by,
                "sortOrder": search_query.sort_order,
            }

            # Execute search
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()

            # Parse RSS feed
            feed = feedparser.parse(response.content)

            # Process results
            results = []
            for entry in feed.entries:
                # Extract authors
                authors = []
                if hasattr(entry, "authors"):
                    authors = [author.name for author in entry.authors]
                elif hasattr(entry, "author"):
                    authors = [entry.author]

                # Extract categories
                categories = []
                if hasattr(entry, "tags"):
                    categories = [tag.term for tag in entry.tags]

                # Extract PDF URL
                pdf_url = ""
                if hasattr(entry, "links"):
                    for link in entry.links:
                        if link.type == "application/pdf":
                            pdf_url = link.href
                            break

                arxiv_result = ArxivResult(
                    title=entry.title.replace("\n", " ").strip(),
                    url=entry.link,
                    abstract=entry.summary.replace("\n", " ").strip(),
                    authors=authors,
                    published_date=entry.published,
                    pdf_url=pdf_url,
                    categories=categories,
                    metadata={
                        "arxiv_id": entry.id.split("/")[-1],
                        "updated": getattr(entry, "updated", ""),
                        "doi": getattr(entry, "arxiv_doi", ""),
                    },
                )
                results.append(arxiv_result)

            logger.info(f"ArXiv search completed: {len(results)} results found")
            return results

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    def extract_key_insights(self, arxiv_results: List[ArxivResult]) -> Dict[str, Any]:
        """
        Extract key insights from ArXiv results using an LLM.

        Args:
            arxiv_results: List of ArXiv results

        Returns:
            Dictionary containing extracted insights
        """
        if not self.model:
            logger.warning(
                "Insight extraction model not available. Returning raw results."
            )
            return {"error": "Model not available"}

        insights = {}

        for i, result in enumerate(arxiv_results):
            # Use full text if available, otherwise use abstract
            if result.full_text:
                content_for_llm = f"Title: {result.title}\n\nFull Paper Content:\n{result.full_text}\n\nAuthors: {', '.join(result.authors)}"
                content_type = "full paper"
            else:
                content_for_llm = f"Title: {result.title}\n\nAbstract: {result.abstract}\n\nAuthors: {', '.join(result.authors)}"
                content_type = "abstract only"

            insights[f"{i+1}"] = {
                "title": result.title,
                "url": result.url,
                "authors": result.authors,
                "published": result.published_date,
                "categories": result.categories,
                "content_type": content_type,
                "pdf_available": result.pdf_url is not None,
                "full_text_extracted": result.full_text is not None,
            }

            prompt = f"""
            Based on the following academic paper ({content_type}), extract relevant insights and key findings.
            Focus on:
            - "key_findings": main research findings and conclusions
            - "methodology": research methods and approaches used
            - "relevance": how this relates to current research trends
            - "implications": potential applications or impact
            - "data_and_experiments": experimental setup and data analysis (if available)
            - "limitations": acknowledged limitations or future work needed
            
            Paper details:
            {content_for_llm}
            """

            try:
                response = self.model.run(prompt)
                insights[f"{i+1}"]["insights"] = response
            except Exception as e:
                logger.error(f"Failed to get insights from LLM: {e}")
                insights[f"{i+1}"]["insights"] = f"Analysis failed due to error: {e}"

        return insights

    def get_search_summary(self, query: str, insights: Dict[str, Any]) -> str:
        """
        Generate a summary of ArXiv search insights using an LLM.

        Args:
            query: The original search query
            insights: The extracted insights from the search results

        Returns:
            A summary string
        """
        if not self.model:
            return "Summary model not available."

        summary_prompt = f"""
        Original Query: {query}
        
        Academic Paper Insights:
        {json.dumps(insights, indent=2)}
        
        Based on the academic papers found, provide a comprehensive summary focusing on:
        1. Current state of research in this field
        2. Key findings and methodologies
        3. Research gaps and future directions
        4. Most relevant papers for further reading
        """

        try:
            summary = self.model.run(summary_prompt)
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Could not generate a summary due to an error."

    def format_results(
        self, query: str, arxiv_results: List[ArxivResult]
    ) -> Dict[str, Any]:
        """Return standardized structure with key insights and summary."""
        if not arxiv_results:
            return {
                "query": query,
                "results": [],
                "summary": "No results",
                "processing_details": {
                    "papers_found": 0,
                    "papers_processed": 0,
                    "pdfs_downloaded": 0,
                    "insights_extracted": 0,
                },
            }

        # Track processing details
        papers_with_pdfs = len([r for r in arxiv_results if r.pdf_url])
        papers_with_full_text = len([r for r in arxiv_results if r.full_text])

        insights_map = self.extract_key_insights(arxiv_results)
        insights_extracted = len(insights_map) if isinstance(insights_map, dict) else 0

        results_list: List[Dict[str, Any]] = []
        for idx, res in enumerate(arxiv_results):
            key_insights = None
            if isinstance(insights_map, dict):
                key_insights = insights_map.get(str(idx + 1), {}).get("insights")

            results_list.append(
                {
                    "title": res.title,
                    "url": res.url,
                    "relevance_score": res.score,
                    "type": "academic",
                    "key_insights": key_insights,
                    "authors": res.authors[:3],  # Show first 3 authors
                    "published_date": res.published_date,
                    "categories": res.categories[:2],  # Show first 2 categories
                    "pdf_available": res.pdf_url is not None,
                    "full_text_extracted": res.full_text is not None,
                }
            )

        summary = ""
        if isinstance(insights_map, dict):
            try:
                summary = self.get_search_summary(query, insights_map)
            except Exception:
                summary = f"Found {len(arxiv_results)} academic papers on ArXiv related to the query."

        return {
            "query": query,
            "results": results_list,
            "summary": summary,
            "processing_details": {
                "papers_found": len(arxiv_results),
                "papers_processed": len(arxiv_results),
                "pdfs_available": papers_with_pdfs,
                "pdfs_downloaded": papers_with_full_text,
                "insights_extracted": insights_extracted,
                "categories_covered": len(
                    set([cat for res in arxiv_results for cat in res.categories])
                ),
                "date_range": {
                    "earliest": (
                        min([res.published_date for res in arxiv_results])
                        if arxiv_results
                        else ""
                    ),
                    "latest": (
                        max([res.published_date for res in arxiv_results])
                        if arxiv_results
                        else ""
                    ),
                },
            },
        }


# Function to test ArXiv API
def test_arxiv_search(query: str) -> List[ArxivResult]:
    """
    Perform a quick ArXiv search.

    Args:
        query: The search query

    Returns:
        List of ArXiv results
    """
    search_tool = ArxivSearchTool()
    return search_tool.search(query)
