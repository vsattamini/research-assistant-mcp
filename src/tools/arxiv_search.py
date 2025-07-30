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
from urllib.parse import quote_plus

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
            self.model = (ModelBuilder()
                          .with_provider("openai")
                          .with_model("gpt-4o-mini")
                          .build())
            logger.info("ArXiv insight extraction model loaded.")
        except Exception as e:
            logger.error(f"Failed to load insight extraction model: {e}")
            self.model = None
    
    def search(self, query: str, **kwargs) -> List[ArxivResult]:
        """
        Perform an ArXiv search.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            List of ArXiv results
        """
        search_query = ArxivQuery(query=query, **kwargs)
        return self._execute_search(search_query)
    
    def search_recent_papers(self, topic: str, days_back: int = 30) -> List[ArxivResult]:
        """
        Search for recent papers on a topic.
        
        Args:
            topic: The research topic
            days_back: Number of days to look back
            
        Returns:
            List of recent ArXiv results
        """
        search_params = {
            "max_results": 15,
            "sort_by": "submittedDate",
            "sort_order": "descending"
        }
        
        return self.search(topic, **search_params)
    
    def search_by_category(self, query: str, categories: List[str]) -> List[ArxivResult]:
        """
        Search within specific ArXiv categories.
        
        Args:
            query: The search query
            categories: List of ArXiv category codes (e.g., ['cs.AI', 'cs.LG'])
            
        Returns:
            List of ArXiv results
        """
        # Construct category filter
        cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
        combined_query = f"({query}) AND ({cat_filter})"
        
        return self.search(combined_query)
    
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
                "search_query": f"{search_query.search_query}:{quote_plus(search_query.query)}",
                "max_results": search_query.max_results,
                "sortBy": search_query.sort_by,
                "sortOrder": search_query.sort_order
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
                if hasattr(entry, 'authors'):
                    authors = [author.name for author in entry.authors]
                elif hasattr(entry, 'author'):
                    authors = [entry.author]
                
                # Extract categories
                categories = []
                if hasattr(entry, 'tags'):
                    categories = [tag.term for tag in entry.tags]
                
                # Extract PDF URL
                pdf_url = ""
                if hasattr(entry, 'links'):
                    for link in entry.links:
                        if link.type == 'application/pdf':
                            pdf_url = link.href
                            break
                
                arxiv_result = ArxivResult(
                    title=entry.title.replace('\n', ' ').strip(),
                    url=entry.link,
                    abstract=entry.summary.replace('\n', ' ').strip(),
                    authors=authors,
                    published_date=entry.published,
                    pdf_url=pdf_url,
                    categories=categories,
                    metadata={
                        "arxiv_id": entry.id.split('/')[-1],
                        "updated": getattr(entry, 'updated', ''),
                        "doi": getattr(entry, 'arxiv_doi', '')
                    }
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
            logger.warning("Insight extraction model not available. Returning raw results.")
            return {"error": "Model not available"}
        
        insights = {}
        
        for i, result in enumerate(arxiv_results):
            insights[f"{i+1}"] = {
                "title": result.title,
                "url": result.url,
                "authors": result.authors,
                "published": result.published_date,
                "categories": result.categories
            }
            
            content_for_llm = f"Title: {result.title}\n\nAbstract: {result.abstract}\n\nAuthors: {', '.join(result.authors)}"
            
            prompt = f"""
            Based on the following academic paper, extract relevant insights and key findings.
            Focus on:
            - "key_findings": main research findings and conclusions
            - "methodology": research methods and approaches used
            - "relevance": how this relates to current research trends
            - "implications": potential applications or impact
            
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
    
    def get_search_summary(
        self, query: str, insights: Dict[str, Any]
    ) -> str:
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