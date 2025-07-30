"""
Web Search Tool for Research Assistant

This module provides web search capabilities using the Tavily API
to find relevant information for research questions.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    content: str
    score: float
    source_type: str
    published_date: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class SearchQuery:
    """Represents a search query with parameters."""
    query: str
    search_depth: str = "basic"  # basic, advanced
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    include_answer: bool = True
    include_raw_content: bool = False
    max_results: int = 10
    search_type: str = "search"  # search, news, places


class WebSearchTool:
    """
    Web search tool using Tavily API for research purposes.
    
    This tool provides:
    - Web search with configurable depth
    - News search
    - Domain-specific search
    - Content extraction and analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.
        
        Args:
            api_key: Tavily API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("Tavily API key not found. Web search will be simulated.")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                self.client = None
    
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Perform a web search.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        search_query = SearchQuery(query=query, **kwargs)
        return self._execute_search(search_query)
    
    def search_research_topic(self, topic: str, include_academic: bool = True) -> List[SearchResult]:
        """
        Search for research-related information on a topic.
        
        Args:
            topic: The research topic
            include_academic: Whether to include academic sources
            
        Returns:
            List of search results
        """
        # Build a research-focused query
        research_query = f"research {topic} latest findings studies"
        
        # Configure search parameters for research
        search_params = {
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 15
        }
        
        if include_academic:
            # Add academic domains
            academic_domains = [
                "scholar.google.com",
                "researchgate.net",
                "arxiv.org",
                "pubmed.ncbi.nlm.nih.gov",
                "ieee.org",
                "acm.org",
                "springer.com",
                "sciencedirect.com"
            ]
            search_params["include_domains"] = academic_domains
        
        return self.search(research_query, **search_params)
    
    def search_news(self, topic: str, days_back: int = 30) -> List[SearchResult]:
        """
        Search for recent news on a topic.
        
        Args:
            topic: The topic to search for
            days_back: Number of days to look back
            
        Returns:
            List of news search results
        """
        news_query = f"news {topic} recent developments"
        
        search_params = {
            "search_type": "news",
            "search_depth": "basic",
            "max_results": 10
        }
        
        return self.search(news_query, **search_params)
    
    def search_statistics(self, topic: str) -> List[SearchResult]:
        """
        Search for statistics and data on a topic.
        
        Args:
            topic: The topic to search for statistics
            
        Returns:
            List of search results with statistical information
        """
        stats_query = f"statistics data numbers {topic} 2024 2023"
        
        # Focus on government and data sources
        data_domains = [
            "worldbank.org",
            "data.worldbank.org",
            "un.org",
            "who.int",
            "cdc.gov",
            "bls.gov",
            "census.gov",
            "data.gov"
        ]
        
        search_params = {
            "include_domains": data_domains,
            "search_depth": "advanced",
            "max_results": 10
        }
        
        return self.search(stats_query, **search_params)
    
    def _execute_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """
        Execute the search using Tavily API or returns empty if API key is not set
        
        Args:
            search_query: The search query configuration
            
        Returns:
            List of search results
        """
        if self.client is None:
            return "Web search returned no results, check API key or Tavily APIs status"
        
        try:
            # Prepare search parameters
            search_params = {
                "query": search_query.query,
                "search_depth": search_query.search_depth,
                "include_answer": search_query.include_answer,
                "include_raw_content": search_query.include_raw_content,
                "max_results": search_query.max_results,
                "search_type": search_query.search_type
            }
            
            # Add optional parameters
            if search_query.include_domains:
                search_params["include_domains"] = search_query.include_domains
            if search_query.exclude_domains:
                search_params["exclude_domains"] = search_query.exclude_domains
            
            # Execute search
            response = self.client.search(**search_params)
            
            # Process results
            results = []
            for result in response.get("results", []):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    source_type=result.get("source_type", "unknown"),
                    published_date=result.get("published_date"),
                    metadata=result.get("metadata", {})
                )
                results.append(search_result)
            
            logger.info(f"Search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return "Web search returned no results, check API key or Tavily APIs status"
    
    
    def extract_key_insights(self, search_results: List[SearchResult]) -> Dict[str, Any]:
        """
        Extract key insights from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            Dictionary containing extracted insights
        """
        insights = {
            "key_facts": [],
            "statistics": [],
            "trends": [],
            "sources": [],
            "confidence_score": 0.0
        }
        
        for result in search_results:
            # Extract key facts (simple heuristic)
            content_words = result.content.split()
            if len(content_words) > 20:
                insights["key_facts"].append(result.content[:200] + "...")
            
            # Extract statistics (look for numbers)
            if any(char.isdigit() for char in result.content):
                insights["statistics"].append(result.content[:150] + "...")
            
            # Track sources
            insights["sources"].append({
                "title": result.title,
                "url": result.url,
                "score": result.score,
                "type": result.source_type
            })
        
        # Calculate confidence score based on result quality
        if search_results:
            avg_score = sum(r.score for r in search_results) / len(search_results)
            insights["confidence_score"] = avg_score
        
        return insights
    
    def get_search_summary(self, search_results: List[SearchResult]) -> str:
        """
        Generate a summary of search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            Summary string
        """
        if not search_results:
            return "No search results found."
        
        summary_parts = [
            f"Found {len(search_results)} relevant sources:",
            ""
        ]
        
        for i, result in enumerate(search_results[:5], 1):
            summary_parts.append(f"{i}. **{result.title}**")
            summary_parts.append(f"   - Source: {result.source_type}")
            summary_parts.append(f"   - Relevance: {result.score:.2f}")
            summary_parts.append(f"   - {result.content[:100]}...")
            summary_parts.append("")
        
        if len(search_results) > 5:
            summary_parts.append(f"... and {len(search_results) - 5} more sources")
        
        return "\n".join(summary_parts)


# Convenience function for quick search
def test_search(query: str, api_key: Optional[str] = None) -> List[SearchResult]:
    """
    Perform a quick web search.
    
    Args:
        query: The search query
        api_key: Optional Tavily API key
        
    Returns:
        List of search results
    """
    search_tool = WebSearchTool(api_key=api_key)
    return search_tool.search(query)


# Example usage
if __name__ == "__main__":
    # Example search
    search_tool = WebSearchTool()
    
    # Research search
    results = search_tool.search_research_topic("renewable energy costs")
    print(f"Found {len(results)} research results")
    
    # Extract insights
    insights = search_tool.extract_key_insights(results)
    print(f"Key insights: {len(insights['key_facts'])} facts found")
    
    # Generate summary
    summary = search_tool.get_search_summary(results)
    print(summary)
