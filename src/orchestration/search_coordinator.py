"""
Search Coordinator for Research Assistant

This module coordinates searches across multiple sources (web and academic)
and returns structured search plans and results.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from tools.web_search import WebSearchTool, SearchResult
from tools.arxiv_search import ArxivSearchTool, ArxivResult

logger = logging.getLogger(__name__)


class SearchSource(str, Enum):
    """Available search sources."""
    WEB = "web"
    ARXIV = "arxiv"
    BOTH = "both"


class SearchType(str, Enum):
    """Types of searches to perform."""
    GENERAL = "general"
    NEWS = "news"
    ACADEMIC = "academic"
    STATISTICS = "statistics"
    RECENT_PAPERS = "recent_papers"


@dataclass
class SearchPlan:
    """Represents a planned search operation."""
    search_term: str
    search_type: SearchType
    search_source: SearchSource
    priority: int  # 1-10, higher is more important
    max_results: int = 10
    metadata: Dict[str, Any] = None


@dataclass
class SearchResults:
    """Combined search results from multiple sources."""
    web_results: List[SearchResult]
    arxiv_results: List[ArxivResult]
    search_plan: SearchPlan
    summary: str = ""
    insights: Dict[str, Any] = None


class SearchCoordinator:
    """
    Coordinates searches across multiple sources using structured planning.
    
    Uses OpenAI function calling to generate structured search plans,
    then executes searches using WebSearchTool and ArxivSearchTool.
    """
    
    def __init__(self, 
                 web_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """Initialize search coordinator with tools."""
        
        # Initialize search tools
        self.web_tool = WebSearchTool(api_key=web_api_key)
        self.arxiv_tool = ArxivSearchTool()
        
        # Initialize OpenAI client for structured planning
        if OPENAI_AVAILABLE and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI not available for structured search planning")
        
        # Function tool for search planning
        self._planning_tool = {
            "type": "function",
            "function": {
                "name": "create_search_plan",
                "description": "Create a structured search plan for research",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "searches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "search_term": {
                                        "type": "string",
                                        "description": "Specific search query"
                                    },
                                    "search_type": {
                                        "type": "string",
                                        "enum": [t.value for t in SearchType],
                                        "description": "Type of search to perform"
                                    },
                                    "search_source": {
                                        "type": "string", 
                                        "enum": [s.value for s in SearchSource],
                                        "description": "Which sources to search"
                                    },
                                    "priority": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                        "description": "Search priority (1-10)"
                                    },
                                    "max_results": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 20,
                                        "description": "Maximum results to return"
                                    }
                                },
                                "required": ["search_term", "search_type", "search_source", "priority"]
                            }
                        }
                    },
                    "required": ["searches"]
                }
            }
        }
    
    def plan_searches(self, research_question: str, focus: str) -> List[SearchPlan]:
        """
        Create a structured search plan for the research question.
        
        Args:
            research_question: The main research question
            focus: Specific focus or aspect to search for
            
        Returns:
            List of SearchPlan objects
        """
        if not self.openai_client:
            # Fallback to simple planning
            return self._create_fallback_plan(research_question, focus)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a research search planner. Create comprehensive "
                        "search strategies that combine web and academic sources."
                    )
                },
                {
                    "role": "user", 
                    "content": f"""
                    Research Question: {research_question}
                    Search Focus: {focus}
                    
                    Create a search plan that covers:
                    1. General web information
                    2. Recent news and developments  
                    3. Academic papers and research
                    4. Statistics and data
                    
                    Prioritize searches that are most likely to provide relevant, 
                    high-quality information for this research question.
                    """
                }
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[self._planning_tool],
                tool_choice="auto",
                temperature=0.1
            )
            
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                arguments = message.tool_calls[0].function.arguments
                payload = json.loads(arguments)
                searches_data = payload.get("searches", [])
                
                plans = []
                for search_data in searches_data:
                    plan = SearchPlan(
                        search_term=search_data["search_term"],
                        search_type=SearchType(search_data["search_type"]),
                        search_source=SearchSource(search_data["search_source"]),
                        priority=search_data["priority"],
                        max_results=search_data.get("max_results", 10)
                    )
                    plans.append(plan)
                
                logger.info(f"Created {len(plans)} search plans")
                return sorted(plans, key=lambda x: x.priority, reverse=True)
            
        except Exception as e:
            logger.warning(f"Structured search planning failed: {e}. Using fallback.")
        
        return self._create_fallback_plan(research_question, focus)
    
    def _create_fallback_plan(self, research_question: str, focus: str) -> List[SearchPlan]:
        """Create a basic search plan when structured planning fails."""
        plans = [
            SearchPlan(
                search_term=f"{focus} {research_question}",
                search_type=SearchType.GENERAL,
                search_source=SearchSource.WEB,
                priority=8,
                max_results=10
            ),
            SearchPlan(
                search_term=research_question,
                search_type=SearchType.ACADEMIC,
                search_source=SearchSource.ARXIV,
                priority=7,
                max_results=8
            ),
            SearchPlan(
                search_term=f"{focus} recent developments",
                search_type=SearchType.NEWS,
                search_source=SearchSource.WEB,
                priority=6,
                max_results=5
            )
        ]
        return plans
    
    def execute_search_plan(self, plan: SearchPlan) -> SearchResults:
        """
        Execute a single search plan.
        
        Args:
            plan: The search plan to execute
            
        Returns:
            SearchResults containing results from relevant sources
        """
        web_results = []
        arxiv_results = []
        
        try:
            # Execute web search if requested
            if plan.search_source in [SearchSource.WEB, SearchSource.BOTH]:
                if plan.search_type == SearchType.GENERAL:
                    web_results = self.web_tool.search(plan.search_term, max_results=plan.max_results)
                elif plan.search_type == SearchType.NEWS:
                    web_results = self.web_tool.search_news(plan.search_term)
                elif plan.search_type == SearchType.STATISTICS:
                    web_results = self.web_tool.search_statistics(plan.search_term)
                else:
                    web_results = self.web_tool.search_research_topic(plan.search_term)
            
            # Execute arxiv search if requested  
            if plan.search_source in [SearchSource.ARXIV, SearchSource.BOTH]:
                if plan.search_type == SearchType.RECENT_PAPERS:
                    arxiv_results = self.arxiv_tool.search_recent_papers(plan.search_term)
                else:
                    arxiv_results = self.arxiv_tool.search(plan.search_term, max_results=plan.max_results)
            
            logger.info(f"Search executed: {len(web_results)} web + {len(arxiv_results)} arxiv results")
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
        
        return SearchResults(
            web_results=web_results,
            arxiv_results=arxiv_results,
            search_plan=plan
        )
    
    def execute_all_searches(self, plans: List[SearchPlan]) -> List[SearchResults]:
        """
        Execute multiple search plans.
        
        Args:
            plans: List of search plans to execute
            
        Returns:
            List of SearchResults
        """
        all_results = []
        for plan in plans:
            results = self.execute_search_plan(plan)
            all_results.append(results)
        
        return all_results 

    # ------------------------------------------------------------------
    # Thin wrapper: one-shot combined search (no LLM planning)
    # ------------------------------------------------------------------

    def simple_search(self, query: str, focus: str = "") -> SearchResults:
        """Run a lightweight combined search across web + arxiv.

        This skips the LLM planning step and just performs:
        1. A general web search for research topics (Tavily)
        2. A recent-papers search on ArXiv

        Args:
            query: The overall research question
            focus: Optional focus string to refine the web search

        Returns:
            SearchResults object with unified lists.
        """
        search_term = f"{focus} {query}".strip()

        try:
            web_results = self.web_tool.search_research_topic(search_term)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            web_results = []

        try:
            arxiv_results = self.arxiv_tool.search_recent_papers(query)
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            arxiv_results = []

        plan = SearchPlan(
            search_term=search_term,
            search_type=SearchType.GENERAL,
            search_source=SearchSource.BOTH,
            priority=10,
            max_results=10,
        )

        return SearchResults(
            web_results=web_results,
            arxiv_results=arxiv_results,
            search_plan=plan,
        ) 