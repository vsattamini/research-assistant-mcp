"""
Search Coordinator for Research Assistant

This module coordinates searches across multiple sources (web and academic)
and returns structured search plans and results.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.tools.web_search import WebSearchTool, SearchResult
from src.tools.arxiv_search import ArxivSearchTool, ArxivResult
from src.tools.intelligent_search_planner import (
    IntelligentSearchPlanner,
    SearchPlan as IntelligentSearchPlan,
)

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
    standardized_results: List[Dict[str, Any]] = None
    # Indicates whether the individual tool results have already been processed
    # (i.e. extract_key_insights has been run) so that downstream steps can
    # decide whether a separate EXTRACT task is necessary.
    results_processed: bool = False
    metadata: Dict[str, Any] = None


class SearchCoordinator:
    """
    Coordinates searches across multiple sources using structured planning.

    Uses OpenAI function calling to generate structured search plans,
    then executes searches using WebSearchTool and ArxivSearchTool.
    """

    def __init__(
        self, web_api_key: Optional[str] = None, openai_api_key: Optional[str] = None
    ):
        """Initialize search coordinator with tools."""

        # Initialize search tools
        self.web_tool = WebSearchTool(api_key=web_api_key)
        self.arxiv_tool = ArxivSearchTool()

        # Initialize intelligent search planner
        self.intelligent_planner = IntelligentSearchPlanner(
            openai_api_key=openai_api_key
        )

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
                                        "description": "Specific search query",
                                    },
                                    "search_type": {
                                        "type": "string",
                                        "enum": [t.value for t in SearchType],
                                        "description": "Type of search to perform",
                                    },
                                    "search_source": {
                                        "type": "string",
                                        "enum": [s.value for s in SearchSource],
                                        "description": "Which sources to search",
                                    },
                                    "priority": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                        "description": "Search priority (1-10)",
                                    },
                                    "max_results": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 20,
                                        "description": "Maximum results to return",
                                    },
                                },
                                "required": [
                                    "search_term",
                                    "search_type",
                                    "search_source",
                                    "priority",
                                ],
                            },
                        }
                    },
                    "required": ["searches"],
                },
            },
        }

    def plan_searches(self, research_question: str, focus: str) -> List[SearchPlan]:
        """
        Create a structured search plan for the research question using intelligent analysis.

        Args:
            research_question: The main research question
            focus: Specific focus or aspect to search for

        Returns:
            List of SearchPlan objects
        """
        # Use intelligent planner to analyze the question
        intelligent_plan = self.intelligent_planner.analyze_research_question(
            research_question
        )

        # Convert intelligent plan to traditional search plans
        plans = self._convert_intelligent_plan_to_search_plans(
            intelligent_plan, research_question, focus
        )

        logger.info(
            f"Created {len(plans)} intelligent search plans for domain: {intelligent_plan.primary_domain}"
        )
        return sorted(plans, key=lambda x: x.priority, reverse=True)

    def _convert_intelligent_plan_to_search_plans(
        self,
        intelligent_plan: IntelligentSearchPlan,
        research_question: str,
        focus: str,
    ) -> List[SearchPlan]:
        """Convert intelligent search plan to traditional search plans."""
        plans = []

        # Always include web search as it's broadly applicable
        if any(
            "web" in strategy.value for strategy in intelligent_plan.search_strategies
        ):
            web_term = f"{focus} {research_question}" if focus else research_question
            plans.append(
                SearchPlan(
                    search_term=web_term,
                    search_type=SearchType.GENERAL,
                    search_source=SearchSource.WEB,
                    priority=9,
                    max_results=10,
                    metadata={
                        "intelligent_analysis": True,
                        "domain": intelligent_plan.primary_domain.value,
                    },
                )
            )

        # Add ArXiv search only if deemed suitable by intelligent analysis
        if intelligent_plan.arxiv_suitable and intelligent_plan.academic_suitable:
            plans.append(
                SearchPlan(
                    search_term=research_question,
                    search_type=SearchType.ACADEMIC,
                    search_source=SearchSource.ARXIV,
                    priority=8,
                    max_results=8,
                    metadata={
                        "intelligent_analysis": True,
                        "confidence": intelligent_plan.confidence_score,
                    },
                )
            )

        # Add news search for current events or time-sensitive topics
        if any(
            "news" in strategy.value for strategy in intelligent_plan.search_strategies
        ):
            plans.append(
                SearchPlan(
                    search_term=(
                        f"{focus} recent developments"
                        if focus
                        else f"{research_question} recent"
                    ),
                    search_type=SearchType.NEWS,
                    search_source=SearchSource.WEB,
                    priority=7,
                    max_results=5,
                    metadata={"intelligent_analysis": True},
                )
            )

        # Add statistical data search if recommended
        if any(
            "statistical" in strategy.value
            for strategy in intelligent_plan.search_strategies
        ):
            plans.append(
                SearchPlan(
                    search_term=f"{research_question} statistics data",
                    search_type=SearchType.STATISTICS,
                    search_source=SearchSource.WEB,
                    priority=6,
                    max_results=5,
                    metadata={"intelligent_analysis": True},
                )
            )

        return plans if plans else self._create_fallback_plan(research_question, focus)

    def _create_fallback_plan(
        self, research_question: str, focus: str
    ) -> List[SearchPlan]:
        """Create a basic search plan when intelligent planning fails."""
        plans = [
            SearchPlan(
                search_term=(
                    f"{focus} {research_question}" if focus else research_question
                ),
                search_type=SearchType.GENERAL,
                search_source=SearchSource.WEB,
                priority=8,
                max_results=10,
                metadata={"fallback_plan": True},
            ),
            SearchPlan(
                search_term=(
                    f"{focus} recent developments"
                    if focus
                    else f"{research_question} recent"
                ),
                search_type=SearchType.NEWS,
                search_source=SearchSource.WEB,
                priority=6,
                max_results=5,
                metadata={"fallback_plan": True},
            ),
        ]
        return plans

    def execute_search_plan(
        self, plan: SearchPlan, process_results: bool = True
    ) -> SearchResults:
        """
        Execute a single search plan.

        Args:
            plan: The search plan to execute

        Returns:
            SearchResults containing results from relevant sources
        """
        web_results: List[SearchResult] = []
        arxiv_results: List[ArxivResult] = []

        try:
            # Execute web search if requested
            if plan.search_source in [SearchSource.WEB, SearchSource.BOTH]:
                if plan.search_type == SearchType.GENERAL:
                    web_results = self.web_tool.search(
                        plan.search_term, max_results=plan.max_results
                    )
                elif plan.search_type == SearchType.NEWS:
                    web_results = self.web_tool.search_news(plan.search_term)
                elif plan.search_type == SearchType.STATISTICS:
                    web_results = self.web_tool.search_statistics(plan.search_term)
                else:
                    web_results = self.web_tool.search_research_topic(plan.search_term)

            # Execute arxiv search if requested and validated by intelligent planner
            if plan.search_source in [SearchSource.ARXIV, SearchSource.BOTH]:
                # Check if this is an intelligently planned search or fallback
                if plan.metadata and plan.metadata.get("intelligent_analysis"):
                    logger.info(
                        "Executing ArXiv search validated by intelligent planner"
                    )
                    if plan.search_type == SearchType.RECENT_PAPERS:
                        arxiv_results = self.arxiv_tool.search_recent_papers(
                            plan.search_term
                        )
                    else:
                        arxiv_results = self.arxiv_tool.search(
                            plan.search_term, max_results=plan.max_results
                        )
                else:
                    logger.info(
                        "Skipping ArXiv search - not validated by intelligent analysis"
                    )
                    arxiv_results = []

            logger.info(
                f"Search executed: {len(web_results)} web + {len(arxiv_results)} arxiv results"
            )

        except Exception as e:
            logger.error(f"Search execution failed: {e}")

        # Optional post-processing – run the insight-extraction helpers
        standardized: List[Dict[str, Any]] = []
        if process_results:
            if web_results:
                standardized.append(
                    self.web_tool.format_results(plan.search_term, web_results)
                )
            if arxiv_results:
                standardized.append(
                    self.arxiv_tool.format_results(plan.search_term, arxiv_results)
                )

        # Generate search metrics
        search_metrics = {
            "execution_time": datetime.now().isoformat(),
            "web_results_count": len(web_results),
            "arxiv_results_count": len(arxiv_results),
            "total_sources": len(web_results) + len(arxiv_results),
            "search_plan_executed": plan.__dict__,
            "processing_completed": process_results and bool(standardized),
            "unique_domains": len(
                set(
                    [
                        getattr(r, "url", "").split("/")[2]
                        for r in web_results
                        if hasattr(r, "url") and r.url
                    ]
                )
            ),
            "search_success": len(web_results) > 0 or len(arxiv_results) > 0,
        }

        return SearchResults(
            web_results=web_results,
            arxiv_results=arxiv_results,
            search_plan=plan,
            standardized_results=standardized if standardized else None,
            results_processed=process_results and bool(standardized),
            summary=f"Search completed: {len(web_results)} web + {len(arxiv_results)} arXiv results",
            metadata=search_metrics,
        )

    def execute_all_searches(
        self, plans: List[SearchPlan], process_results: bool = True
    ) -> List[SearchResults]:
        """
        Execute multiple search plans.

        Args:
            plans: List of search plans to execute

        Returns:
            List of SearchResults
        """
        all_results = []
        for plan in plans:
            results = self.execute_search_plan(plan, process_results=process_results)
            all_results.append(results)

        return all_results

    def simple_search(
        self, query: str, focus: str = "", process_results: bool = True
    ) -> SearchResults:
        """Run an intelligent combined search across appropriate sources.

        Uses the intelligent search planner to determine the best approach:
        1. Analyzes the research question to determine optimal strategy
        2. Performs web search (always appropriate)
        3. Only uses ArXiv if the question is suitable for academic papers

        Args:
            query: The overall research question
            focus: Optional focus string to refine the web search

        Returns:
            SearchResults object with unified lists.
        """
        search_term = f"{focus} {query}".strip() if focus else query

        # Use intelligent planner to analyze the question
        intelligent_plan = self.intelligent_planner.analyze_research_question(query)
        logger.info(
            f"Intelligent analysis: Domain={intelligent_plan.primary_domain.value}, "
            f"ArXiv suitable={intelligent_plan.arxiv_suitable}, "
            f"Confidence={intelligent_plan.confidence_score}"
        )

        # Always perform web search as it's broadly applicable
        try:
            web_results = self.web_tool.search_research_topic(search_term)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            web_results = []

        # Only perform ArXiv search if validated by intelligent analysis
        arxiv_results = []
        if intelligent_plan.arxiv_suitable and intelligent_plan.academic_suitable:
            try:
                logger.info(
                    "Performing ArXiv search - validated by intelligent analysis"
                )
                arxiv_results = self.arxiv_tool.search_recent_papers(query)
            except Exception as e:
                logger.error(f"ArXiv search failed: {e}")
                arxiv_results = []
        else:
            logger.info(
                f"Skipping ArXiv search - not suitable for domain: {intelligent_plan.primary_domain.value}"
            )

        plan = SearchPlan(
            search_term=search_term,
            search_type=SearchType.GENERAL,
            search_source=(
                SearchSource.WEB
                if not intelligent_plan.arxiv_suitable
                else SearchSource.BOTH
            ),
            priority=10,
            max_results=10,
            metadata={
                "intelligent_analysis": True,
                "domain": intelligent_plan.primary_domain.value,
                "arxiv_suitable": intelligent_plan.arxiv_suitable,
                "confidence": intelligent_plan.confidence_score,
                "reasoning": intelligent_plan.reasoning,
            },
        )

        # Optional post-processing step
        standardized: List[Dict[str, Any]] = []
        if process_results:
            if web_results:
                standardized.append(
                    self.web_tool.format_results(search_term, web_results)
                )
            if arxiv_results:
                standardized.append(
                    self.arxiv_tool.format_results(query, arxiv_results)
                )

        # Generate search metrics for simple search
        search_metrics = {
            "execution_time": datetime.now().isoformat(),
            "web_results_count": len(web_results),
            "arxiv_results_count": len(arxiv_results),
            "total_sources": len(web_results) + len(arxiv_results),
            "search_method": "simple_search",
            "processing_completed": process_results and bool(standardized),
            "search_success": len(web_results) > 0 or len(arxiv_results) > 0,
        }

        # Add intelligent analysis metadata to search metrics
        search_metrics.update(
            {
                "intelligent_analysis": {
                    "domain": intelligent_plan.primary_domain.value,
                    "arxiv_suitable": intelligent_plan.arxiv_suitable,
                    "academic_suitable": intelligent_plan.academic_suitable,
                    "confidence_score": intelligent_plan.confidence_score,
                    "reasoning": intelligent_plan.reasoning,
                    "search_strategies": [
                        s.value for s in intelligent_plan.search_strategies
                    ],
                }
            }
        )

        # Validate results if we have both the planner and results
        if web_results or arxiv_results:
            try:
                results_summary = f"Found {len(web_results)} web results and {len(arxiv_results)} academic papers"
                validation = self.intelligent_planner.validate_search_results(
                    query, results_summary, intelligent_plan
                )
                search_metrics["validation"] = {
                    "overall_score": validation.overall_score,
                    "relevance_score": validation.relevance_score,
                    "quality_score": validation.quality_score,
                    "credibility_score": validation.credibility_score,
                    "completeness_score": validation.completeness_score,
                    "validation_notes": validation.validation_notes,
                }
            except Exception as e:
                logger.warning(f"Search validation failed: {e}")

        return SearchResults(
            web_results=web_results,
            arxiv_results=arxiv_results,
            search_plan=plan,
            standardized_results=standardized if standardized else None,
            results_processed=process_results and bool(standardized),
            summary=f"Intelligent search completed: {len(web_results)} web + {len(arxiv_results)} arXiv results (Domain: {intelligent_plan.primary_domain.value})",
            metadata=search_metrics,
        )
