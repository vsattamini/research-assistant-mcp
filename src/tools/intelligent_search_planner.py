"""
Intelligent Search Planner for Research Assistant

This module provides an LLM-powered search planning system that analyzes research questions
and determines optimal search strategies, replacing the limited ArXiv-only approach.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.models.model_builder import ModelBuilder

logger = logging.getLogger(__name__)


class SearchDomain(str, Enum):
    """Domains for different types of research questions."""

    TECHNOLOGY = "technology"
    SCIENCE = "science"
    MEDICINE = "medicine"
    ECONOMICS = "economics"
    POLITICS = "politics"
    SOCIAL = "social"
    BUSINESS = "business"
    ACADEMIC = "academic"
    CURRENT_EVENTS = "current_events"
    STATISTICS = "statistics"
    GENERAL = "general"


class SearchStrategy(str, Enum):
    """Different search strategies to employ."""

    WEB_COMPREHENSIVE = "web_comprehensive"
    NEWS_FOCUSED = "news_focused"
    ACADEMIC_PAPERS = "academic_papers"
    STATISTICAL_DATA = "statistical_data"
    EXPERT_SOURCES = "expert_sources"
    GOVERNMENT_DATA = "government_data"
    COMPANY_REPORTS = "company_reports"
    MIXED_APPROACH = "mixed_approach"


@dataclass
class SearchPlan:
    """Represents an intelligent search plan."""

    primary_domain: SearchDomain
    search_strategies: List[SearchStrategy]
    search_terms: List[str]
    academic_suitable: bool
    arxiv_suitable: bool
    focus_areas: List[str]
    validation_criteria: Dict[str, Any]
    confidence_score: float = 0.0
    reasoning: str = ""


@dataclass
class SearchValidation:
    """Represents validation results for search results."""

    relevance_score: float
    quality_score: float
    credibility_score: float
    completeness_score: float
    overall_score: float
    validation_notes: str
    suggested_improvements: List[str]


class IntelligentSearchPlanner:
    """
    LLM-powered search planner that analyzes research questions and creates
    optimal search strategies with validation capabilities.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the intelligent search planner."""
        self.model_builder = None
        self.openai_client = None

        # Try to initialize model builder
        try:
            self.model_builder = (
                ModelBuilder()
                .with_provider("openai")
                .with_model("gpt-4.1-nano")
                .with_temperature(0.3)
                .build()
            )
            logger.info("Model builder initialized for search planning")
        except Exception as e:
            logger.warning(f"Failed to initialize model builder: {e}")

        # Try to initialize OpenAI client for function calling
        if OPENAI_AVAILABLE and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            logger.warning("OpenAI client not available for structured planning")

        # Function tool for search planning
        self._planning_tool = {
            "type": "function",
            "function": {
                "name": "create_search_plan",
                "description": "Analyze a research question and create an optimal search strategy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_domain": {
                            "type": "string",
                            "enum": [d.value for d in SearchDomain],
                            "description": "Primary domain of the research question",
                        },
                        "search_strategies": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [s.value for s in SearchStrategy],
                            },
                            "description": "Recommended search strategies",
                        },
                        "search_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optimized search terms to use",
                        },
                        "academic_suitable": {
                            "type": "boolean",
                            "description": "Whether this topic is suitable for academic paper search",
                        },
                        "arxiv_suitable": {
                            "type": "boolean",
                            "description": "Whether this topic is suitable for ArXiv search (CS/Physics/Math/Stats)",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key focus areas to concentrate search efforts",
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence in the search strategy (0-1)",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of the search strategy reasoning",
                        },
                    },
                    "required": [
                        "primary_domain",
                        "search_strategies",
                        "search_terms",
                        "academic_suitable",
                        "arxiv_suitable",
                        "focus_areas",
                        "reasoning",
                    ],
                },
            },
        }

        # Function tool for search validation
        self._validation_tool = {
            "type": "function",
            "function": {
                "name": "validate_search_results",
                "description": "Validate and score search results for quality and relevance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relevance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "How relevant are the results to the research question",
                        },
                        "quality_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Overall quality of the information sources",
                        },
                        "credibility_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Credibility and trustworthiness of sources",
                        },
                        "completeness_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "How complete the information is for answering the question",
                        },
                        "validation_notes": {
                            "type": "string",
                            "description": "Detailed notes about the validation results",
                        },
                        "suggested_improvements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Suggestions for improving the search",
                        },
                    },
                    "required": [
                        "relevance_score",
                        "quality_score",
                        "credibility_score",
                        "completeness_score",
                        "validation_notes",
                    ],
                },
            },
        }

    def analyze_research_question(self, question: str) -> SearchPlan:
        """
        Analyze a research question and create an intelligent search plan.

        Args:
            question: The research question to analyze

        Returns:
            SearchPlan with optimized search strategy
        """
        if self.openai_client:
            return self._structured_analysis(question)
        else:
            return self._fallback_analysis(question)

    def _structured_analysis(self, question: str) -> SearchPlan:
        """Use OpenAI function calling for structured analysis."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert research strategist. Analyze research questions "
                        "and determine the optimal search approach. Consider the domain, "
                        "complexity, and information sources needed. Be realistic about "
                        "what can be found through different search methods."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this research question and create an optimal search strategy:
                    
                    Question: {question}
                    
                    Consider:
                    1. What domain does this question belong to?
                    2. What search strategies would be most effective?
                    3. Is this suitable for academic paper search?
                    4. Is this suitable for ArXiv (CS/Physics/Math/Stats topics only)?
                    5. What are the key focus areas?
                    6. What search terms would be most effective?
                    """,
                },
            ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                tools=[self._planning_tool],
                tool_choice="auto",
                temperature=0.3,
            )

            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                arguments = message.tool_calls[0].function.arguments
                data = json.loads(arguments)

                return SearchPlan(
                    primary_domain=SearchDomain(data["primary_domain"]),
                    search_strategies=[
                        SearchStrategy(s) for s in data["search_strategies"]
                    ],
                    search_terms=data["search_terms"],
                    academic_suitable=data["academic_suitable"],
                    arxiv_suitable=data["arxiv_suitable"],
                    focus_areas=data["focus_areas"],
                    validation_criteria={},
                    confidence_score=data.get("confidence_score", 0.8),
                    reasoning=data["reasoning"],
                )

        except Exception as e:
            logger.warning(f"Structured analysis failed: {e}. Using fallback.")

        return self._fallback_analysis(question)

    def _fallback_analysis(self, question: str) -> SearchPlan:
        """Fallback analysis using simple LLM prompting."""
        if not self.model_builder:
            return self._basic_fallback(question)

        try:
            analysis_prompt = f"""
            Analyze this research question and provide a search strategy:
            
            Question: {question}
            
            Provide your analysis in this JSON format:
            {{
                "primary_domain": "one of: technology, science, medicine, economics, politics, social, business, academic, current_events, statistics, general",
                "search_strategies": ["list of recommended strategies"],
                "search_terms": ["optimized search terms"],
                "academic_suitable": true/false,
                "arxiv_suitable": true/false (only for CS/Physics/Math/Stats topics),
                "focus_areas": ["key areas to focus on"],
                "reasoning": "explanation of strategy"
            }}
            
            Only return the JSON, no additional text.
            """

            response = self.model_builder.run(analysis_prompt)
            data = json.loads(response.strip())

            return SearchPlan(
                primary_domain=SearchDomain(data.get("primary_domain", "general")),
                search_strategies=[SearchStrategy("web_comprehensive")],  # Safe default
                search_terms=data.get("search_terms", [question]),
                academic_suitable=data.get("academic_suitable", False),
                arxiv_suitable=data.get("arxiv_suitable", False),
                focus_areas=data.get("focus_areas", []),
                validation_criteria={},
                confidence_score=0.7,
                reasoning=data.get("reasoning", "Fallback analysis"),
            )

        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return self._basic_fallback(question)

    def _basic_fallback(self, question: str) -> SearchPlan:
        """Most basic fallback when all else fails."""
        # Simple heuristics for domain detection
        question_lower = question.lower()

        # Determine if ArXiv suitable
        arxiv_keywords = [
            "algorithm",
            "computer",
            "software",
            "physics",
            "mathematics",
            "machine learning",
            "ai",
            "artificial intelligence",
            "quantum",
            "statistics",
            "probability",
            "neural network",
        ]
        arxiv_suitable = any(keyword in question_lower for keyword in arxiv_keywords)

        # Determine domain
        if any(
            word in question_lower
            for word in ["economy", "economic", "financial", "market"]
        ):
            domain = SearchDomain.ECONOMICS
        elif any(
            word in question_lower
            for word in ["health", "medical", "disease", "treatment"]
        ):
            domain = SearchDomain.MEDICINE
        elif any(
            word in question_lower
            for word in ["technology", "tech", "software", "computer"]
        ):
            domain = SearchDomain.TECHNOLOGY
        elif any(
            word in question_lower for word in ["recent", "latest", "current", "news"]
        ):
            domain = SearchDomain.CURRENT_EVENTS
        else:
            domain = SearchDomain.GENERAL

        return SearchPlan(
            primary_domain=domain,
            search_strategies=[SearchStrategy.WEB_COMPREHENSIVE],
            search_terms=[question],
            academic_suitable=arxiv_suitable,
            arxiv_suitable=arxiv_suitable,
            focus_areas=[],
            validation_criteria={},
            confidence_score=0.5,
            reasoning="Basic heuristic analysis",
        )

    def validate_search_results(
        self, question: str, results_summary: str, search_plan: SearchPlan
    ) -> SearchValidation:
        """
        Validate search results using LLM analysis.

        Args:
            question: Original research question
            results_summary: Summary of search results found
            search_plan: The search plan that was executed

        Returns:
            SearchValidation with detailed assessment
        """
        if self.openai_client:
            return self._structured_validation(question, results_summary, search_plan)
        else:
            return self._fallback_validation(question, results_summary, search_plan)

    def _structured_validation(
        self, question: str, results_summary: str, search_plan: SearchPlan
    ) -> SearchValidation:
        """Use OpenAI function calling for structured validation."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a research quality assessor. Evaluate search results "
                        "for relevance, quality, credibility, and completeness. "
                        "Provide constructive feedback for improvement."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
                    Validate these search results:
                    
                    Original Question: {question}
                    Search Strategy Used: {search_plan.reasoning}
                    
                    Results Summary:
                    {results_summary}
                    
                    Assess the results on:
                    1. Relevance to the research question (0-1)
                    2. Quality of information sources (0-1)  
                    3. Credibility and trustworthiness (0-1)
                    4. Completeness for answering the question (0-1)
                    5. Provide detailed validation notes
                    6. Suggest improvements if needed
                    """,
                },
            ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                tools=[self._validation_tool],
                tool_choice="auto",
                temperature=0.1,
            )

            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                arguments = message.tool_calls[0].function.arguments
                data = json.loads(arguments)

                overall_score = (
                    data["relevance_score"]
                    + data["quality_score"]
                    + data["credibility_score"]
                    + data["completeness_score"]
                ) / 4

                return SearchValidation(
                    relevance_score=data["relevance_score"],
                    quality_score=data["quality_score"],
                    credibility_score=data["credibility_score"],
                    completeness_score=data["completeness_score"],
                    overall_score=overall_score,
                    validation_notes=data["validation_notes"],
                    suggested_improvements=data.get("suggested_improvements", []),
                )

        except Exception as e:
            logger.warning(f"Structured validation failed: {e}. Using fallback.")

        return self._fallback_validation(question, results_summary, search_plan)

    def _fallback_validation(
        self, question: str, results_summary: str, search_plan: SearchPlan
    ) -> SearchValidation:
        """Fallback validation using simple assessment."""
        if not self.model_builder:
            return SearchValidation(
                relevance_score=0.7,
                quality_score=0.7,
                credibility_score=0.7,
                completeness_score=0.7,
                overall_score=0.7,
                validation_notes="Basic validation - detailed assessment not available",
                suggested_improvements=[],
            )

        try:
            validation_prompt = f"""
            Assess these search results for the research question:
            
            Question: {question}
            Results Summary: {results_summary}
            
            Rate each aspect from 0.0 to 1.0:
            1. Relevance to question
            2. Information quality  
            3. Source credibility
            4. Completeness
            
            Provide brief validation notes and suggestions.
            
            Return only this JSON format:
            {{
                "relevance_score": 0.0-1.0,
                "quality_score": 0.0-1.0,
                "credibility_score": 0.0-1.0,
                "completeness_score": 0.0-1.0,
                "validation_notes": "brief assessment",
                "suggested_improvements": ["suggestion1", "suggestion2"]
            }}
            """

            response = self.model_builder.run(validation_prompt)
            data = json.loads(response.strip())

            overall_score = (
                data["relevance_score"]
                + data["quality_score"]
                + data["credibility_score"]
                + data["completeness_score"]
            ) / 4

            return SearchValidation(
                relevance_score=data["relevance_score"],
                quality_score=data["quality_score"],
                credibility_score=data["credibility_score"],
                completeness_score=data["completeness_score"],
                overall_score=overall_score,
                validation_notes=data["validation_notes"],
                suggested_improvements=data.get("suggested_improvements", []),
            )

        except Exception as e:
            logger.error(f"Fallback validation failed: {e}")
            return SearchValidation(
                relevance_score=0.6,
                quality_score=0.6,
                credibility_score=0.6,
                completeness_score=0.6,
                overall_score=0.6,
                validation_notes=f"Validation failed: {e}",
                suggested_improvements=[
                    "Review search strategy",
                    "Try different search terms",
                ],
            )

    def suggest_search_improvements(
        self, question: str, current_results: str, validation: SearchValidation
    ) -> List[str]:
        """
        Suggest improvements to search strategy based on validation results.

        Args:
            question: Original research question
            current_results: Current search results summary
            validation: Validation assessment

        Returns:
            List of improvement suggestions
        """
        if validation.overall_score >= 0.8:
            return [
                "Search results are already high quality - consider adding more sources for comprehensiveness"
            ]

        suggestions = []

        if validation.relevance_score < 0.7:
            suggestions.append(
                "Refine search terms to be more specific to the research question"
            )
            suggestions.append("Try alternative phrasings of the research question")

        if validation.quality_score < 0.7:
            suggestions.append(
                "Look for more authoritative sources (government, academic institutions)"
            )
            suggestions.append("Search for recent publications and reports")

        if validation.credibility_score < 0.7:
            suggestions.append(
                "Focus on peer-reviewed sources and established organizations"
            )
            suggestions.append(
                "Cross-reference information across multiple reliable sources"
            )

        if validation.completeness_score < 0.7:
            suggestions.append(
                "Expand search to cover more aspects of the research question"
            )
            suggestions.append("Look for data, statistics, and specific examples")

        # Add suggestions from validation
        suggestions.extend(validation.suggested_improvements)

        return list(set(suggestions))  # Remove duplicates
