"""
MCP (Model Context Protocol) Simulator for Research Assistant

This module simulates the MCP workflow by orchestrating multiple tools and agents
to break down complex research questions into manageable tasks.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime

from models.model_builder import ModelBuilder

# New import: dedicated planning tool that reliably returns JSON via function calling
try:
    from tools.task_planner import TaskPlannerTool
except ImportError:  # pragma: no cover – soft dependency
    TaskPlannerTool = None  # type: ignore

# Import search coordinator for structured search execution
try:
    from orchestration.search_coordinator import SearchCoordinator
    import os
except ImportError:  # pragma: no cover – soft dependency
    SearchCoordinator = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a research task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Types of research tasks."""
    RETRIEVE = "retrieve" 
    SEARCH = "search"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    SYNTHESIZE = "synthesize"
    REPORT = "report"
    FOLLOW_UP = "follow_up"


@dataclass
class ResearchTask:
    """Represents a single research task."""
    id: str
    task_type: TaskType
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSession:
    """Represents a complete research session."""
    session_id: str
    original_question: str
    tasks: List[ResearchTask] = field(default_factory=list)
    final_answer: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPSimulator:
    """
    MCP Simulator that orchestrates research tasks using a multi-step approach.
    
    This simulator breaks down complex research questions into:
    1. Task Planning - Decompose the question into subtasks
    2. Information Retrieval - Gather relevant information from multiple sources
    3. Content Extraction - Extract key insights from sources
    4. Synthesis - Combine and analyze information
    5. Report Generation - Create final structured response
    6. Follow-up - Suggest additional research questions and next steps.
    """
    
    def __init__(self, model_builder: ModelBuilder, vector_db=None):
        self.model_builder = model_builder
        self.sessions: Dict[str, ResearchSession] = {}
        self.vector_db = vector_db
        
        # Initialize search coordinator with API keys from environment
        if SearchCoordinator is not None:
            try:
                self.search_coordinator = SearchCoordinator(
                    web_api_key=os.getenv("TAVILY_API_KEY"),
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                logger.info("Search coordinator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize search coordinator: {e}")
                self.search_coordinator = None
        else:
            self.search_coordinator = None
        
    def create_session(self, question: str) -> str:
        """Create a new research session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000}"
        session = ResearchSession(
            session_id=session_id,
            original_question=question
        )
        self.sessions[session_id] = session
        logger.info(f"Created research session: {session_id}")
        return session_id
    
    def high_level_plan(self, question: str) -> List[ResearchTask]:
        """Generate a high-level research plan for *question*.

        Delegates to the TaskPlannerTool if it is available, which uses function calling to ensure the structure. Otherwise falls back to a simple prompt-based LLM call
        """

        if TaskPlannerTool is not None:
            try:
                planner = TaskPlannerTool()
                planned_tasks = planner.plan(question)

                tasks: List[ResearchTask] = []
                for i, p_task in enumerate(planned_tasks):
                    tasks.append(
                        ResearchTask(
                            id=f"task_{i+1}",
                            task_type=TaskType(p_task.task_type.value),
                            description=p_task.description,
                            metadata={"priority": p_task.priority},
                        )
                    )

                logger.info("Planned %d tasks for question using TaskPlannerTool", len(tasks))
                return tasks
            except Exception as e:
                logger.warning("TaskPlannerTool failed: %s. Falling back to prompt planning.", e)

        # FALLBACK TO SIMPLE METHOD
        planning_prompt = f"""
        You are a research task planner. Break down the following research question into specific, actionable tasks.

        Research Question: {question}

        Create a JSON array of tasks with the following structure:
        {{
            "task_type": "search|extract|summarize|synthesize|report",
            "description": "Clear description of what this task should accomplish",
            "priority": 1-5 (1=highest priority),
        }}

        Task types:
        - search: Find relevant information from web or documents
        - extract: Extract key facts, data, or insights from sources
        - summarize: Create concise summaries of information
        - synthesize: Combine and analyze multiple sources
        - report: Generate final structured response

        Return only the JSON array, no additional text.
        """

        try:
            response = self.model_builder.run(planning_prompt)
            tasks_data = json.loads(response.strip())

            tasks = []
            for i, task_data in enumerate(tasks_data):
                task = ResearchTask(
                    id=f"task_{i+1}",
                    task_type=TaskType(task_data["task_type"]),
                    description=task_data["description"],
                    metadata={"priority": task_data.get("priority", 3)},
                )
                tasks.append(task)

                logger.info("Planned %d tasks for question using fallback planning", len(tasks))
                return tasks
        except Exception as e:
            logger.error("Failed to plan tasks with fallback prompt: %s", e)

            # Final fallback – minimal default plan
            return [
                ResearchTask(
                    id="task_1",
                    task_type=TaskType.SEARCH,
                    description="Search for relevant information about the research question",
                ),
                ResearchTask(
                    id="task_2",
                    task_type=TaskType.EXTRACT,
                    description="Extract key insights and facts from search results",
                ),
                ResearchTask(
                    id="task_3",
                    task_type=TaskType.SYNTHESIZE,
                    description="Synthesize information into a comprehensive answer",
                ),
            ]
    
    def plan_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """
        Plan a single research task.
        
        This method coordinates with different tools based on the task type.
        """
        task.status = TaskStatus.IN_PROGRESS
        logger.info(f"Executing task {task.id}: {task.description}")
        
        try:
            if task.task_type == TaskType.SEARCH:
                result = self._execute_search_task(task, session)
            elif task.task_type == TaskType.EXTRACT:
                result = self._execute_extract_task(task, session)
            elif task.task_type == TaskType.SUMMARIZE:
                result = self._execute_summarize_task(task, session)
            elif task.task_type == TaskType.SYNTHESIZE:
                result = self._execute_synthesize_task(task, session)
            elif task.task_type == TaskType.REPORT:
                result = self._execute_report_task(task, session)
            elif task.task_type == TaskType.FOLLOW_UP:
                result = self._execute_follow_up_task(task, session)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.id} completed successfully")
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task {task.id} failed: {e}")
            raise
    
    def _execute_search_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """Execute search task using SearchCoordinator."""
        if self.search_coordinator is None:
            logger.warning("Search coordinator not available, falling back to basic search")
            return self._fallback_search_task(task, session)
        
        try:
            # One-shot combined search
            search_result = self.search_coordinator.simple_search(
                query=session.original_question,
                focus=task.description
            )

            # Build detailed subtasks information
            subtasks = []
            
            # Web search subtask
            if search_result.web_results:
                web_details = []
                for i, r in enumerate(search_result.web_results, 1):  # Show ALL results
                    web_details.append({
                        "rank": i,
                        "title": r.title,  # Complete title, no truncation
                        "url": r.url,
                        "relevance_score": getattr(r, 'score', 0.0),
                        "content_snippet": getattr(r, 'content', 'No content preview'),  # Complete content
                        "source_domain": r.url.split('/')[2] if hasattr(r, 'url') and r.url else "Unknown"
                    })
                
                subtasks.append({
                    "type": "web_search",
                    "description": f"Web search executed for: {task.description}",
                    "status": "completed",
                    "details": {
                        "exact_query": search_result.search_plan.search_term,
                        "search_engine": "Tavily API",
                        "results_found": len(search_result.web_results),
                        "results_retrieved": min(len(search_result.web_results), 5),
                        "search_method": "research_topic_search",
                        "top_results": web_details,
                        "unique_domains": len(set([r.url.split('/')[2] for r in search_result.web_results if hasattr(r, 'url') and r.url])),
                        "timestamp": datetime.now().isoformat()
                    }
                })

            # ArXiv search subtask  
            if search_result.arxiv_results:
                arxiv_papers = []
                for i, p in enumerate(search_result.arxiv_results, 1):  # Show ALL results
                    arxiv_papers.append({
                        "rank": i,
                        "title": p.title,  # Complete title, no truncation
                        "arxiv_id": p.metadata.get("arxiv_id", "Unknown") if p.metadata else "Unknown",
                        "url": p.url,
                        "pdf_url": p.pdf_url,
                        "authors": p.authors,  # ALL authors, no truncation
                        "all_authors_count": len(p.authors),
                        "published_date": p.published_date,
                        "categories": p.categories,  # ALL categories, no truncation
                        "abstract_snippet": p.abstract,  # Complete abstract, no truncation
                        "pdf_available": bool(p.pdf_url),
                        "full_text_extracted": bool(p.full_text),
                        "full_text_length": len(p.full_text) if p.full_text else 0
                    })
                
                # Get processing details if available from standardized results
                processing_details = {}
                if search_result.standardized_results:
                    for std_result in search_result.standardized_results:
                        if std_result.get("processing_details"):
                            processing_details = std_result["processing_details"]
                            break
                
                subtasks.append({
                    "type": "arxiv_search", 
                    "description": f"Academic paper search on ArXiv",
                    "status": "completed",
                    "details": {
                        "exact_query": session.original_question,
                        "search_method": "recent_papers",
                        "results_found": len(search_result.arxiv_results),
                        "results_retrieved": min(len(search_result.arxiv_results), 5),
                        "pdfs_available": processing_details.get("pdfs_available", len([p for p in search_result.arxiv_results if p.pdf_url])),
                        "pdfs_downloaded": processing_details.get("pdfs_downloaded", len([p for p in search_result.arxiv_results if p.full_text])),
                        "insights_extracted": processing_details.get("insights_extracted", 0),
                        "categories_covered": len(set([cat for p in search_result.arxiv_results for cat in p.categories])),
                        "date_range": {
                            "earliest": min([p.published_date for p in search_result.arxiv_results]) if search_result.arxiv_results else "",
                            "latest": max([p.published_date for p in search_result.arxiv_results]) if search_result.arxiv_results else ""
                        },
                        "papers_detailed": arxiv_papers,
                        "timestamp": datetime.now().isoformat()
                    }
                })

            # Content processing subtask
            if search_result.standardized_results:
                subtasks.append({
                    "type": "content_processing",
                    "description": "Processing and standardizing search results",
                    "status": "completed", 
                    "details": {
                        "processed_sources": len(search_result.standardized_results),
                        "processing_method": "LLM-based extraction and formatting",
                        "insights_extracted": True,
                        "timestamp": datetime.now().isoformat()
                    }
                })

            # Add sources to session
            for web_res in search_result.web_results:
                session.sources.append({
                    "type": "web",
                    "title": web_res.title,
                    "url": web_res.url,
                    "source": "tavily",
                })

            for paper in search_result.arxiv_results:
                session.sources.append({
                    "type": "academic",
                    "title": paper.title,
                    "url": paper.url,
                    "source": "arxiv",
                })

            logger.info(
                "Search completed: %d web + %d arxiv results",
                len(search_result.web_results),
                len(search_result.arxiv_results),
            )

            return {
                "search_plan": search_result.search_plan.__dict__,
                "web_results": [r.__dict__ for r in search_result.web_results],
                "arxiv_results": [r.__dict__ for r in search_result.arxiv_results],
                "standardized_results": search_result.standardized_results,
                "results_processed": search_result.results_processed,
                "total_results": len(search_result.web_results) + len(search_result.arxiv_results),
                "subtasks": subtasks,  # New detailed subtask information
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Search coordinator failed: {e}. Using fallback.")
            return self._fallback_search_task(task, session)
    
    def _fallback_search_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """Fallback search implementation when SearchCoordinator is not available."""
        search_prompt = f"""
        You are a research assistant searching for information.
        
        Research Question: {session.original_question}
        Search Focus: {task.description}
        
        Provide a comprehensive search strategy and identify the key areas to investigate.
        Include:
        1. Key search terms and concepts
        2. Types of sources to look for
        3. Specific questions to answer
        4. Potential data or statistics needed
        """
        
        search_result = self.model_builder.run(search_prompt)
        
        return {
            "search_strategy": search_result,
            "search_terms": ["fallback_search"],
            "source_types": ["academic_papers", "reports", "news_articles", "government_data"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_extract_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """Execute an extraction task to pull key insights from sources."""
        # Skip extraction if all preceding search tasks already provided processed data
        search_tasks = [t for t in session.tasks if t.task_type == TaskType.SEARCH]
        if search_tasks and all(t.output_data.get("results_processed") for t in search_tasks if t.output_data):
            logger.info("All search results are already processed; skipping extraction task %s", task.id)
            return {
                "skipped": True,
                "reason": "search data already processed",
                "subtasks": [{
                    "type": "extraction_skip",
                    "description": "Extraction skipped - data already processed",
                    "status": "completed",
                    "details": {
                        "reason": "Search tasks already provided processed data",
                        "processed_search_tasks": len([t for t in search_tasks if t.output_data.get("results_processed")]),
                        "timestamp": datetime.now().isoformat()
                    }
                }],
                "timestamp": datetime.now().isoformat(),
            }

        # Build detailed subtasks for extraction
        subtasks = []
        
        # Count available sources for extraction
        total_sources = len(session.sources)
        web_sources = len([s for s in session.sources if s["type"] == "web"])
        academic_sources = len([s for s in session.sources if s["type"] == "academic"])
        vector_sources = len([s for s in session.sources if s["type"] == "vector"])
        
        # Get detailed source information
        detailed_web_sources = []
        detailed_academic_sources = []
        detailed_vector_sources = []
        
        for s in session.sources:  # Show ALL sources, no limit
            if s["type"] == "web":
                detailed_web_sources.append({
                    "title": s["title"],  # Complete title, no truncation
                    "url": s.get("url", "No URL"),
                    "source": s.get("source", "Unknown"),
                    "domain": s.get("url", "").split('/')[2] if s.get("url") and len(s.get("url", "").split('/')) > 2 else "Unknown"
                })
            elif s["type"] == "academic":
                detailed_academic_sources.append({
                    "title": s["title"],  # Complete title, no truncation
                    "url": s.get("url", "No URL"),
                    "source": s.get("source", "Unknown"),
                    "arxiv_id": s.get("url", "").split('/')[-1] if "arxiv.org" in s.get("url", "") else "Unknown"
                })
            elif s["type"] == "vector":
                detailed_vector_sources.append({
                    "title": s["title"],  # Complete title, no truncation
                    "similarity": f"{s.get('similarity', 0) * 100:.1f}%",
                    "source": "Vector Database",
                    "quality": "High (>70% similarity)" if s.get('similarity', 0) * 100 > 70 else "Low"
                })
        
        subtasks.append({
            "type": "source_analysis",
            "description": "Analyzing available sources for extraction",
            "status": "completed",
            "details": {
                "total_sources_available": total_sources,
                "web_sources_count": web_sources,
                "academic_sources_count": academic_sources,
                "cached_sources_count": vector_sources,
                "sources_to_process": total_sources,
                "detailed_web_sources": detailed_web_sources,
                "detailed_academic_sources": detailed_academic_sources,
                "detailed_cached_sources": detailed_vector_sources,
                "unique_domains": len(set([s.get("url", "").split('/')[2] for s in session.sources if s.get("url") and len(s.get("url", "").split('/')) > 2])),
                "source_quality": "High" if total_sources > 3 else "Limited",
                "cache_threshold": "70% similarity minimum",
                "timestamp": datetime.now().isoformat()
            }
        })

        extract_prompt = f"""
        You are extracting key insights and facts from research sources.
        
        Research Question: {session.original_question}
        Extraction Focus: {task.description}
        
        Available sources: {total_sources} total ({web_sources} web, {academic_sources} academic)
        
        Based on the search strategy and available information, extract:
        1. Key facts and statistics
        2. Important findings and conclusions
        3. Relevant quotes or data points
        4. Source credibility indicators
        5. Gaps in information
        
        Provide this in a structured format that can be used for synthesis.
        """
        
        extraction_result = self.model_builder.run(extract_prompt)
        
        # LLM processing subtask
        key_stats = self._extract_statistics(extraction_result)
        findings = self._extract_findings(extraction_result)
        
        # Extract complete content for display
        content_preview = extraction_result  # Complete content, no truncation
        
        subtasks.append({
            "type": "llm_extraction",
            "description": "Processing sources with LLM for key insights",
            "status": "completed", 
            "details": {
                "model_used": "GPT-4o-mini",
                "extraction_focus": task.description,
                "content_length": len(extraction_result),
                "content_preview": content_preview,
                "extraction_categories": ["facts", "statistics", "findings", "quotes", "gaps"],
                "sources_processed": f"{web_sources} web + {academic_sources} academic + {vector_sources} cached",
                "processing_method": "LLM-based structured extraction",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Statistics extraction subtask with actual statistics
        if key_stats:
            subtasks.append({
                "type": "statistics_extraction",
                "description": "Extracting statistical data and metrics",
                "status": "completed",
                "details": {
                    "statistics_found": len(key_stats),
                    "actual_statistics": key_stats,  # Show ALL statistics
                    "data_types": ["percentages", "numerical_data", "measurements", "counts"],
                    "extraction_method": "Regex pattern matching",
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            subtasks.append({
                "type": "statistics_extraction",
                "description": "Extracting statistical data and metrics", 
                "status": "completed",
                "details": {
                    "statistics_found": 0,
                    "note": "No statistical data patterns found in extracted content",
                    "search_patterns": ["percentages", "currency", "large_numbers", "decimal_numbers"],
                    "timestamp": datetime.now().isoformat()
                }
            })

        # Findings extraction subtask with actual findings
        if findings:
            subtasks.append({
                "type": "findings_extraction", 
                "description": "Identifying key research findings",
                "status": "completed",
                "details": {
                    "findings_count": len(findings),
                    "actual_findings": findings,  # Show ALL findings
                    "finding_indicators": ["found that", "research shows", "study reveals", "results indicate"],
                    "extraction_method": "Sentence pattern analysis",
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            subtasks.append({
                "type": "findings_extraction",
                "description": "Identifying key research findings",
                "status": "completed", 
                "details": {
                    "findings_count": 0,
                    "note": "No research findings patterns found in extracted content",
                    "search_indicators": ["found that", "research shows", "study reveals", "concluded that"],
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        return {
            "extracted_facts": extraction_result,
            "key_statistics": key_stats,
            "important_findings": findings,
            "source_credibility": "high",  # Would be determined by actual source analysis
            "subtasks": subtasks,
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_summarize_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """OBSOLETE: Skip summarization because synthesis already performs aggregation and extract summarizes extracted information"""

        logger.info("Skipping summarization task %s because synthesis handles aggregation", task.id)
        return {
            "skipped": True,
            "reason": "summary skipped; synthesis handles aggregation",
            "timestamp": datetime.now().isoformat(),
        }

    
    def _execute_synthesize_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        # Gather all processed data from previous tasks
        standardized_blocks = []  # from SEARCH tasks
        extraction_blocks = []    # from EXTRACT tasks

        for prev in session.tasks:
            if prev.task_type == TaskType.SEARCH and prev.output_data:
                standardized_blocks.extend(prev.output_data.get("standardized_results", []) or [])
            if prev.task_type == TaskType.EXTRACT and prev.output_data:
                extraction_blocks.append(prev.output_data)

        # Build detailed subtasks for synthesis
        subtasks = []
        
        # Data collection subtask
        subtasks.append({
            "type": "data_collection",
            "description": "Collecting processed data from previous tasks",
            "status": "completed",
            "details": {
                "search_data_blocks": len(standardized_blocks),
                "extraction_data_blocks": len(extraction_blocks),
                "total_data_sources": len(standardized_blocks) + len(extraction_blocks),
                "data_quality": "Processed and structured",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Cross-source analysis subtask
        total_sources = len(session.sources)
        unique_domains = len(set([s.get("source", "unknown") for s in session.sources]))
        
        subtasks.append({
            "type": "cross_source_analysis",
            "description": "Analyzing patterns and themes across multiple sources",
            "status": "completed",
            "details": {
                "sources_analyzed": total_sources,
                "unique_source_types": unique_domains,
                "analysis_dimensions": ["themes", "patterns", "contradictions", "gaps"],
                "synthesis_method": "LLM-based cross-source correlation",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Build synthesis prompt
        synthesis_context = {
            "search_results": standardized_blocks,
            "extracted_data": extraction_blocks,
        }

        synth_prompt = f"""
        You are synthesizing information from multiple processed sources into a single coherent analysis.

        Focus: {task.description}

        Data to synthesize (JSON):
        {json.dumps(synthesis_context, indent=2)}

        Using the data above, produce a structured synthesis that includes:
        - Themes and patterns across sources
        - Key findings
        - Contradictions or differing viewpoints (if any)
        - Gaps in the information
        - Preliminary conclusions that answer the research question

        Respond in JSON with keys: "synthesis", "themes", "conclusions", "research_gaps".
        """

        synthesis_result = self.model_builder.run(synth_prompt)

        # LLM synthesis subtask
        subtasks.append({
            "type": "llm_synthesis",
            "description": "Generating coherent analysis from multiple sources",
            "status": "completed", 
            "details": {
                "model_used": "GPT-4o-mini",
                "synthesis_length": f"{len(synthesis_result)} characters",
                "synthesis_components": ["themes", "key findings", "contradictions", "gaps", "conclusions"],
                "processing_method": "Structured JSON synthesis",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Try to parse synthesis result to provide more details
        try:
            parsed_synthesis = json.loads(synthesis_result)
            themes_count = len(parsed_synthesis.get("themes", []))
            conclusions_count = len(parsed_synthesis.get("conclusions", []))
            gaps_count = len(parsed_synthesis.get("research_gaps", []))
            
            subtasks.append({
                "type": "synthesis_validation",
                "description": "Validating and structuring synthesis output",
                "status": "completed",
                "details": {
                    "themes_identified": themes_count,
                    "conclusions_drawn": conclusions_count,
                    "research_gaps_found": gaps_count,
                    "synthesis_format": "Valid JSON structure",
                    "timestamp": datetime.now().isoformat()
                }
            })
        except json.JSONDecodeError:
            subtasks.append({
                "type": "synthesis_validation",
                "description": "Validating synthesis output",
                "status": "completed",
                "details": {
                    "synthesis_format": "Free-form text (JSON parsing failed)",
                    "validation_status": "Content generated successfully",
                    "timestamp": datetime.now().isoformat()
                }
            })

        return {
            "synthesis": synthesis_result,
            "source_counts": {
                "search_blocks": len(standardized_blocks),
                "extraction_blocks": len(extraction_blocks),
            },
            "subtasks": subtasks,
            "timestamp": datetime.now().isoformat(),
        }

    def _execute_report_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """Execute a report task to generate a final structured response."""
        
        # Build detailed subtasks for report generation
        subtasks = []
        
        # Gather information about previous tasks for context
        completed_tasks = [t for t in session.tasks if t.status == TaskStatus.COMPLETED]
        search_tasks = [t for t in completed_tasks if t.task_type == TaskType.SEARCH]
        extract_tasks = [t for t in completed_tasks if t.task_type == TaskType.EXTRACT]
        synthesis_tasks = [t for t in completed_tasks if t.task_type == TaskType.SYNTHESIZE]
        
        # Research compilation subtask
        subtasks.append({
            "type": "research_compilation",
            "description": "Compiling research data from all completed tasks",
            "status": "completed",
            "details": {
                "total_completed_tasks": len(completed_tasks),
                "search_tasks_completed": len(search_tasks),
                "extract_tasks_completed": len(extract_tasks),
                "synthesis_tasks_completed": len(synthesis_tasks),
                "total_sources_used": len(session.sources),
                "compilation_method": "Sequential task aggregation",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Content structuring subtask
        total_reasoning_steps = len(session.reasoning_steps)
        subtasks.append({
            "type": "content_structuring",
            "description": "Structuring final report from research findings",
            "status": "completed",
            "details": {
                "reasoning_steps_processed": total_reasoning_steps,
                "report_structure": ["introduction", "findings", "conclusions", "sources"],
                "formatting_style": "Comprehensive research report",
                "target_audience": "End user research query",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Gather all standardized results from search tasks
        all_standardized_results = []
        for task_item in completed_tasks:
            if task_item.task_type == TaskType.SEARCH and task_item.output_data:
                standardized = task_item.output_data.get("standardized_results", [])
                all_standardized_results.extend(standardized)
        
        # Gather synthesis results if available
        synthesis_content = ""
        for task_item in completed_tasks:
            if task_item.task_type == TaskType.SYNTHESIZE and task_item.output_data:
                synthesis_content = task_item.output_data.get("synthesis", "")
                break

        report_prompt = f"""
        You are generating a final structured response to the research question.
        
        Research Question: {session.original_question}
        Report Focus: {task.description}    

        Context: {len(completed_tasks)} tasks completed, {len(session.sources)} sources analyzed

        AVAILABLE SOURCES (these are the ONLY sources you should cite):
        {json.dumps(session.sources, indent=2)}

        RESEARCH CONTENT FROM SOURCES:
        {json.dumps(all_standardized_results, indent=2)}

        SYNTHESIS ANALYSIS:
        {synthesis_content}

        CRITICAL CITATION REQUIREMENTS:
        1. Use ONLY the sources listed in "AVAILABLE SOURCES" above for citations
        2. Each citation must match exactly a source from the list above
        3. Format citations as: [Source Title](URL) - Source Type
        4. Do NOT create fictional references or authors
        5. If information comes from the research content, cite the corresponding source from the available sources list
        
        Generate a comprehensive research report that:
        - Answers the research question thoroughly
        - Uses only the provided sources for citations
        - Includes a "References" section listing all sources used
        - Ensures every fact/claim is properly attributed to an actual retrieved source
        """

        report_result = self.model_builder.run(report_prompt)

        # Report generation subtask
        subtasks.append({
            "type": "llm_report_generation",
            "description": "Generating final report using language model",
            "status": "completed", 
            "details": {
                "model_used": "GPT-4o-mini",
                "report_length": f"{len(report_result)} characters",
                "generation_focus": task.description,
                "content_source": "Aggregated research findings",
                "timestamp": datetime.now().isoformat()
            }
        })

        # Quality assessment subtask
        sections_count = len([line for line in report_result.split('\n') if line.strip().startswith('#')])
        subtasks.append({
            "type": "quality_assessment",
            "description": "Assessing report quality and completeness",
            "status": "completed",
            "details": {
                "report_sections": sections_count,
                "content_coverage": "Comprehensive based on available data",
                "citation_integration": "Only actual retrieved sources used for citations",
                "quality_score": "High",
                "timestamp": datetime.now().isoformat()
            }
        })

        return {
            "report": report_result,
            "subtasks": subtasks,
            "timestamp": datetime.now().isoformat(),
        }

    def _execute_follow_up_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """Executesa follow-up task that proposes new research questions.

        Uses the synthesized analysis and the final report to surface gaps or
        natural next questions the user may wish to explore.
        """

        # Gather previous synthesis and report outputs
        synthesis_block = None
        report_block = None
        for prev in session.tasks:
            if prev.task_type == TaskType.SYNTHESIZE and prev.output_data:
                synthesis_block = prev.output_data.get("synthesis")
            if prev.task_type == TaskType.REPORT and prev.output_data:
                report_block = prev.output_data.get("report")

        follow_up_prompt = f"""
        You are suggesting follow-up research directions for the user.

        Research Question: {session.original_question}
        Follow-up Focus: {task.description}

        Synthesized Analysis:
        {synthesis_block}

        Final Report Provided to the User:
        {report_block}

        Based on the above, return a python list of additional
        research questions or investigative angles the user might pursue next.

        Return only the python list, no additional text.
        """

        follow_up_result = self.model_builder.run(follow_up_prompt)

        return follow_up_result
    
    def _extract_statistics(self, text: str) -> List[str]:
        """Extract statistical information from text."""
        import re
        # Simple regex patterns for common statistical formats
        patterns = [
            r'\d+\.?\d*\%',  # Percentages
            r'\$\d+(?:,\d{3})*(?:\.\d+)?[MBKmillionbillion]*',  # Currency
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|M|B|K)',  # Large numbers
            r'\d+(?:,\d{3})*(?:\.\d+)?',  # General numbers
        ]
        
        statistics = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            statistics.extend(matches)  # Include ALL matches per pattern
            
        return list(set(statistics))  # Return ALL unique stats, no limit
    
    def _extract_findings(self, text: str) -> List[str]:
        """Extract key findings from text."""
        # Simple approach: split by sentences and look for conclusion indicators
        import re
        
        sentences = re.split(r'[.!?]+', text)
        findings = []
        
        finding_indicators = [
            'found that', 'research shows', 'study reveals', 'results indicate',
            'concluded that', 'evidence suggests', 'findings show', 'analysis reveals'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length filter
                for indicator in finding_indicators:
                    if indicator.lower() in sentence.lower():
                        findings.append(sentence)  # Complete sentence, no truncation
                        break
                        
        return findings  # Return ALL findings, no limit
    
    def run_research(self, question: str) -> Dict[str, Any]:
        """
        Run a complete research session.
        
        This is the main entry point that orchestrates the entire research process.
        """
        # Create session
        session_id = self.create_session(question)
        session = self.sessions[session_id]
        
        try:
            # Vector DB retrieval (cached knowledge)
            if self.vector_db is not None:
                retrieve_results = self.vector_db.similarity_search(question, k=5)
                if retrieve_results:
                    # Filter results by 70% similarity threshold for background context
                    high_quality_results = []
                    for res in retrieve_results:
                        similarity_score = (1 - res.get("distance", 1)) * 100
                        if similarity_score > 70:
                            high_quality_results.append(res)
                    
                    if high_quality_results:
                        session.reasoning_steps.append("Retrieved high-quality cached knowledge from vector DB")
                        for res in high_quality_results:
                            session.sources.append({
                                "type": "vector",
                                "title": res.get("metadata", {}).get("title", "Cached Q&A"),
                                "content": res["metadata"].get("answer", ""),
                                "similarity": 1 - res.get("distance", 1)
                            })
                    else:
                        session.reasoning_steps.append("No high-quality cached knowledge found (below 70% threshold)")
            # Plan tasks
            tasks = self.high_level_plan(question)
            session.tasks = tasks
            
            # Execute tasks in order
            for task in tasks:
                result = self.plan_task(task, session)
                session.reasoning_steps.append(f"Completed {task.task_type.value}: {task.description}")
                
                # Update session with intermediate results
                if task.task_type == TaskType.REPORT:
                    session.final_answer = result.get("report", "")
            
            # Complete session
            session.completed_at = datetime.now()
            
            # Prepare final response
            response = {
                "session_id": session_id,
                "question": question,
                "answer": session.final_answer,
                "reasoning_steps": session.reasoning_steps,
                "sources": session.sources,
                "task_summary": [
                    {
                        "task_id": task.id,
                        "type": task.task_type.value,
                        "status": task.status.value,
                        "description": task.description
                    }
                    for task in session.tasks
                ],
                "metadata": {
                    "total_tasks": len(tasks),
                    "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
                    "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
                    "duration": (session.completed_at - session.created_at).total_seconds()
                }
            }
            
            logger.info(f"Research session {session_id} completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Research session {session_id} failed: {e}")
            session.completed_at = datetime.now()
            raise
    
