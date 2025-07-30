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
    from tools.search_coordinator import SearchCoordinator
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
    SEARCH = "search"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    SYNTHESIZE = "synthesize"
    REPORT = "report",
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
    6. Follow-up - Is there anything to follow up on? # TODO: Something like suggested next steps, or something like that.
    """
    
    def __init__(self, model_builder: ModelBuilder):
        self.model_builder = model_builder
        self.sessions: Dict[str, ResearchSession] = {}
        
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
    def plan_tasks(self, question: str) -> List[ResearchTask]:
        """Public alias for :pymeth:`high_level_plan` (deprecated).

        Existing code and tests expect a ``plan_tasks`` method. Internally it
        simply forwards to :pymeth:`high_level_plan`.
        """

        return self.high_level_plan(question)
    
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
                "timestamp": datetime.now().isoformat(),
            }

        extract_prompt = f"""
        You are extracting key insights and facts from research sources.
        
        Research Question: {session.original_question}
        Extraction Focus: {task.description}
        
        Based on the search strategy and available information, extract:
        1. Key facts and statistics
        2. Important findings and conclusions
        3. Relevant quotes or data points
        4. Source credibility indicators
        5. Gaps in information
        
        Provide this in a structured format that can be used for synthesis.
        """
        
        extraction_result = self.model_builder.run(extract_prompt)
        
        return {
            "extracted_facts": extraction_result,
            "key_statistics": self._extract_statistics(extraction_result),
            "important_findings": self._extract_findings(extraction_result),
            "source_credibility": "high",  # Would be determined by actual source analysis
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
        # TODO: this should ynthesize all the information from the search and extraction tasks so the information can be used for the report task
        """Execute a synthesis task to combine multiple sources."""
        synthesize_prompt = f"""
        You are synthesizing information from multiple sources into a coherent analysis.
        
        Research Question: {session.original_question}
        Synthesis Focus: {task.description}
        
        Combine all available information to:
        1. Identify common themes and patterns
        2. Resolve contradictions or conflicts
        3. Create a comprehensive understanding
        4. Identify gaps that need further research
        5. Form preliminary conclusions
        
        Provide a structured synthesis that builds toward a final answer.
        """
        
        synthesis_result = self.model_builder.run(synthesize_prompt)
        
        return {
            "synthesis": synthesis_result,
            "themes": self._extract_themes(synthesis_result),
            "conclusions": self._extract_conclusions(synthesis_result),
            "research_gaps": self._extract_gaps(synthesis_result),
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_report_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        # TODO: this should take in the synthesis of the information and the report task and generate a concise report with the information
        """Execute a report generation task."""
        report_prompt = f"""
        You are generating a final research report.
        
        Research Question: {session.original_question}
        Report Focus: {task.description}
        
        Create a comprehensive, well-structured research report that includes:
        1. Executive Summary
        2. Key Findings and Insights
        3. Detailed Analysis
        4. Conclusions and Recommendations
        5. Areas for Further Research
        6. Source Citations (simulated)
        
        Make the report engaging, informative, and actionable.
        Structure it clearly with headings and bullet points where appropriate.
        """
        
        report_result = self.model_builder.run(report_prompt)
        
        return {
            "report": report_result,
            "executive_summary": self._extract_executive_summary(report_result),
            "key_findings": self._extract_key_findings(report_result),
            "recommendations": self._extract_recommendations(report_result),
            "timestamp": datetime.now().isoformat()
        }
    def _execute_follow_up_task(self, task: ResearchTask, session: ResearchSession) -> Dict[str, Any]:
        """Execute a follow-up task to suggest next steps."""
        follow_up_prompt = f"""
        You are suggesting follow-up steps for research.
        
        Research Question: {session.original_question}
        Follow-up Focus: {task.description} 

        Based on the research and analysis, suggest:
        1. Additional research questions
        2. Areas for further investigation
        3. Next steps for the research process
        """

        follow_up_result = self.model_builder.run(follow_up_prompt)
        
        return {
            "follow_up": follow_up_result,
            "additional_questions": self._extract_additional_questions(follow_up_result),
            "timestamp": datetime.now().isoformat()
        }
    
    def run_research(self, question: str) -> Dict[str, Any]:
        """
        Run a complete research session.
        
        This is the main entry point that orchestrates the entire research process.
        """
        # Create session
        session_id = self.create_session(question)
        session = self.sessions[session_id]
        
        try:
            # Plan tasks
            tasks = self.plan_tasks(question)
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
    
    # Helper methods for extracting structured information JESUS I DONT THINK ALL OF THIS CAN BE INTEGRATED, I DONT THINK IT CAN BE USED FOR THE REPORT TASK, AND I DONT THINK IT CAN BE USED FOR THE SYNTHESIS TASK, LETS REPLACE THE LOGIC WITH SPECIFIC INSTANCES
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract search terms from text."""
        # TODO: THIS SHOULD GET THE SEARCH TERMS FROM THE SEARCH PLANNING, WHIOCH SHOULD ALREADY BE A LIST OF DICTS WITH THE FOLLOWING KEYS: "search_term" (STRING), "search_type" (STRING), "search_priority" (INT 0-10), "search_source" (ONE OF LIMITED OPTIONS FROM ENUM). THE LOGIC DICTATING THE RETURN FOR THIS FUNCTION ALSO HAS TO BE REWORKED
        # Simple extraction - in practice, this would be more sophisticated
        return [word.strip() for word in text.split() if len(word) > 3][:10]
