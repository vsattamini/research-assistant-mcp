"""
Tests for the MCP Simulator
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.orchestration.mcp_simulator import MCPSimulator, TaskType, TaskStatus
from src.models.model_builder import ModelBuilder


class TestMCPSimulator:
    """Test cases for the MCP Simulator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock model builder
        self.mock_model = Mock()
        self.mock_model.run.return_value = "Mock response"
        
        # Create MCP simulator with mock model
        self.simulator = MCPSimulator(self.mock_model)
    
    def test_create_session(self):
        """Test session creation."""
        question = "What is the capital of France?"
        session_id = self.simulator.create_session(question)
        
        assert session_id in self.simulator.sessions
        assert self.simulator.sessions[session_id].original_question == question
        assert self.simulator.sessions[session_id].session_id == session_id
    
    def test_plan_tasks_success(self):
        """Test successful task planning."""
        # Mock JSON response from LLM
        mock_tasks_json = '''[
            {"task_type": "search", "description": "Search for information about France", "priority": 1, "estimated_duration": "short"},
            {"task_type": "extract", "description": "Extract key facts about Paris", "priority": 2, "estimated_duration": "medium"},
            {"task_type": "synthesize", "description": "Synthesize findings into answer", "priority": 3, "estimated_duration": "long"}
        ]'''
        
        self.mock_model.run.return_value = mock_tasks_json
        
        tasks = self.simulator.plan_tasks("What is the capital of France?")
        
        assert len(tasks) == 3
        assert tasks[0].task_type == TaskType.SEARCH
        assert tasks[1].task_type == TaskType.EXTRACT
        assert tasks[2].task_type == TaskType.SYNTHESIZE
    
    def test_plan_tasks_fallback(self):
        """Test task planning fallback when JSON parsing fails."""
        # Mock invalid JSON response
        self.mock_model.run.return_value = "Invalid JSON response"
        
        tasks = self.simulator.plan_tasks("What is the capital of France?")
        
        # Should fall back to basic task structure
        assert len(tasks) == 3
        assert tasks[0].task_type == TaskType.SEARCH
        assert tasks[1].task_type == TaskType.EXTRACT
        assert tasks[2].task_type == TaskType.SYNTHESIZE
    
    def test_execute_search_task(self):
        """Test search task execution."""
        from src.orchestration.mcp_simulator import ResearchTask, ResearchSession
        
        # Create test task and session
        task = ResearchTask(
            id="test_task",
            task_type=TaskType.SEARCH,
            description="Search for information about France"
        )
        
        session = ResearchSession(
            session_id="test_session",
            original_question="What is the capital of France?"
        )
        
        # Mock model response for search
        self.mock_model.run.return_value = "Search strategy: Look for information about France and its capital city."
        
        result = self.simulator.execute_task(task, session)
        
        assert task.status == TaskStatus.COMPLETED
        assert "search_strategy" in result
        assert "search_terms" in result
        assert "source_types" in result
    
    def test_execute_extract_task(self):
        """Test extract task execution."""
        from src.orchestration.mcp_simulator import ResearchTask, ResearchSession
        
        task = ResearchTask(
            id="test_task",
            task_type=TaskType.EXTRACT,
            description="Extract key facts about Paris"
        )
        
        session = ResearchSession(
            session_id="test_session",
            original_question="What is the capital of France?"
        )
        
        # Mock model response for extraction
        self.mock_model.run.return_value = "Key facts: Paris is the capital of France. It is known for the Eiffel Tower."
        
        result = self.simulator.execute_task(task, session)
        
        assert task.status == TaskStatus.COMPLETED
        assert "extracted_facts" in result
        assert "key_statistics" in result
        assert "important_findings" in result
    
    def test_execute_synthesize_task(self):
        """Test synthesize task execution."""
        from src.orchestration.mcp_simulator import ResearchTask, ResearchSession
        
        task = ResearchTask(
            id="test_task",
            task_type=TaskType.SYNTHESIZE,
            description="Synthesize findings into answer"
        )
        
        session = ResearchSession(
            session_id="test_session",
            original_question="What is the capital of France?"
        )
        
        # Mock model response for synthesis
        self.mock_model.run.return_value = "Synthesis: Based on the research, Paris is the capital of France."
        
        result = self.simulator.execute_task(task, session)
        
        assert task.status == TaskStatus.COMPLETED
        assert "synthesis" in result
        assert "themes" in result
        assert "conclusions" in result
    
    def test_execute_report_task(self):
        """Test report task execution."""
        from src.orchestration.mcp_simulator import ResearchTask, ResearchSession
        
        task = ResearchTask(
            id="test_task",
            task_type=TaskType.REPORT,
            description="Generate final report"
        )
        
        session = ResearchSession(
            session_id="test_session",
            original_question="What is the capital of France?"
        )
        
        # Mock model response for report
        self.mock_model.run.return_value = """
        Executive Summary: Paris is the capital of France.
        
        Key Findings:
        - Paris is the capital city of France
        - It is a major cultural and economic center
        
        Conclusions: Paris serves as the political and cultural heart of France.
        """
        
        result = self.simulator.execute_task(task, session)
        
        assert task.status == TaskStatus.COMPLETED
        assert "report" in result
        assert "executive_summary" in result
        assert "key_findings" in result
    
    def test_run_research_complete_workflow(self):
        """Test complete research workflow."""
        # Mock task planning response
        mock_tasks_json = '''[
            {"task_type": "search", "description": "Search for information", "priority": 1, "estimated_duration": "short"},
            {"task_type": "synthesize", "description": "Synthesize findings", "priority": 2, "estimated_duration": "medium"},
            {"task_type": "report", "description": "Generate report", "priority": 3, "estimated_duration": "long"}
        ]'''
        
        # Mock responses for different task types
        def mock_model_run(prompt):
            if "search" in prompt.lower():
                return "Search strategy: Look for information about the topic."
            elif "synthesize" in prompt.lower():
                return "Synthesis: Based on the research, here are the findings."
            elif "report" in prompt.lower():
                return "Final Report: Comprehensive answer with all findings."
            else:
                return mock_tasks_json
        
        self.mock_model.run.side_effect = mock_model_run
        
        # Run complete research workflow
        result = self.simulator.run_research("What is the capital of France?")
        
        # Verify result structure
        assert "session_id" in result
        assert "question" in result
        assert "answer" in result
        assert "reasoning_steps" in result
        assert "task_summary" in result
        assert "metadata" in result
        
        # Verify metadata
        metadata = result["metadata"]
        assert metadata["total_tasks"] == 3
        assert metadata["completed_tasks"] == 3
        assert metadata["failed_tasks"] == 0
    
    def test_task_execution_error_handling(self):
        """Test error handling in task execution."""
        from src.orchestration.mcp_simulator import ResearchTask, ResearchSession
        
        task = ResearchTask(
            id="test_task",
            task_type=TaskType.SEARCH,
            description="Search for information"
        )
        
        session = ResearchSession(
            session_id="test_session",
            original_question="What is the capital of France?"
        )
        
        # Mock model to raise exception
        self.mock_model.run.side_effect = Exception("API Error")
        
        # Task should fail gracefully
        with pytest.raises(Exception):
            self.simulator.execute_task(task, session)
        
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "API Error"
    
    def test_helper_methods(self):
        """Test helper methods for text processing."""
        # Test search terms extraction
        text = "This is a test document with important information about research topics"
        search_terms = self.simulator._extract_search_terms(text)
        assert len(search_terms) > 0
        assert all(len(term) > 3 for term in search_terms)
        
        # Test statistics extraction
        text_with_stats = "The study found that 75% of participants showed improvement and 25% remained the same."
        statistics = self.simulator._extract_statistics(text_with_stats)
        assert len(statistics) > 0
        
        # Test findings extraction
        text_with_findings = "The research found that Paris is the capital. The study revealed important facts."
        findings = self.simulator._extract_findings(text_with_findings)
        assert len(findings) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 