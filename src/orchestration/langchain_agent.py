"""
LangChain Agent for Research Assistant

This module provides a simple LangChain agent implementation that uses the wrapped
research tools to demonstrate LangChain-based orchestration as an alternative
to the custom MCP simulator.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import AIMessage, HumanMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.orchestration.langchain_tools import create_langchain_tools

logger = logging.getLogger(__name__)


class LangChainResearchAgent:
    """
    Simple LangChain agent for research tasks.

    This demonstrates how the same research tools can be orchestrated using
    LangChain agents instead of the custom MCP simulator approach.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the LangChain research agent."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain packages not available. Install with: pip install langchain langchain-openai"
            )

        # Initialize the LLM
        try:
            self.llm = ChatOpenAI(
                model="gpt-4.1-nano",
                temperature=0.7,
                api_key=openai_api_key,
            )
            logger.info("LangChain ChatOpenAI model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise

        # Create research tools
        self.tools = create_langchain_tools()
        if not self.tools:
            logger.warning("No tools available for LangChain agent")

        # Create the agent
        self.agent_executor = self._create_agent()

    def _create_agent(self) -> Optional[AgentExecutor]:
        """Create the LangChain agent with tools."""
        if not self.tools:
            logger.error("Cannot create agent without tools")
            return None

        try:
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a research assistant that helps users find comprehensive answers to research questions.

Your approach:
1. Analyze the research question to determine what types of sources would be most helpful
2. Use multiple tools to gather information from different sources:
   - Use web_search for current information, news, and general facts
   - Use arxiv_search for academic research and scientific papers
   - Use csv_analysis for statistical data and quantitative insights
3. Synthesize the information from all sources into a comprehensive answer
4. Provide proper citations and source references

Always aim to:
- Use multiple sources to provide a well-rounded answer
- Cite your sources properly
- Highlight any gaps in the available information
- Provide actionable insights when possible

Be thorough but concise in your research and response.""",
                    ),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Create the agent
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)

            # Create the executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate",
            )

            logger.info(f"LangChain agent created with {len(self.tools)} tools")
            return agent_executor

        except Exception as e:
            logger.error(f"Failed to create LangChain agent: {e}")
            return None

    def research(self, question: str) -> Dict[str, Any]:
        """
        Conduct research using the LangChain agent.

        Args:
            question: The research question to investigate

        Returns:
            Dictionary containing the research results and metadata
        """
        if not self.agent_executor:
            return {
                "error": "LangChain agent not available",
                "question": question,
                "answer": "LangChain agent failed to initialize",
                "metadata": {
                    "status": "failed",
                    "reason": "agent_initialization_failed",
                },
            }

        start_time = datetime.now()

        try:
            logger.info(f"Starting LangChain research for: {question}")

            # Execute the agent
            result = self.agent_executor.invoke({"input": question})

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Extract the final answer
            answer = result.get("output", "No answer generated")

            # Create response in format similar to MCP simulator
            response = {
                "question": question,
                "answer": answer,
                "approach": "langchain_agent",
                "tools_used": [tool.name for tool in self.tools],
                "metadata": {
                    "status": "completed",
                    "duration": duration,
                    "approach": "LangChain OpenAI Tools Agent",
                    "model": "gpt-4.1-nano",
                    "tools_available": len(self.tools),
                    "execution_time": end_time.isoformat(),
                },
                "reasoning_steps": [
                    "Used LangChain agent to orchestrate research tools",
                    f"Available tools: {', '.join([tool.name for tool in self.tools])}",
                    "Agent autonomously decided which tools to use and how to synthesize results",
                ],
            }

            logger.info(f"LangChain research completed in {duration:.1f}s")
            return response

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.error(f"LangChain research failed: {e}")
            return {
                "error": str(e),
                "question": question,
                "answer": f"Research failed due to error: {str(e)}",
                "approach": "langchain_agent",
                "metadata": {
                    "status": "failed",
                    "duration": duration,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            }

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools] if self.tools else []

    def is_available(self) -> bool:
        """Check if the LangChain agent is properly initialized and ready."""
        return (
            LANGCHAIN_AVAILABLE
            and self.agent_executor is not None
            and len(self.tools) > 0
        )


def create_langchain_research_agent(
    openai_api_key: Optional[str] = None,
) -> Optional[LangChainResearchAgent]:
    """
    Factory function to create a LangChain research agent.

    Args:
        openai_api_key: Optional OpenAI API key

    Returns:
        LangChainResearchAgent instance or None if creation fails
    """
    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain not available")
        return None

    try:
        agent = LangChainResearchAgent(openai_api_key=openai_api_key)
        if agent.is_available():
            logger.info("LangChain research agent created successfully")
            return agent
        else:
            logger.error("LangChain agent not properly initialized")
            return None
    except Exception as e:
        logger.error(f"Failed to create LangChain research agent: {e}")
        return None
