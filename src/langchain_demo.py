"""
LangChain Demo Interface - Research Assistant

This module provides a demonstration interface that shows both the custom MCP simulator
and LangChain agent approaches working with the same underlying tools.
"""

import sys
from pathlib import Path
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv

# Add the parent directory to the Python path to enable absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import both approaches
from src.models.model_builder import ModelBuilder
from src.orchestration.mcp_simulator import MCPSimulator
from src.tools.vector_db import VectorDBTool

# Import LangChain components
try:
    from src.orchestration.langchain_agent import create_langchain_research_agent

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()


def create_research_components():
    """Create both MCP and LangChain research components."""
    # Create MCP components (existing approach)
    model = (
        ModelBuilder()
        .with_provider("openai")
        .with_model("gpt-4.1-nano")
        .with_temperature(0.7)
        .with_max_tokens(1000)
        .with_system_prompt(
            """You are an expert research assistant. Provide comprehensive, 
            well-structured responses with proper source citations."""
        )
        .build()
    )

    vector_db = VectorDBTool(persist_directory="data/vector_db")
    mcp_simulator = MCPSimulator(model, vector_db=vector_db)

    # Create LangChain agent (new approach)
    langchain_agent = None
    if LANGCHAIN_AVAILABLE:
        try:
            langchain_agent = create_langchain_research_agent()
        except Exception as e:
            print(f"LangChain agent initialization failed: {e}")

    return mcp_simulator, langchain_agent


def research_with_mcp(question: str) -> str:
    """Conduct research using the custom MCP simulator."""
    if not question.strip():
        return "Please ask a research question!"

    try:
        mcp_simulator, _ = create_research_components()
        result = mcp_simulator.run_research(question)

        # Format the response
        response_parts = []

        # Main answer
        if result.get("answer"):
            response_parts.append("## 🔬 MCP Simulator Results\n")
            response_parts.append(result["answer"])

        # Task summary
        if result.get("task_summary"):
            response_parts.append("\n## 📋 Task Execution Summary\n")
            for task in result["task_summary"]:
                status_emoji = "✅" if task["status"] == "completed" else "❌"
                response_parts.append(
                    f"{status_emoji} **{task['type'].title()}**: {task['description']}"
                )

        # Sources
        if result.get("sources"):
            response_parts.append("\n## 📚 Sources\n")
            for src in result["sources"]:
                title = src.get("title", "Untitled Source")
                url = src.get("url") or src.get("pdf_url") or ""
                src_type = src.get("type", "source")
                if url:
                    response_parts.append(f"- [{title}]({url}) - {src_type}")
                else:
                    response_parts.append(f"- {title} - {src_type}")

        # Metadata
        if result.get("metadata"):
            metadata = result["metadata"]
            response_parts.append(f"\n## 📊 Execution Metrics\n")
            response_parts.append(f"- **Approach**: Custom MCP Simulator")
            response_parts.append(
                f"- **Total Tasks**: {metadata.get('total_tasks', 0)}"
            )
            response_parts.append(
                f"- **Completed**: {metadata.get('completed_tasks', 0)}"
            )
            response_parts.append(
                f"- **Duration**: {metadata.get('duration', 0):.1f} seconds"
            )

        return "\n".join(response_parts)

    except Exception as e:
        return f"❌ **MCP Research Failed**: {str(e)}"


def research_with_langchain(question: str) -> str:
    """Conduct research using the LangChain agent."""
    if not question.strip():
        return "Please ask a research question!"

    if not LANGCHAIN_AVAILABLE:
        return "❌ **LangChain Not Available**: Install with `pip install langchain langchain-openai`"

    try:
        _, langchain_agent = create_research_components()

        if not langchain_agent:
            return "❌ **LangChain Agent Failed to Initialize**: Check API keys and dependencies"

        result = langchain_agent.research(question)

        # Format the response
        response_parts = []

        # Main answer
        if result.get("answer"):
            response_parts.append("## 🤖 LangChain Agent Results\n")
            response_parts.append(result["answer"])

        # Tools used
        if result.get("tools_used"):
            response_parts.append("\n## 🛠️ Tools Utilized\n")
            for tool in result["tools_used"]:
                response_parts.append(f"- **{tool.replace('_', ' ').title()}**")

        # Reasoning steps
        if result.get("reasoning_steps"):
            response_parts.append("\n## 🧠 Agent Reasoning\n")
            for step in result["reasoning_steps"]:
                response_parts.append(f"- {step}")

        # Metadata
        if result.get("metadata"):
            metadata = result["metadata"]
            response_parts.append(f"\n## 📊 Execution Metrics\n")
            response_parts.append(f"- **Approach**: LangChain OpenAI Tools Agent")
            response_parts.append(f"- **Status**: {metadata.get('status', 'unknown')}")
            response_parts.append(f"- **Model**: {metadata.get('model', 'unknown')}")
            response_parts.append(
                f"- **Duration**: {metadata.get('duration', 0):.1f} seconds"
            )
            response_parts.append(
                f"- **Tools Available**: {metadata.get('tools_available', 0)}"
            )

        return "\n".join(response_parts)

    except Exception as e:
        return f"❌ **LangChain Research Failed**: {str(e)}"


def compare_approaches(question: str) -> tuple[str, str]:
    """Run the same question through both approaches for comparison."""
    if not question.strip():
        return "Please ask a research question!", "Please ask a research question!"

    mcp_result = research_with_mcp(question)
    langchain_result = research_with_langchain(question)

    return mcp_result, langchain_result


def create_demo_interface():
    """Create the demo interface showing both approaches."""
    with gr.Blocks(
        title="Research Assistant - MCP vs LangChain Demo", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 🔬 Research Assistant - Approach Comparison")
        gr.Markdown(
            """
        This demo shows the same research tools working with two different orchestration approaches:
        - **Custom MCP Simulator**: Our custom multi-step task orchestration system
        - **LangChain Agent**: Standard LangChain agent using OpenAI tools
        
        Both approaches use the same underlying research tools (web search, ArXiv, CSV analysis).
        """
        )

        with gr.Tab("🆚 Side-by-Side Comparison"):
            gr.Markdown("### Compare both approaches with the same question")

            question_input = gr.Textbox(
                label="Research Question",
                placeholder="e.g., What are the latest developments in quantum computing?",
                lines=2,
            )

            compare_btn = gr.Button(
                "🚀 Research with Both Approaches", variant="primary", size="lg"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔧 Custom MCP Simulator")
                    mcp_output = gr.Markdown(label="MCP Results")

                with gr.Column():
                    gr.Markdown("### 🤖 LangChain Agent")
                    langchain_output = gr.Markdown(label="LangChain Results")

            compare_btn.click(
                fn=compare_approaches,
                inputs=[question_input],
                outputs=[mcp_output, langchain_output],
            )

        with gr.Tab("🔧 MCP Simulator Only"):
            gr.Markdown("### Test the custom MCP simulator approach")

            mcp_question = gr.Textbox(
                label="Research Question",
                placeholder="e.g., How effective are renewable energy policies?",
                lines=2,
            )

            mcp_btn = gr.Button("Research with MCP", variant="primary")
            mcp_solo_output = gr.Markdown(label="MCP Results")

            mcp_btn.click(
                fn=research_with_mcp,
                inputs=[mcp_question],
                outputs=[mcp_solo_output],
            )

        with gr.Tab("🤖 LangChain Agent Only"):
            gr.Markdown("### Test the LangChain agent approach")

            lc_question = gr.Textbox(
                label="Research Question",
                placeholder="e.g., What are the economic impacts of AI automation?",
                lines=2,
            )

            lc_btn = gr.Button("Research with LangChain", variant="primary")
            lc_solo_output = gr.Markdown(label="LangChain Results")

            lc_btn.click(
                fn=research_with_langchain,
                inputs=[lc_question],
                outputs=[lc_solo_output],
            )

        with gr.Tab("ℹ️ Approach Comparison"):
            gr.Markdown(
                """
            ## 🔧 Custom MCP Simulator
            **Advantages:**
            - **Predictable**: Explicit task sequencing (SEARCH → EXTRACT → SYNTHESIZE → REPORT)
            - **Transparent**: Clear visibility into each step
            - **Optimized**: Built specifically for research workflows
            - **Detailed**: Comprehensive reasoning steps and source tracking
            
            **Trade-offs:**
            - More code to maintain
            - Custom orchestration logic
            
            ## 🤖 LangChain Agent
            **Advantages:**
            - **Standard Framework**: Uses established LangChain patterns
            - **Autonomous**: Agent decides which tools to use and when
            - **Flexible**: Can adapt tool usage based on question type
            - **Community**: Leverages LangChain ecosystem
            
            **Trade-offs:**
            - Less predictable tool execution order
            - Black-box decision making
            - Dependent on external framework
            
            ## 🎯 Same Foundation
            Both approaches use identical underlying tools:
            - **Web Search Tool** (Tavily API)
            - **ArXiv Search Tool** (Academic papers)  
            - **CSV Analysis Tool** (Statistical data)
            - **Vector Database** (Cached results)
            """
            )

        # Add example questions
        gr.Markdown("## 📝 Example Questions to Try")
        gr.Examples(
            examples=[
                ["What are the latest developments in quantum computing?"],
                ["How effective are renewable energy policies globally?"],
                ["What are the economic impacts of AI automation?"],
                ["What are the most effective malaria prevention strategies?"],
                ["How does climate change affect agricultural productivity?"],
            ],
            inputs=[question_input],
        )

    return demo


def main():
    """Main entry point for the LangChain demo."""
    print("🚀 Starting Research Assistant - MCP vs LangChain Demo...")
    print("🔧 Custom MCP Simulator available")

    if LANGCHAIN_AVAILABLE:
        print("🤖 LangChain Agent available")
    else:
        print(
            "❌ LangChain Agent not available (install with: pip install langchain langchain-openai)"
        )

    demo = create_demo_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)


if __name__ == "__main__":
    main()
