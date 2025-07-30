"""
Research Assistant - MCP Style
Enhanced version with MCP simulator and tools integration
"""

import gradio as gr
import os
import json
from dotenv import load_dotenv

# Import the model builder and MCP simulator
from models.model_builder import ModelBuilder, create_openai_model
from orchestration.mcp_simulator import MCPSimulator
from tools.web_search import WebSearchTool
from tools.document_processor import DocumentProcessor

load_dotenv()

def create_research_assistant():
    """
    Create and configure the research assistant with MCP workflow.
    """
    # Create model builder
    model = (ModelBuilder()
            .with_provider("openai")
            .with_model("gpt-4o-mini")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_system_prompt("""You are an expert research assistant using MCP (Model Context Protocol) workflow.
            Your role is to:
            1. Break down complex research questions into manageable tasks
            2. Coordinate multiple tools and agents to gather information
            3. Synthesize findings into comprehensive, well-structured responses
            4. Provide traceable reasoning and source citations
            5. Identify gaps and suggest further research areas
            
            Always provide clear, actionable insights with proper structure and formatting.""")
            .build())
    
    # Create MCP simulator
    mcp_simulator = MCPSimulator(model)
    
    # Create tools
    web_search_tool = WebSearchTool()
    document_processor = DocumentProcessor()
    
    return model, mcp_simulator, web_search_tool, document_processor

def research_assistant(question: str, show_reasoning: bool = True) -> str:
    """
    Enhanced research assistant using MCP workflow.
    """
    if not question.strip():
        return "Please ask me a research question!"
    
    try:
        # Initialize components
        model, mcp_simulator, web_search_tool, document_processor = create_research_assistant()
        
        # Run MCP research workflow
        research_result = mcp_simulator.run_research(question)
        
        # Format the response
        response_parts = []
        
        # Main answer
        if research_result.get("answer"):
            response_parts.append("## 🔬 Research Results\n")
            response_parts.append(research_result["answer"])
        
        # Reasoning steps (if requested)
        if show_reasoning and research_result.get("reasoning_steps"):
            response_parts.append("\n## 🧠 Research Process\n")
            for i, step in enumerate(research_result["reasoning_steps"], 1):
                response_parts.append(f"{i}. {step}")
        
        # Task summary
        if research_result.get("task_summary"):
            response_parts.append("\n## 📋 Task Summary\n")
            for task in research_result["task_summary"]:
                status_emoji = "✅" if task["status"] == "completed" else "❌"
                response_parts.append(f"{status_emoji} **{task['type'].title()}**: {task['description']}")
        
        # Metadata
        if research_result.get("metadata"):
            metadata = research_result["metadata"]
            response_parts.append(f"\n## 📊 Research Metrics\n")
            response_parts.append(f"- **Total Tasks**: {metadata.get('total_tasks', 0)}")
            response_parts.append(f"- **Completed**: {metadata.get('completed_tasks', 0)}")
            response_parts.append(f"- **Failed**: {metadata.get('failed_tasks', 0)}")
            response_parts.append(f"- **Duration**: {metadata.get('duration', 0):.1f} seconds")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"""
        ❌ **Error**: Unable to process your research question at the moment.
        
        **Technical Details**: {str(e)}
        
        **Troubleshooting**:
        - Make sure you have set up your OpenAI API key in the .env file
        - Check your internet connection
        - Verify that the MCP simulator is properly configured
        """

def advanced_research_assistant(question: str, search_depth: str = "advanced", include_academic: bool = True) -> str:
    """
    Advanced research assistant with more detailed workflow and tool integration.
    """
    if not question.strip():
        return "Please ask me a research question!"
    
    try:
        # Initialize components
        model, mcp_simulator, web_search_tool, document_processor = create_research_assistant()
        
        # Step 1: Web Search
        search_results = web_search_tool.search_research_topic(question, include_academic=include_academic)
        
        # Step 2: Process search results
        processed_documents = []
        for result in search_results[:5]:  # Process top 5 results
            doc_analysis = document_processor.process_text(
                result.content, 
                f"search_result_{result.title[:30]}"
            )
            processed_documents.append(doc_analysis)
        
        # Step 3: Extract cross-document insights
        insights = document_processor.extract_key_insights(processed_documents)
        
        # Step 4: Generate comprehensive response using MCP workflow
        enhanced_question = f"""
        Research Question: {question}
        
        Additional Context from Web Search:
        - Found {len(search_results)} relevant sources
        - Key themes: {', '.join(insights.get('common_themes', []))}
        - Supporting evidence: {len(insights.get('supporting_evidence', []))} pieces
        - Research gaps: {len(insights.get('gaps', []))} identified
        - Confidence score: {insights.get('confidence_score', 0):.2f}
        
        Please provide a comprehensive research answer incorporating this information.
        """
        
        research_result = mcp_simulator.run_research(enhanced_question)
        
        # Format detailed response
        response_parts = []
        
        # Main answer
        if research_result.get("answer"):
            response_parts.append("## 🔬 Comprehensive Research Results\n")
            response_parts.append(research_result["answer"])
        
        # Search summary
        if search_results:
            response_parts.append("\n## 🔍 Information Sources\n")
            search_summary = web_search_tool.get_search_summary(search_results)
            response_parts.append(search_summary)
        
        # Key insights
        if insights:
            response_parts.append("\n## 💡 Key Insights\n")
            if insights.get("common_themes"):
                response_parts.append("**Common Themes:**")
                for theme in insights["common_themes"]:
                    response_parts.append(f"- {theme}")
            
            if insights.get("supporting_evidence"):
                response_parts.append("\n**Supporting Evidence:**")
                for evidence in insights["supporting_evidence"][:3]:
                    response_parts.append(f"- {evidence[:100]}...")
            
            if insights.get("gaps"):
                response_parts.append("\n**Research Gaps:**")
                for gap in insights["gaps"][:3]:
                    response_parts.append(f"- {gap}")
        
        # Research process
        if research_result.get("reasoning_steps"):
            response_parts.append("\n## 🧠 Research Process\n")
            for i, step in enumerate(research_result["reasoning_steps"], 1):
                response_parts.append(f"{i}. {step}")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"""
        ❌ **Error**: Unable to process your research question at the moment.
        
        **Technical Details**: {str(e)}
        
        **Troubleshooting**:
        - Make sure you have set up your API keys in the .env file
        - Check your internet connection
        - Verify that all tools are properly configured
        """

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 Research Assistant")
        gr.Markdown("Ask me complex research questions and I'll help you find comprehensive answers using AI-powered MCP workflow!")
        
        with gr.Tabs():
            # Basic Research Tab
            with gr.TabItem("🔍 Basic Research"):
                gr.Markdown("### Quick research with MCP workflow")
                
                question_input = gr.Textbox(
                    label="Your Research Question",
                    placeholder="e.g., What are the most effective global health programs for malaria in sub-Saharan Africa?",
                    lines=3
                )
                
                show_reasoning = gr.Checkbox(
                    label="Show reasoning steps",
                    value=True,
                    info="Display the step-by-step research process"
                )
                
                submit_btn = gr.Button("🚀 Research", variant="primary")
                
                output = gr.Markdown(label="Research Results")
                
                submit_btn.click(
                    fn=research_assistant,
                    inputs=[question_input, show_reasoning],
                    outputs=[output]
                )
            
            # Advanced Research Tab
            with gr.TabItem("⚡ Advanced Research"):
                gr.Markdown("### Advanced research with web search and document analysis")
                
                adv_question_input = gr.Textbox(
                    label="Your Research Question",
                    placeholder="e.g., How do renewable energy costs compare globally in 2024?",
                    lines=3
                )
                
                with gr.Row():
                    search_depth = gr.Dropdown(
                        choices=["basic", "advanced"],
                        value="advanced",
                        label="Search Depth",
                        info="Level of search detail"
                    )
                    
                    include_academic = gr.Checkbox(
                        label="Include Academic Sources",
                        value=True,
                        info="Search academic databases and research papers"
                    )
                
                adv_submit_btn = gr.Button("🔬 Advanced Research", variant="primary")
                
                adv_output = gr.Markdown(label="Advanced Research Results")
                
                adv_submit_btn.click(
                    fn=advanced_research_assistant,
                    inputs=[adv_question_input, search_depth, include_academic],
                    outputs=[adv_output]
                )
        
        # Add example questions
        gr.Markdown("## 📝 Example Questions")
        gr.Examples(
            examples=[
                ["What are the most effective global health programs for malaria in sub-Saharan Africa?"],
                ["How do renewable energy costs compare globally in 2024?"],
                ["What are the key factors in successful startup fundraising?"],
                ["What are the latest developments in quantum computing?"],
                ["How does climate change affect agricultural productivity?"],
                ["What are the emerging trends in artificial intelligence for healthcare?"],
                ["How effective are different COVID-19 vaccination strategies?"],
                ["What are the economic impacts of remote work adoption?"]
            ],
            inputs=[question_input]
        )
        
        # Add information about the system
        gr.Markdown("""
        ## 🛠️ System Information
        
        This research assistant uses:
        - **MCP (Model Context Protocol) Simulator**: Orchestrates multi-step research workflows
        - **Web Search Tool**: Finds relevant information from the internet
        - **Document Processor**: Analyzes and extracts insights from content
        - **LLM Integration**: Uses OpenAI models for reasoning and synthesis
        
        The system breaks down complex questions into manageable tasks and coordinates multiple tools to provide comprehensive, well-structured research answers.
        """)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Research Assistant...")
    print("🔧 Using MCP Simulator for research orchestration...")
    print("🌐 Web search tool operational...")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )