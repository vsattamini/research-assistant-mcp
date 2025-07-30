"""
Research Assistant - MCP Style
Enhanced version with MCP simulator and tools integration
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv

# Add the parent directory to the Python path to enable absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the model builder and MCP simulator
from src.models.model_builder import ModelBuilder, create_openai_model
from src.orchestration.mcp_simulator import MCPSimulator
from src.tools.web_search import WebSearchTool
from src.utils.document_processor import DocumentProcessor
from src.tools.vector_db import VectorDBTool

load_dotenv()


def create_research_assistant():
    """
    Create and configure the research assistant with MCP workflow.
    """
    # Create model builder
    model = (
        ModelBuilder()
        .with_provider("openai")
        .with_model("gpt-4o-mini")
        .with_temperature(0.7)
        .with_max_tokens(1000)
        .with_system_prompt(
            """You are an expert research assistant using MCP (Model Context Protocol) workflow.
            Your role is to:
            1. Break down complex research questions into manageable tasks
            2. Coordinate multiple tools and agents to gather information
            3. Synthesize findings into comprehensive, well-structured responses
            4. Provide traceable reasoning and source citations
            5. Identify gaps and suggest further research areas
            
            Always provide clear, actionable insights with proper structure and formatting."""
        )
        .build()
    )

    # Create tools
    web_search_tool = WebSearchTool()
    document_processor = DocumentProcessor()

    # Persistent Vector DB for caching previous Q&A pairs
    vector_db = VectorDBTool(persist_directory="data/vector_db")

    # Create MCP simulator (pass vector_db for retrieval)
    mcp_simulator = MCPSimulator(model, vector_db=vector_db)

    # Persistent Vector DB for caching previous Q&A pairs
    vector_db = VectorDBTool(persist_directory="data/vector_db")

    return model, mcp_simulator, web_search_tool, document_processor, vector_db


def research_assistant_streaming(question: str, show_reasoning: bool = True):
    """
    Enhanced research assistant with streaming updates using MCP workflow.
    """
    if not question.strip():
        yield "Please ask me a research question!", ""
        return

    try:
        # Initialize components
        yield "🔧 **Initializing Research Assistant...**\n\nSetting up AI models and tools...", "🔧 **INITIALIZATION**\n\n▶ Setting up AI models and tools...\n▶ Loading OpenAI GPT-4o-mini model\n▶ Initializing web search capabilities\n▶ Connecting to vector database\n▶ Preparing research orchestration system\n\n"

        (
            model,
            mcp_simulator,
            web_search_tool,
            document_processor,
            vector_db,
        ) = create_research_assistant()

        yield "🔍 **Checking Cache...**\n\nSearching for previously answered similar questions...", "🔧 **INITIALIZATION COMPLETE**\n\n✅ AI models loaded successfully\n✅ Web search tools ready\n✅ Vector database connected\n✅ Research system operational\n\n🔍 **CACHE CHECK**\n\n▶ Searching vector database for similar questions...\n▶ Checking similarity threshold (>90%)\n"

        # Check vector DB cache first
        cached = vector_db.similarity_search(question, k=1)
        if (
            cached
            and (1 - cached[0]["distance"]) >= 0.9
            and cached[0]["metadata"].get("answer")
        ):
            cache_info = f"🔍 **CACHE CHECK COMPLETE**\n\n✅ Found cached answer with {(1-cached[0]['distance'])*100:.1f}% similarity\n▶ Returning cached result to save time and resources\n\n"
            yield cached[0]["metadata"]["answer"], cache_info
            return

        cache_result = f"🔍 **CACHE CHECK COMPLETE**\n\n❌ No sufficiently similar cached answers found\n▶ Proceeding with new research\n\n"

        # Start research workflow with custom callback for streaming
        yield "🚀 **Starting Research Workflow...**\n\nPlanning research tasks and gathering information...", cache_result + "🚀 **RESEARCH WORKFLOW INITIATED**\n\n▶ Creating new research session\n▶ Planning comprehensive research strategy\n"

        # Create session manually to get updates
        session_id = mcp_simulator.create_session(question)
        session = mcp_simulator.sessions[session_id]

        detailed_reasoning = [
            cache_result
            + f"🚀 **RESEARCH WORKFLOW INITIATED**\n\n✅ Research session created: {session_id}\n▶ Question: {question}\n▶ Session started at: {session.created_at.strftime('%H:%M:%S')}\n"
        ]

        # Vector DB retrieval
        if mcp_simulator.vector_db is not None:
            retrieve_results = mcp_simulator.vector_db.similarity_search(question, k=5)
            if retrieve_results:
                # Filter results by 70% similarity threshold for background context
                high_quality_results = []
                similarity_details = []

                for i, result in enumerate(retrieve_results[:3], 1):  # Show top 3
                    similarity_score = (1 - result.get("distance", 1)) * 100
                    title = result.get("metadata", {}).get(
                        "title", f"Cached Result {i}"
                    )
                    similarity_details.append(
                        f"   • **Result {i}**: {title} (Similarity: {similarity_score:.1f}%)"
                    )

                    # Only use results with >70% similarity as background context
                    if similarity_score > 70:
                        high_quality_results.append(result)

                if high_quality_results:
                    session.reasoning_steps.append(
                        "Retrieved high-quality cached knowledge from vector DB"
                    )
                    # Add high-quality cached results to session for background context
                    for result in high_quality_results:
                        session.sources.append(
                            {
                                "type": "vector",
                                "title": result.get("metadata", {}).get(
                                    "title", "Cached Q&A"
                                ),
                                "content": result["metadata"].get("answer", ""),
                                "similarity": 1 - result.get("distance", 1),
                            }
                        )

                    retrieval_details = f"\n📚 **KNOWLEDGE RETRIEVAL**\n\n✅ Found {len(retrieve_results)} total cached documents\n"
                    retrieval_details += (
                        f"▶ Cache threshold for usage: >70% similarity required\n"
                    )
                    retrieval_details += f"▶ High-quality matches: {len(high_quality_results)} documents above threshold\n"
                    retrieval_details += f"▶ Top matches found:\n" + "\n".join(
                        similarity_details
                    )
                    retrieval_details += f"\n▶ Using {len(high_quality_results)} high-quality cached documents as background context\n"
                    detailed_reasoning.append(retrieval_details)
                    yield "📚 **Retrieved High-Quality Cached Knowledge**\n\nUsing relevant cached information as background context...", "\n".join(
                        detailed_reasoning
                    )
                else:
                    retrieval_details = f"\n📚 **KNOWLEDGE RETRIEVAL**\n\n✅ Found {len(retrieve_results)} total cached documents\n"
                    retrieval_details += (
                        f"▶ Cache threshold for usage: >70% similarity required\n"
                    )
                    retrieval_details += (
                        f"▶ High-quality matches: 0 documents above threshold\n"
                    )
                    retrieval_details += f"▶ Top matches found:\n" + "\n".join(
                        similarity_details
                    )
                    retrieval_details += f"\n❌ No cached documents meet quality threshold - proceeding with fresh research only\n"
                    detailed_reasoning.append(retrieval_details)
                    yield "📚 **Cached Knowledge Check**\n\nNo high-quality cached information found - performing fresh research...", "\n".join(
                        detailed_reasoning
                    )
            else:
                retrieval_details = f"\n📚 **KNOWLEDGE RETRIEVAL**\n\n❌ No cached documents found in vector database\n▶ Proceeding with fresh research\n"
                detailed_reasoning.append(retrieval_details)
                yield "📚 **Cached Knowledge Check**\n\nNo cached information available - performing fresh research...", "\n".join(
                    detailed_reasoning
                )

        # Plan tasks
        yield "📋 **Planning Research Tasks...**\n\nBreaking down the question into manageable research tasks...", "\n".join(
            detailed_reasoning
        ) + "\n📋 **TASK PLANNING**\n\n▶ Analyzing research question complexity\n▶ Breaking down into manageable subtasks\n▶ Determining optimal research strategy\n"

        tasks = mcp_simulator.high_level_plan(question)
        session.tasks = tasks

        task_plan_details = f"\n📋 **TASK PLANNING COMPLETE**\n\n✅ Generated {len(tasks)} research tasks:\n"
        for i, task in enumerate(tasks, 1):
            task_plan_details += (
                f"   {i}. {task.task_type.value.upper()}: {task.description}\n"
            )
        task_plan_details += f"\n▶ Ready to execute planned research workflow\n"
        detailed_reasoning.append(task_plan_details)

        # Execute tasks with live updates
        for i, task in enumerate(tasks, 1):
            status_msg = f"⚡ **Executing Task {i}/{len(tasks)}**\n\n**{task.task_type.value.title()}:** {task.description}"

            task_start_details = f"\n⚡ **EXECUTING TASK {i}/{len(tasks)}**\n\n▶ Task Type: {task.task_type.value.upper()}\n▶ Description: {task.description}\n▶ Status: IN PROGRESS\n▶ Started at: {datetime.now().strftime('%H:%M:%S')}\n"
            detailed_reasoning.append(task_start_details)

            yield status_msg, "\n".join(detailed_reasoning)

            # Execute the task and capture detailed results
            result = mcp_simulator.plan_task(task, session)

            # Add detailed task completion information
            task_complete_details = f"\n✅ **TASK {i} COMPLETED**\n\n"
            task_complete_details += f"▶ Task: {task.task_type.value.upper()}\n"
            task_complete_details += f"▶ Status: {task.status.value.upper()}\n"
            task_complete_details += (
                f"▶ Completed at: {datetime.now().strftime('%H:%M:%S')}\n"
            )

            # Add specific task result details
            if task.task_type.value == "search":
                search_info = session.sources if hasattr(session, "sources") else []
                task_complete_details += (
                    f"▶ Sources found: {len(search_info)} total sources\n"
                )
                if search_info:
                    web_count = len([s for s in search_info if s.get("type") == "web"])
                    academic_count = len(
                        [s for s in search_info if s.get("type") == "academic"]
                    )
                    task_complete_details += f"▶ Web sources: {web_count}, Academic papers: {academic_count}\n"

            elif task.task_type.value == "extract":
                task_complete_details += (
                    f"▶ Key insights extracted from search results\n"
                )
                task_complete_details += f"▶ Information structured for synthesis\n"

            elif task.task_type.value == "synthesize":
                if isinstance(result, dict) and "synthesis" in result:
                    task_complete_details += f"▶ Cross-source analysis completed\n"
                    task_complete_details += f"▶ Patterns and themes identified\n"

            elif task.task_type.value == "report":
                if isinstance(result, dict) and "report" in result:
                    session.final_answer = result.get("report", "")
                    task_complete_details += f"▶ Final research report generated\n"
                    task_complete_details += (
                        f"▶ Report length: {len(session.final_answer)} characters\n"
                    )

            # Add detailed subtasks if available
            if isinstance(result, dict) and "subtasks" in result:
                task_complete_details += (
                    f"\n🔍 **DETAILED SUBTASKS** (Click to expand):\n"
                )
                for j, subtask in enumerate(result["subtasks"], 1):
                    status_emoji = "✅" if subtask.get("status") == "completed" else "⏳"
                    task_complete_details += f"\n   **{j}. {subtask.get('type', 'unknown').replace('_', ' ').title()}** {status_emoji}\n"
                    task_complete_details += (
                        f"   • {subtask.get('description', 'No description')}\n"
                    )

                    # Add specific details for each subtask
                    details = subtask.get("details", {})
                    if details:
                        if subtask.get("type") == "web_search":
                            task_complete_details += f"   • Exact Query: '{details.get('exact_query', 'N/A')}'\n"
                            task_complete_details += f"   • Search Engine: {details.get('search_engine', 'Unknown')}\n"
                            task_complete_details += f"   • Results Found: {details.get('results_found', 0)} total sources\n"
                            task_complete_details += f"   • Unique Domains: {details.get('unique_domains', 0)}\n"
                            all_results = details.get(
                                "top_results", []
                            )  # Show ALL results
                            if all_results:
                                task_complete_details += f"   • ALL Web Results Retrieved ({len(all_results)} total):\n"
                                for result_item in all_results:
                                    title = result_item.get(
                                        "title", "Unknown"
                                    )  # Complete title
                                    domain = result_item.get("source_domain", "Unknown")
                                    relevance = result_item.get("relevance_score", 0.0)
                                    url = result_item.get("url", "No URL")
                                    task_complete_details += f"     • [{result_item.get('rank', '?')}] {title}\n"
                                    task_complete_details += (
                                        f"       - Domain: {domain}\n"
                                    )
                                    task_complete_details += (
                                        f"       - Relevance: {relevance:.3f}\n"
                                    )
                                    task_complete_details += f"       - URL: {url}\n"
                                    if result_item.get("content_snippet"):
                                        content = result_item[
                                            "content_snippet"
                                        ]  # Complete content
                                        task_complete_details += (
                                            f"       - Content: {content}\n"
                                        )
                                    task_complete_details += f"\n"

                        elif subtask.get("type") == "arxiv_search":
                            task_complete_details += f"   • Exact Query: '{details.get('exact_query', 'N/A')}'\n"
                            task_complete_details += f"   • Search Method: {details.get('search_method', 'Unknown')}\n"
                            task_complete_details += f"   • Papers Found: {details.get('results_found', 0)} academic papers\n"
                            task_complete_details += f"   • PDFs Available: {details.get('pdfs_available', 0)}, Downloaded: {details.get('pdfs_downloaded', 0)}\n"
                            task_complete_details += f"   • Categories Covered: {details.get('categories_covered', 0)}\n"

                            date_range = details.get("date_range", {})
                            if date_range.get("earliest") and date_range.get("latest"):
                                task_complete_details += f"   • Date Range: {date_range['earliest'][:10]} to {date_range['latest'][:10]}\n"

                            papers = details.get(
                                "papers_detailed", []
                            )  # Show ALL papers
                            if papers:
                                task_complete_details += f"   • ALL ArXiv Papers Retrieved ({len(papers)} total):\n"
                                for paper in papers:
                                    title = paper.get(
                                        "title", "Unknown"
                                    )  # Complete title
                                    authors = ", ".join(
                                        paper.get("authors", [])
                                    )  # ALL authors
                                    categories = ", ".join(
                                        paper.get("categories", [])
                                    )  # ALL categories
                                    arxiv_id = paper.get("arxiv_id", "Unknown")
                                    published = paper.get("published_date", "Unknown")[
                                        :10
                                    ]
                                    url = paper.get("url", "No URL")
                                    pdf_url = paper.get("pdf_url", "No PDF URL")

                                    task_complete_details += (
                                        f"     • [{paper.get('rank', '?')}] {title}\n"
                                    )
                                    task_complete_details += (
                                        f"       - Authors: {authors}\n"
                                    )
                                    task_complete_details += (
                                        f"       - ArXiv ID: {arxiv_id}\n"
                                    )
                                    task_complete_details += (
                                        f"       - Published: {published}\n"
                                    )
                                    task_complete_details += (
                                        f"       - Categories: [{categories}]\n"
                                    )
                                    task_complete_details += f"       - URL: {url}\n"
                                    task_complete_details += (
                                        f"       - PDF URL: {pdf_url}\n"
                                    )
                                    task_complete_details += f"       - PDF Available: {'✅' if paper.get('pdf_available') else '❌'}\n"
                                    task_complete_details += f"       - Full Text Extracted: {'✅' if paper.get('full_text_extracted') else '❌'}\n"
                                    task_complete_details += f"       - Full Text Length: {paper.get('full_text_length', 0)} characters\n"

                                    if paper.get("abstract_snippet"):
                                        abstract = paper[
                                            "abstract_snippet"
                                        ]  # Complete abstract
                                        task_complete_details += (
                                            f"       - Abstract: {abstract}\n"
                                        )
                                    task_complete_details += f"\n"

                        elif subtask.get("type") == "source_analysis":
                            task_complete_details += f"   • Total Sources Available: {details.get('total_sources_available', 0)}\n"
                            task_complete_details += f"   • Web: {details.get('web_sources_count', 0)}, Academic: {details.get('academic_sources_count', 0)}, Cached: {details.get('cached_sources_count', 0)}\n"
                            task_complete_details += f"   • Sources to Process: {details.get('sources_to_process', 0)}\n"
                            task_complete_details += f"   • Cache Threshold: {details.get('cache_threshold', 'N/A')}\n"
                            task_complete_details += f"   • Unique Domains: {details.get('unique_domains', 0)}\n"
                            task_complete_details += f"   • Source Quality: {details.get('source_quality', 'Unknown')}\n"

                            # Show ALL detailed web sources
                            web_sources = details.get(
                                "detailed_web_sources", []
                            )  # Show ALL
                            if web_sources:
                                task_complete_details += f"   • ALL Web Sources Details ({len(web_sources)} total):\n"
                                for ws in web_sources:
                                    title = ws.get("title", "Unknown")  # Complete title
                                    task_complete_details += f"     • {title}\n"
                                    task_complete_details += f"       - Domain: {ws.get('domain', 'Unknown')}\n"
                                    task_complete_details += f"       - Source: {ws.get('source', 'Unknown')}\n"
                                    task_complete_details += (
                                        f"       - URL: {ws.get('url', 'No URL')}\n\n"
                                    )

                            # Show ALL detailed academic sources
                            academic_sources = details.get(
                                "detailed_academic_sources", []
                            )  # Show ALL
                            if academic_sources:
                                task_complete_details += f"   • ALL Academic Sources Details ({len(academic_sources)} total):\n"
                                for acs in academic_sources:
                                    title = acs.get(
                                        "title", "Unknown"
                                    )  # Complete title
                                    task_complete_details += f"     • {title}\n"
                                    task_complete_details += f"       - ArXiv ID: {acs.get('arxiv_id', 'Unknown')}\n"
                                    task_complete_details += f"       - Source: {acs.get('source', 'Unknown')}\n"
                                    task_complete_details += (
                                        f"       - URL: {acs.get('url', 'No URL')}\n\n"
                                    )

                            # Show ALL detailed cached sources
                            cached_sources = details.get(
                                "detailed_cached_sources", []
                            )  # Show ALL
                            if cached_sources:
                                task_complete_details += f"   • ALL Cached Sources Details ({len(cached_sources)} total, >70% similarity):\n"
                                for cs in cached_sources:
                                    title = cs.get("title", "Unknown")  # Complete title
                                    task_complete_details += f"     • {title}\n"
                                    task_complete_details += f"       - Similarity: {cs.get('similarity', 'Unknown')}\n"
                                    task_complete_details += f"       - Quality: {cs.get('quality', 'Unknown')}\n"
                                    task_complete_details += f"       - Source: {cs.get('source', 'Unknown')}\n\n"

                        elif subtask.get("type") == "cross_source_analysis":
                            task_complete_details += f"   • Sources analyzed: {details.get('sources_analyzed', 0)}\n"
                            task_complete_details += f"   • Analysis dimensions: {len(details.get('analysis_dimensions', []))}\n"

                        elif subtask.get("type") == "llm_extraction":
                            task_complete_details += f"   • Model Used: {details.get('model_used', 'Unknown')}\n"
                            task_complete_details += f"   • Content Length: {details.get('content_length', 0)} characters\n"
                            task_complete_details += f"   • Sources Processed: {details.get('sources_processed', 'Unknown')}\n"
                            task_complete_details += f"   • Processing Method: {details.get('processing_method', 'Unknown')}\n"
                            if details.get("content_preview"):
                                complete_content = details[
                                    "content_preview"
                                ]  # Complete content, no truncation
                                task_complete_details += f"   • Complete Extracted Content:\n{complete_content}\n"

                        elif subtask.get("type") == "statistics_extraction":
                            stats_count = details.get("statistics_found", 0)
                            task_complete_details += (
                                f"   • Statistics Found: {stats_count}\n"
                            )
                            if stats_count > 0:
                                actual_stats = details.get(
                                    "actual_statistics", []
                                )  # Show ALL statistics
                                task_complete_details += f"   • Extraction Method: {details.get('extraction_method', 'Unknown')}\n"
                                if actual_stats:
                                    task_complete_details += f"   • ALL Statistics Found ({len(actual_stats)} total):\n"
                                    for stat in actual_stats:
                                        task_complete_details += f"     • {stat}\n"
                            else:
                                task_complete_details += f"   • Note: {details.get('note', 'No additional info')}\n"

                        elif subtask.get("type") == "findings_extraction":
                            findings_count = details.get("findings_count", 0)
                            task_complete_details += (
                                f"   • Findings Found: {findings_count}\n"
                            )
                            if findings_count > 0:
                                actual_findings = details.get(
                                    "actual_findings", []
                                )  # Show ALL findings
                                task_complete_details += f"   • Extraction Method: {details.get('extraction_method', 'Unknown')}\n"
                                if actual_findings:
                                    task_complete_details += f"   • ALL Findings Found ({len(actual_findings)} total):\n"
                                    for finding in actual_findings:
                                        finding_text = (
                                            finding  # Complete finding, no truncation
                                        )
                                        task_complete_details += (
                                            f"     • {finding_text}\n"
                                        )
                            else:
                                task_complete_details += f"   • Note: {details.get('note', 'No additional info')}\n"

                        elif subtask.get("type") == "similarity_check":
                            task_complete_details += f"   • Similarity threshold: {details.get('threshold', 'N/A')}\n"
                            task_complete_details += f"   • Best match: {details.get('best_similarity', 'N/A')}%\n"

                        # Generic details for other subtask types
                        else:
                            key_details = {
                                k: v
                                for k, v in details.items()
                                if k not in ["timestamp", "content_preview"]
                                and not k.endswith("_count")
                                and not k.endswith("_used")
                            }
                            for key, value in list(key_details.items())[
                                :2
                            ]:  # Show top 2 details
                                if (
                                    isinstance(value, (int, float, str))
                                    and str(value).strip()
                                ):
                                    task_complete_details += f"   • {key.replace('_', ' ').title()}: {value}\n"

            task_complete_details += f"\n"
            detailed_reasoning.append(task_complete_details)

            step_desc = f"Completed {task.task_type.value}: {task.description}"
            session.reasoning_steps.append(step_desc)

        # Complete session
        session.completed_at = datetime.now()

        completion_details = f"\n🎯 **RESEARCH SESSION COMPLETED**\n\n✅ All {len(tasks)} tasks executed successfully\n▶ Total duration: {(session.completed_at - session.created_at).total_seconds():.1f} seconds\n▶ Session ID: {session_id}\n▶ Final answer generated and ready\n\n"
        detailed_reasoning.append(completion_details)

        yield "✨ **Finalizing Results...**\n\nProcessing and formatting the research findings...", "\n".join(
            detailed_reasoning
        )

        # Format the final response
        response_parts = []

        # Main answer
        if session.final_answer:
            response_parts.append("## 🔬 Research Results\n")
            response_parts.append(session.final_answer)

        # Task summary
        task_summaries = []
        for task in tasks:
            status_emoji = "✅" if task.status.value == "completed" else "❌"
            task_summaries.append(
                f"{status_emoji} **{task.task_type.value.title()}**: {task.description}"
            )

        if task_summaries:
            response_parts.append("\n## 📋 Task Summary\n")
            response_parts.extend(task_summaries)

        # Sources / Citations
        if session.sources:
            response_parts.append("\n## 📚 Sources\n")
            for src in session.sources:
                title = src.get("title", "Untitled Source")
                url = src.get("url") or src.get("pdf_url") or src.get("source") or ""
                src_type = src.get("type", "source")
                if url:
                    response_parts.append(f"- [{title}]({url}) - {src_type}")
                else:
                    response_parts.append(f"- {title} - {src_type}")

        # Metadata
        duration = (session.completed_at - session.created_at).total_seconds()
        response_parts.append(f"\n## 📊 Research Metrics\n")
        response_parts.append(f"- **Total Tasks**: {len(tasks)}")
        response_parts.append(
            f"- **Completed**: {len([t for t in tasks if t.status.value == 'completed'])}"
        )
        response_parts.append(
            f"- **Failed**: {len([t for t in tasks if t.status.value == 'failed'])}"
        )
        response_parts.append(f"- **Duration**: {duration:.1f} seconds")

        final_response = "\n".join(response_parts)

        # Cache result in vector DB
        try:
            vector_db.add_texts([question], metadatas=[{"answer": final_response}])
            caching_details = f"\n💾 **RESULT CACHING**\n\n✅ Research results cached for future queries\n▶ Vector embedding stored\n▶ Answer saved to knowledge base\n▶ Ready for instant retrieval\n\n"
            detailed_reasoning.append(caching_details)
        except Exception as e:
            caching_details = (
                f"\n💾 **RESULT CACHING**\n\n❌ Failed to cache results: {str(e)}\n\n"
            )
            detailed_reasoning.append(caching_details)

        yield final_response, "\n".join(detailed_reasoning)

    except Exception as e:
        error_msg = f"""❌ **Error**: Unable to process your research question at the moment.

**Technical Details**: {str(e)}

**Troubleshooting**:
- Make sure you have set up your OpenAI API key in the .env file
- Check your internet connection
- Verify that the MCP simulator is properly configured"""

        error_details = f"❌ **ERROR OCCURRED**\n\n▶ Error Type: {type(e).__name__}\n▶ Error Message: {str(e)}\n▶ Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
        yield error_msg, error_details


def research_assistant(question: str, show_reasoning: bool = True) -> str:
    """
    Enhanced research assistant using MCP workflow (non-streaming version).
    """
    # Get the final result from the streaming version
    final_result = ""
    for result, _ in research_assistant_streaming(question, show_reasoning):
        final_result = result
    return final_result


def advanced_research_assistant(
    question: str, search_depth: str = "advanced", include_academic: bool = True
) -> str:
    """
    Advanced research assistant with more detailed workflow and tool integration.
    """
    if not question.strip():
        return "Please ask me a research question!"

    try:
        # Initialize components
        (
            model,
            mcp_simulator,
            web_search_tool,
            document_processor,
            vector_db,
        ) = create_research_assistant()

        # Check vector DB cache first
        cached = vector_db.similarity_search(question, k=1)
        if (
            cached
            and (1 - cached[0]["distance"]) >= 0.9
            and cached[0]["metadata"].get("answer")
        ):
            return cached[0]["metadata"]["answer"]

        # Step 1: Web Search
        search_results = web_search_tool.search_research_topic(
            question, include_academic=include_academic
        )

        # Step 2: Process search results
        processed_documents = []
        for result in search_results[:5]:  # Process top 5 results
            doc_analysis = document_processor.process_text(
                result.content, f"search_result_{result.title[:30]}"
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

        # Cache result in vector DB
        try:
            vector_db.add_texts(
                [question], metadatas=[{"answer": "\n".join(response_parts)}]
            )
        except Exception:
            pass
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
        gr.Markdown(
            "Ask me complex research questions and I'll help you find comprehensive answers using AI-powered MCP workflow with **real-time progress updates**!"
        )

        # Main Research Interface
        gr.Markdown("### ⚡ Live Research with Real-time Updates")

        question_input = gr.Textbox(
            label="Your Research Question",
            placeholder="e.g., What are the latest developments in quantum computing?",
            lines=3,
        )

        show_reasoning = gr.Checkbox(
            label="Show detailed reasoning steps",
            value=True,
            info="Display the step-by-step research process in real-time",
        )

        submit_btn = gr.Button("🚀 Start Live Research", variant="primary", size="lg")

        # Main output area
        output = gr.Markdown(label="Research Progress & Results")

        # Collapsible reasoning section
        with gr.Accordion(
            "🧠 Detailed Reasoning Steps", open=False
        ) as reasoning_accordion:
            reasoning_output = gr.Markdown(label="Step-by-step reasoning and outputs")

        def stream_research(question, show_reasoning):
            """Handle streaming research with live updates."""
            if not question.strip():
                yield "Please ask me a research question!", ""
                return

            for status_update, reasoning_update in research_assistant_streaming(
                question, show_reasoning
            ):
                yield status_update, reasoning_update

        submit_btn.click(
            fn=stream_research,
            inputs=[question_input, show_reasoning],
            outputs=[output, reasoning_output],
        )

        # Add example questions
        gr.Markdown("## 📝 Example Questions")
        gr.Markdown("Click any example below to try it out:")
        gr.Examples(
            examples=[
                ["What are the latest developments in quantum computing?"],
                ["How do renewable energy costs compare globally in 2024?"],
                [
                    "What are the emerging trends in artificial intelligence for healthcare?"
                ],
                ["What are the key factors in successful startup fundraising?"],
                [
                    "What are the most effective global health programs for malaria in sub-Saharan Africa?"
                ],
                ["How does climate change affect agricultural productivity?"],
                ["How effective are different COVID-19 vaccination strategies?"],
                ["What are the economic impacts of remote work adoption?"],
            ],
            inputs=[question_input],
        )

        # Add information about the system
        gr.Markdown(
            """
        ## 🛠️ System Features
        
        ### ⚡ Live Research with Real-time Updates
        - **Real-time progress updates**: See what the AI is doing as it happens
        - **Streaming interface**: No more waiting for static results
        - **Collapsible reasoning**: View detailed step-by-step thought process and outputs
        - **Live status indicators**: Know exactly which task is currently running
        - **Multi-source synthesis**: Combines web search and academic paper discovery
        - **Intelligent caching**: Faster responses for similar questions
        
        ### 🔧 Core Technologies
        - **MCP (Model Context Protocol) Simulator**: Orchestrates multi-step research workflows
        - **Web Search Tool**: Finds relevant information from the internet using Tavily
        - **ArXiv Search**: Academic paper discovery and analysis
        - **Vector Database**: Caches previous research for faster responses
        - **LLM Integration**: Uses OpenAI models for reasoning and synthesis
        - **Task Planning**: AI-powered breakdown of complex research questions
        """
        )

    return demo


def main():
    """Main entry point for the research assistant application."""
    print("🚀 Starting Research Assistant...")
    print("🔧 Using MCP Simulator for research orchestration...")
    print("🌐 Web search tool operational...")

    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
