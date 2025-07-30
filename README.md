# Research Assistant - MCP Style Orchestration

A research assistant that answers complex questions using multi-step task orchestration inspired by Anthropic's Model Context Protocol (MCP). Built for the Copoly.ai take-home assignment.

## 🎯 Overview

This assistant demonstrates AI workflow orchestration by:
- **Breaking down complex questions** into manageable research tasks (SEARCH → EXTRACT → SYNTHESIZE → REPORT)
- **Coordinating multiple tools** (web search, ArXiv papers, vector database, table analysis)
- **Providing traceable reasoning** with step-by-step research processes
- **Generating structured responses** with proper source citations

## 🏗️ Architecture

The system uses a custom **MCP Simulator** (`src/orchestration/mcp_simulator.py`) that orchestrates research workflows by decomposing questions into typed tasks (SEARCH, EXTRACT, SYNTHESIZE, REPORT), then executing each task with appropriate tools.

**Key Components:**
- **Search Coordinator** - Combines web search (Tavily) + academic papers (ArXiv) with intelligent planning
- **Vector Database** - ChromaDB for caching previous Q&A and document retrieval  
- **Document Processor** - Extract insights, detect contradictions, analyze themes
- **Model Builder** - Provider-agnostic LLM integration (OpenAI, Ollama)

**Tech Stack:**
- Python 3.8+ with custom orchestration (no LangChain agents used)
- OpenAI GPT-4o-mini for reasoning tasks
- ChromaDB for vector storage and similarity search
- Gradio for web interface with streaming progress updates

## ✨ Features

- **Task Decomposition**: LLM automatically breaks questions into SEARCH → EXTRACT → SYNTHESIZE → REPORT steps
- **Multi-Source Research**: Web search (Tavily) + academic papers (ArXiv) + vector cache
- **Source Citations**: Proper attribution with URLs and document types  
- **Streaming Progress**: Real-time task execution updates in Gradio interface
- **Vector Caching**: ChromaDB stores previous Q&A for faster similar queries
- **Fallback Handling**: Works with limited API keys (search simulation modes)

## Design Decisions
- **Builder Pattern for LLM Model Class**: Very good at extendability, easy to understand, creates objects that are easy to interact with. Also, modular and very favourable to provider-agnostic structure
- **Provider**: Chose OpenAI as main provider due to cost (using 4.1-nano for most of development) and quality of embeddings. Additional bonuss is the presence of tools
- **Bespoke Orchestration**: Chosen over Langchain and Langgraph due to greater familiarity and easy of understanding. Not necessarily better or worse
- **Chroma for VectorDB**: Simple, easy to use, light and free
- **Tavily**: Easy solution for web search, POC-friendly free tier
- **Interface**: Gradio comes mostly ready out-of-the-box and looks good enough. Optimized for LLMs and chat from the get-go
- **Tools for Demo**: Web search, more specific academic search, csv analysis (very simple) and vector search are very common in production
- **Strong error Handling**: This was a big focus in order to develop as robust an application as possible in a short time. Errors and component failures should not be impediments to the core functionalities

### Next Steps
- **Better tabular analysis tools**: Implementation is a very simple version of what it could be. The possibilities are much wider, especially if we can download tabular data from the internet or if we have a more narrow objective
- **Langchain Integration**: Add this option for a v2 ofthe project. Structure of the project should be quite pick-and-mix.
- **Aesthetics and presentation**: Project would be better served by a react frontend or better configuration of gradio.
- **Web Hosting**: Would allow project to be run anywhere permanently

## 📊 Example

**Question:** *"What are the most effective malaria prevention programs in sub-Saharan Africa?"*

**System Response:**
1. **Task Planning** → Breaks into search, extract, synthesize, report tasks
2. **Web Search** → Finds current WHO reports, research papers, health data  
3. **ArXiv Search** → Retrieves recent academic studies on malaria interventions
4. **Content Extraction** → Identifies key programs: ITNs, IRS, seasonal chemoprevention
5. **Synthesis** → Compares effectiveness, costs, implementation challenges
6. **Final Report** → Structured answer with source citations and reasoning steps

## 🧪 Development

**Project Structure:**
```
data/                          # Data storage for vectors and csvs
notebooks/                     # Contains demo notebook
src/                           # Main project folder
├── main.py                    # Gradio interface
├── orchestration/
│   ├── mcp_simulator.py       # Task orchestration
│   └── search_coordinator.py  # Multi-source search planning
├── tools/                     # Individual research tools
└── models/                    # LLM provider abstraction
└── utils/                     # General Utilities
tests/                     # Tests for key components of code   
```

**Tests:** `pytest tests/` (includes MCP simulator unit tests)

**Notebooks:** See `demo.ipynb` for a clearer demo

---

## 🚀 Quick Start

**Prerequisites:** Python 3.8+, OpenAI API key

```bash
# Install and run
pip install -e .
echo "OPENAI_API_KEY=your_key_here" > .env
python src/main.py
# Open http://localhost:7860
```

**Optional:** Add `TAVILY_API_KEY=your_key` to `.env` for enhanced web search


**Assignment Requirements Met:**
✅ Multi-step task decomposition  
✅ Tool orchestration & coordination  
✅ Document/web source retrieval  
✅ Structured synthesis with citations  
✅ Traceable reasoning steps  
✅ Well-documented codebase & architecture