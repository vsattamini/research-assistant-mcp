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
- OpenAI GPT-4.1-nano for reasoning tasks
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
- **Bespoke Orchestration**: Chosen over Langchain and Langgraph due to greater control, predictability of behaviour, transparency in how the process is orchestrated and to apply the concepts as a challenge
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

**Present in example.md**

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

**Prerequisites:** Python 3.8+, OpenAI API key, Tavily API key (ideally)

```bash
# Install and run
pip install -e .
echo "OPENAI_API_KEY=your_key_here" > .env
echo "TAVILY_API_KEY=your_key_here" > .env
python src/main.py
# Open http://localhost:7860
```

**Optional:** Add `TAVILY_API_KEY=your_key` to `.env` for enhanced web search

## 🤖 LangChain Integration (Optional Demo)

For demonstration purposes, we've also implemented a **LangChain agent** that uses the same underlying research tools. This shows how both custom orchestration and standard LangChain patterns can work with identical infrastructure.

### Run the Comparison Demo
```bash
# Run side-by-side comparison of both approaches
python run_langchain_demo.py
# Open http://localhost:7861
```

## 🎁 BONUS: LangChain Integration & Comparison Demo

As a **bonus demonstration**, we've implemented a parallel **LangChain agent** that uses the exact same research tools to show how multiple orchestration approaches can work with identical infrastructure.

### 🎯 What This Demonstrates
- **Architectural Flexibility**: Same tools work with different orchestration patterns
- **LangChain Compatibility**: Standard agent framework using our custom tools
- **Approach Comparison**: Side-by-side results from both methodologies

### Limitations
- **No detailed step-by-step reasoning**
- **No detailed source breakdown**

### Benefits
- **Ease of implementation from base project**
- **Faster, optimized result (3x faster)**

### 🚀 Try the Comparison Demo

**From the main project folder:**

```bash
# Install optional LangChain dependencies
pip install langchain langchain-openai

# Run the side-by-side comparison interface
python run_langchain_demo.py
# Open http://localhost:7861
```

### 🆚 What You'll See
- **Custom MCP Simulator**: Predictable task sequencing with full transparency
- **LangChain Agent**: Autonomous tool selection using OpenAI function calling
- **Same Question, Both Approaches**: Compare results, reasoning, and execution
- **Performance Metrics**: Duration, tools used, and approach details

### 🔧 Implementation Details
The LangChain integration consists of:
- **Tool Wrappers** (`src/orchestration/langchain_tools.py`): LangChain-compatible versions of our research tools
- **Agent Implementation** (`src/orchestration/langchain_agent.py`): Simple OpenAI Tools agent
- **Comparison Interface** (`src/langchain_demo.py`): Side-by-side demo with both approaches
- **Zero Impact**: Original app functionality completely unchanged

This demonstrates that our custom MCP approach is a **deliberate architectural choice** rather than a limitation, showing we understand both custom orchestration and standard framework integration.

**LangChain Dependencies** (optional):
```bash
pip install langchain langchain-openai
```

### Key Differences

| Approach | Custom MCP Simulator | LangChain Agent |
|----------|---------------------|-----------------|
| **Control** | Explicit task sequencing | Autonomous tool selection |
| **Transparency** | Full visibility into each step | Agent decides internally |
| **Optimization** | Research-workflow specific | General-purpose framework |
| **Predictability** | Deterministic task flow | LLM-driven decisions |

Both approaches demonstrate the same core capability: multi-tool research orchestration with comprehensive source integration.