# Research Assistant - MCP Style Orchestration

A sophisticated research assistant that answers complex questions using multi-agent orchestration inspired by Anthropic's Model Context Protocol (MCP). Built for a Copoly.ai take-home assignment.

## 🎯 Overview

This research assistant demonstrates advanced AI orchestration by:
- **Breaking down complex questions** into manageable research tasks
- **Coordinating multiple tools** (web search, document processing, LLM reasoning)
- **Providing traceable reasoning** with step-by-step research processes
- **Generating comprehensive, well-structured responses** with source citations

## 🏗️ Architecture

### Core Components

1. **MCP Simulator** (`src/orchestration/mcp_simulator.py`)
   - Orchestrates research workflows using task decomposition
   - Manages research sessions and task execution
   - Provides reasoning traceability and progress tracking

2. **Model Builder** (`src/models/model_builder.py`)
   - Flexible LLM integration with multiple providers (OpenAI, Ollama)
   - Model-Agnostic Builder pattern for easy model configuration
   - Support for different model types and parameters

3. **Web Search Tool** (`src/tools/web_search.py`)
   - Tavily API integration for real-time web search
   - Academic source filtering and research-focused queries
   - Fallback simulation when API keys aren't available

4. **Document Processor** (`src/tools/document_processor.py`)
   - Text analysis and content extraction
   - Key insights identification and summarization
   - Cross-document analysis and contradiction detection

### Research Workflow

```
User Question → Task Planning → Information Retrieval → Content Extraction → Synthesis → Report Generation
```

1. **Task Planning**: LLM breaks down the question into specific research tasks
2. **Information Retrieval**: Web search tool finds relevant sources
3. **Content Extraction**: Document processor analyzes and extracts key insights
4. **Synthesis**: LLM combines information from multiple sources
5. **Report Generation**: Final structured response with reasoning and citations

### Stack
- **Orchestration**: LangGraph for state-based workflow management
- **Tools**: LangChain for individual research components
- **LLMs**: Model-agnostic design (OpenAI, Anthropic, etc.)
- **Vector Store**: ChromaDB for knowledge base and query caching
- **Frontend**: Gradio for rapid prototyping and demos
- **Deployment**: Local scripts + Render cloud hosting

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM responses)
- Tavily API key (optional, for web search)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url> */CHANGE
   cd research-assistant-mcp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # OPENAI_API_KEY=your_key_here
   # TAVILY_API_KEY=your_tavily_key
   ```

5. **Run the application**
   ```bash
   gradio src/main.py
   ```

6. **Access the interface**
   - Open your browser to `http://localhost:7860`
   - Start asking research questions!

## 📋 Features

### Basic Research
- **MCP Workflow**: Multi-step research orchestration
- **Task Decomposition**: Automatic breakdown of complex questions
- **Reasoning Traceability**: Step-by-step research process
- **Structured Output**: Well-formatted research reports

### Advanced Research
- **Web Search Integration**: Real-time information retrieval
- **Academic Source Filtering**: Focus on research papers and credible sources
- **Cross-Document Analysis**: Identify themes, contradictions, and gaps
- **Confidence Scoring**: Quality assessment of research results

### User Interface
- **Dual-Mode Interface**: Basic and Advanced research tabs
- **Example Questions**: Pre-loaded research examples
- **Progress Tracking**: Real-time task execution status
- **Export Options**: Structured research results

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for LLM responses
OPENAI_API_KEY=your_openai_api_key_here

# Optional for enhanced web search
TAVILY_API_KEY=your_tavily_api_key_here

# Optional for local models
OLLAMA_BASE_URL=http://localhost:11434
```

### Model Configuration

The system supports multiple LLM providers:

```python
# OpenAI (default)
model = ModelBuilder().with_provider("openai").with_model("gpt-4o-mini").build()

# Ollama (local)
model = ModelBuilder().with_provider("ollama").with_model("llama2").build()
```

## 📊 Example Usage

### Basic Research Question
```
"What are the most effective global health programs for malaria in sub-Saharan Africa?"
```

**System Response:**
- Task breakdown into search, extraction, and synthesis phases
- Step-by-step reasoning process
- Comprehensive answer with key findings
- Research gaps and recommendations

### Advanced Research with Web Search
```
"How do renewable energy costs compare globally in 2024?"
```

**Enhanced Response:**
- Real-time web search results
- Academic source analysis
- Statistical data extraction
- Cross-source synthesis
- Confidence scoring

## 🔧 Development

### Project Structure
```
research-assistant-mcp/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── models/
│   │   └── model_builder.py    # LLM integration and configuration
│   ├── orchestration/
│   │   └── mcp_simulator.py    # MCP workflow orchestration
│   └── tools/
│       ├── web_search.py       # Web search capabilities
│       └── document_processor.py # Document analysis tools
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks for experimentation
├── deployment/                 # Deployment configurations
└── requirements.txt            # Python dependencies
```

### Adding New Tools

1. **Create tool module** in `src/tools/`
2. **Implement tool interface** with standard methods
3. **Integrate with MCP simulator** in task execution
4. **Update main application** to use new tool

Example tool structure:
```python
class NewTool:
    def __init__(self, config):
        self.config = config
    
    def execute(self, task_data):
        # Tool implementation
        return result
```

### Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_mcp_simulator.py
```

## 🎓 Learning Objectives

This project demonstrates:

1. **MCP Concepts**: Model Context Protocol workflow simulation
2. **Agent Orchestration**: Multi-step task coordination
3. **Tool Integration**: Seamless combination of different AI tools
4. **LLM Prompting**: Advanced prompt engineering for research tasks
5. **Error Handling**: Robust error management and fallbacks
6. **User Experience**: Intuitive interface for complex AI workflows

## 🔮 Future Enhancements

- **Vector Database Integration**: ChromaDB for document storage and retrieval
- **Multi-Modal Support**: Image and document upload capabilities
- **Collaborative Research**: Multi-user research sessions
- **Research Templates**: Pre-defined research methodologies
- **Export Formats**: PDF, Word, and LaTeX report generation
- **API Endpoints**: RESTful API for programmatic access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Anthropic**: For MCP protocol inspiration
- **OpenAI**: For LLM capabilities
- **Tavily**: For web search integration
- **Gradio**: For the user interface framework

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Check the documentation in the `notebooks/` directory
- Review the example usage in the main interface

---

**Built with ❤️ for AI research and education**