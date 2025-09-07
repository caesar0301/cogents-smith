# Cogents-Tools

[![CI](https://github.com/mirasurf/cogents-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/mirasurf/cogents-tools/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cogents-tools.svg)](https://pypi.org/project/cogents-tools/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mirasurf/cogents-tools)

A comprehensive collection of essential building blocks for constructing cognitive multi-agent systems (MAS). Rather than building a full agent framework, Cogents provides a lightweight repository of key components designed to bridge the final mile in MAS development. Our philosophy focuses on modular, composable components that can be easily integrated into existing systems or used to build new ones from the ground up.

For the underlying philosophy, refer to my talk on MAS ([link](https://github.com/caesar0301/mas-talk-2508/blob/master/mas-talk-xmingc.pdf)).


## 🛠️ Current Project Status

Cogents-tools has evolved into a mature, production-ready toolkit ecosystem featuring **advanced lazy loading** and **semantic organization**. The project now offers 17+ specialized toolkits organized into 10 semantic groups, providing comprehensive coverage for cognitive agent development.

### 🚀 Latest Features (v0.1.0+)
- **⚡ Lazy Loading System**: 3-5x faster imports with on-demand module loading
- **📦 Semantic Groups**: Organized toolkit collections for specific domains
- **🔧 Zero Configuration**: Works out-of-the-box with intelligent defaults
- **📊 Performance Monitoring**: Built-in tracking of module usage and loading

### 🎯 Core Capabilities

#### Extensible Resources & Infrastructure
- **Web Search**: Multi-provider integration (Tavily, Google AI Search, Serper)
- **Vector Stores**: Production-ready backends (Weaviate, PgVector) with semantic search
- **Document Processing**: Intelligent text extraction and chunking for RAG workflows
- **Voice Processing**: Advanced transcription and audio analysis capabilities

#### Toolkit Ecosystem (17+ Tools)
- **Academic Research**: arXiv integration for paper discovery and analysis
- **Development Tools**: Bash execution, file editing, GitHub integration, Python execution
- **Media Processing**: Image analysis, video processing, audio transcription
- **Information Retrieval**: Wikipedia, web search, and knowledge extraction
- **Data Management**: Tabular data processing, memory systems, document handling
- **Human Interaction**: User communication and feedback collection systems

#### Architecture & Performance
- **Lazy Loading**: Only load what you need, when you need it
- **Semantic Organization**: Intuitive grouping reduces cognitive overhead
- **Async-First Design**: Built for high-performance concurrent operations
- **Extensible Registry**: Easy integration of custom tools and capabilities
- **Error Resilience**: Graceful handling of missing dependencies and failures

## ⚡ Lazy Loading & Performance

Cogents-tools features an advanced **lazy loading system** that dramatically improves import performance and reduces memory usage:

- **🚀 Fast imports**: Only load what you need, when you need it
- **📦 Group-wise loading**: Import semantic groups of related toolkits
- **💾 Memory efficient**: Unused toolkits remain unloaded
- **🔧 Zero configuration**: Lazy loading is enabled by default

### Quick Start with Groups

```python
# Import by semantic groups - fast and organized
from cogents_tools import groups

# Development tools
dev_tools = groups.development()
bash = dev_tools.bash_toolkit()
file_editor = dev_tools.file_edit_toolkit()

# Academic research
academic = groups.academic()
arxiv = academic.arxiv_toolkit()

# Information retrieval
search_tools = groups.info_retrieval()
wiki = search_tools.wikipedia_toolkit()
```

### Available Toolkit Groups

| Group | Description | Toolkits |
|-------|-------------|----------|
| `academic` | Academic research tools | arxiv_toolkit |
| `audio` | Audio processing | audio_toolkit, audio_aliyun_toolkit |
| `communication` | Communication & messaging | memory_toolkit |
| `development` | Development tools | bash_toolkit, file_edit_toolkit, github_toolkit, python_executor_toolkit, tabular_data_toolkit |
| `file_processing` | File manipulation | document_toolkit, file_edit_toolkit, tabular_data_toolkit |
| `hitl` | Human-in-the-loop | user_interaction_toolkit |
| `image` | Image processing | image_toolkit |
| `info_retrieval` | Information search | search_toolkit, serper_toolkit, wikipedia_toolkit |
| `persistence` | Data storage | memory_toolkit |
| `video` | Video processing | video_toolkit |

## Install

```bash
pip install -U cogents-tools
```

## 🚀 Quick Examples

### Lazy Loading Management

```python
import cogents_tools

# Check lazy loading status
print(f"Lazy loading enabled: {cogents_tools.is_lazy_loading_enabled()}")

# Get available groups
print(f"Available groups: {cogents_tools.get_available_groups()}")

# Load specific group
dev_toolkits = cogents_tools.load_toolkit_group('development')

# Monitor loaded modules
print(f"Loaded modules: {cogents_tools.get_loaded_modules()}")
```

### Group-wise Usage Examples

```python
from cogents_tools import groups

# Academic research workflow
academic = groups.academic()
arxiv = academic.arxiv_toolkit()
papers = await arxiv.search_papers("machine learning", max_results=5)

# Development workflow  
dev = groups.development()
bash = dev.bash_toolkit()
result = await bash.execute_command("ls -la")

# File processing workflow
files = groups.file_processing()
doc_processor = files.document_toolkit()
content = await doc_processor.extract_text("document.pdf")

# Information retrieval workflow
search = groups.info_retrieval()
wiki = search.wikipedia_toolkit()
info = await wiki.search("artificial intelligence")
```

## 📚 Demo Scripts

Explore the capabilities with our comprehensive demo scripts:

### Lazy Loading Demo
```bash
python examples/lazy_importing.py
```
Demonstrates:
- Group-wise imports and performance benefits
- On-demand toolkit loading
- Memory usage optimization
- Error handling and fallbacks

### Vector Store Demo  
```bash
python examples/vector_store_demo.py
```
Showcases:
- Multiple vector store backends (Weaviate, PgVector)
- Document processing and embedding
- Semantic search capabilities
- Best practices and configuration

### Web Search Demo
```bash
python examples/web_search_demo.py  
```
Features:
- Multiple search providers (Tavily, Google AI Search)
- Advanced search strategies and filtering
- Result processing and analysis
- Integration patterns with other tools

## 🔧 Advanced Lazy Loading

### Manual Control

```python
import cogents_tools

# Disable lazy loading for immediate imports
cogents_tools.disable_lazy_loading()

# Re-enable when needed
cogents_tools.enable_lazy_loading()

# Load all toolkits at once (not recommended for production)
all_toolkits = cogents_tools.load_all_toolkits()
```

### Group Information

```python
from cogents_tools.groups import list_groups, get_group_info

# List all available groups with descriptions
groups_info = list_groups()
for name, description in groups_info.items():
    print(f"{name}: {description}")

# Get detailed info about a specific group
dev_info = get_group_info('development')
print(f"Toolkits in development: {dev_info['toolkits']}")
```

### Performance Benefits

The lazy loading system provides significant performance improvements:

- **⚡ 3-5x faster imports**: Only essential modules loaded initially
- **💾 Reduced memory footprint**: Unused toolkits don't consume memory  
- **🔄 On-demand loading**: Toolkits loaded only when accessed
- **📊 Load monitoring**: Track which modules are actually used
- **🎯 Semantic grouping**: Logical organization reduces cognitive load

### Best Practices

1. **Use group imports** for related functionality
2. **Monitor loaded modules** in production to optimize usage
3. **Keep lazy loading enabled** unless you have specific requirements
4. **Import at the group level** rather than individual toolkits
5. **Use the demos** to understand performance characteristics

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgment

- Tencent [Youtu-agent](https://github.com/Tencent/Youtu-agent) toolkits integration.
