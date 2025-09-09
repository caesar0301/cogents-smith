# Cogents-Tools

[![CI](https://github.com/mirasurf/cogents-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/mirasurf/cogents-tools/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cogents-tools.svg)](https://pypi.org/project/cogents-tools/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mirasurf/cogents-tools)

This is part of [project Cogents](https://github.com/mirasurf/cogents), an initiative to develop a cognitive, computation-driven agentic system. This repo is built upon [cogents-core](https://github.com/mirasurf/cogents-core) and hosts an extensive and extendable list of integrated services, well-tested toolkits, and ready-to-go agents. Our philosophy focuses on a modular, composable design that can be easily integrated into existing systems or used to build new ones from the ground up.

## 🎯 Core Capabilities

Cogents-tools has evolved into a mature, production-ready toolkit ecosystem featuring **advanced lazy loading** and **semantic organization**. The project now offers 17+ specialized toolkits organized into 10 semantic groups, providing comprehensive coverage for cognitive agent development.

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

### Lazy Loading

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

### Agent-Tools Integration

TODO

## 📚 Demo Scripts

Explore the capabilities with our comprehensive demo scripts under [examples](./examples) folder.

## Best Practices

1. **Use group imports** for related functionality
2. **Keep lazy loading enabled** unless you have specific requirements
3. **Use the demos** to understand performance characteristics

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgment

- Tencent [Youtu-agent](https://github.com/Tencent/Youtu-agent) toolkits integration.
