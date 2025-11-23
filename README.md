# CogentsSmith

[![CI](https://github.com/caesar0301/cogents-smith/actions/workflows/ci.yml/badge.svg)](https://github.com/caesar0301/cogents-smith/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cogents-smith.svg)](https://pypi.org/project/cogents-smith/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/caesar0301/cogents-smith)

This is part of [project COGENTS](https://github.com/caesar0301/cogents), an initiative to develop a cognitive, computation-driven agentic system. This repo is built upon [cogents-core](https://github.com/caesar0301/cogents-core) and hosts an extensive and extendable list of integrated services, well-tested toolkits, and ready-to-go agents. Our philosophy focuses on a modular, composable design that can be easily integrated into existing systems or used to build new ones from the ground up.

## 🎯 Core Capabilities

Cogents-smith has evolved into a mature, production-ready toolkit ecosystem featuring **semantic organization**. The project now offers **18 specialized toolkits** organized into **10 semantic groups**, providing comprehensive coverage for cognitive agent development, plus **1 production-ready agent** for specialized tasks.

#### Toolkit Ecosystem (18 Tools)
- **Academic Research**: arXiv integration for paper discovery and analysis
- **Development Tools**: Bash execution, file editing, GitHub integration, Python execution
- **Media Processing**: Image analysis, video processing, audio transcription
- **Information Retrieval**: Wikipedia, web search, and knowledge extraction
- **Data Management**: Tabular data processing, memory systems, document handling
- **Communication**: Gmail integration for email management
- **Human Interaction**: User communication and feedback collection systems

#### Ready-to-Use Agents
- **Askura Agent**: Dynamic conversational agent for collecting structured information through natural dialogue
- **Seekra Agent**: Deep research agent for comprehensive topic investigation and report generation

#### Architecture & Performance
- **Lazy Loading**: Only load what you need, when you need it
- **Semantic Organization**: Intuitive grouping reduces cognitive overhead
- **Async-First Design**: Built for high-performance concurrent operations
- **Extensible Registry**: Easy integration of custom tools and capabilities
- **Error Resilience**: Graceful handling of missing dependencies and failures

## 📦 Semantic Organization

Cogents-smith features **semantic organization** that makes it easy to find and use related toolkits:

- **🎯 Organized structure**: Toolkits grouped by functionality
- **📦 Group-wise loading**: Import semantic groups of related toolkits
- **🔧 Easy discovery**: Simple group-based API

### Available Toolkit Groups

| Group | Description | Toolkits |
|-------|-------------|----------|
| `academic` | Academic research tools | arxiv_toolkit |
| `audio` | Audio processing | audio_toolkit, audio_aliyun_toolkit |
| `communication` | Communication & messaging | gmail_toolkit |
| `development` | Development tools | bash_toolkit, file_edit_toolkit, github_toolkit, python_executor_toolkit, tabular_data_toolkit |
| `file_processing` | File manipulation | document_toolkit, file_edit_toolkit, tabular_data_toolkit |
| `hitl` | Human-in-the-loop | user_interaction_toolkit |
| `image` | Image processing | image_toolkit |
| `info_retrieval` | Information search | search_toolkit, serper_toolkit, wikipedia_toolkit |
| `memorization` | Data storage & memory | memory_toolkit |
| `video` | Video processing | video_toolkit |

## Install

```bash
pip install -U cogents-smith
```

## 🚀 Quick Examples

### Group Loading

```python
import cogents_smith

# Get available groups
print(f"Available groups: {cogents_smith.get_available_groups()}")

# Load specific group
dev_toolkits = cogents_smith.load_toolkit_group('development')

# Or use semantic group imports
from cogents_smith.groups import development, info_retrieval

# Access toolkits from groups
bash = development().bash_toolkit()
search = info_retrieval().search_toolkit()
```

### Using Askura Agent (Conversational Data Collection)

```python
from cogents_smith.agents.askura_agent import AskuraAgent
from cogents_smith.agents.askura_agent.models import AskuraConfig, InformationSlot

# Define what information you want to collect
config = AskuraConfig(
    information_slots=[
        InformationSlot(
            name="trip_info",
            description="Travel plan details: destination, dates, interests",
            priority=5,
            required=True
        )
    ],
    conversation_purpose=["collect user information about planned trip"]
)

# Start conversation
agent = AskuraAgent(config=config)
response = agent.start_conversation(
    user_id="user123",
    initial_message="I want to plan a trip"
)
```

### Using Seekra Agent (Deep Research)

```python
from cogents_smith.agents.seekra_agent import SeekraAgent, Configuration

# Initialize research agent
researcher = SeekraAgent(
    configuration=Configuration(
        search_engine="tavily",
        number_of_initial_queries=2,
        max_research_loops=2
    )
)

# Conduct research
result = researcher.research(
    user_message="Deep learning trends in 2025"
)

print(result.summary)
print(f"Sources: {len(result.sources)}")
```

## 📚 Demo Scripts

Explore the capabilities with our comprehensive demo scripts:
- **Agents**: [examples/agents/](./examples/agents/) - Interactive demos for Askura and Seekra agents
- **Tools**: [examples/tools/](./examples/tools/) - Toolkit usage examples

## 🤖 Agents

### Askura Agent
A dynamic conversational agent designed for structured information collection through natural dialogue. Askura adapts to user communication styles and maintains conversation purpose alignment.

**Key Features:**
- Structured information slot collection
- Adaptive conversation flow
- Memory and context management
- Reflection and summarization capabilities
- Token usage tracking

**Use Cases:** User interviews, data collection, form filling, customer onboarding

### Seekra Agent
A deep research agent that conducts comprehensive investigations on given topics, generating detailed reports with source citations.

**Key Features:**
- Multi-source web research
- Iterative research loops for depth
- Automatic query generation
- Source citation and aggregation
- Configurable search engines (Tavily, etc.)

**Use Cases:** Market research, academic literature reviews, competitive analysis, knowledge synthesis

## Best Practices

1. **Use group imports** for related functionality to keep dependencies organized
2. **Use semantic groups** to discover and access toolkits intuitively
3. **Leverage async capabilities** for better performance in concurrent operations
4. **Check the demos** to understand agent capabilities and performance characteristics
5. **Use lazy loading** to minimize startup time and memory footprint

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgment

- Tencent [Youtu-agent](https://github.com/Tencent/Youtu-agent) toolkits integration.
