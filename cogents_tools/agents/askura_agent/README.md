# Askura Agent

A dynamic conversation agent that adapts to user communication styles and maintains conversation purpose alignment.

## Key Features

### Conversation Purpose Alignment
The Askura agent evaluates how well conversations align with their intended purpose and provides confidence scores for tracking.

- **Purpose Evaluation**: Analyzes conversation relevance to stated goals
- **Confidence Scoring**: Provides 0.0-1.0 confidence scores for alignment
- **Dynamic Adaptation**: Adjusts conversation flow based on purpose alignment
- **Smart Redirection**: Automatically redirects off-track conversations

### Conversation Analysis
- **Style Detection**: Identifies direct, exploratory, or casual communication styles
- **Sentiment Analysis**: Monitors user sentiment and conversation momentum
- **Information Density**: Measures how much information is packed into messages
- **Depth Assessment**: Evaluates conversation depth (surface, moderate, deep)

### Adaptive Response Generation
- **Unified Next Action Determination**: Single LLM call for intent classification and action selection
- **Context-Aware Actions**: Selects next actions based on conversation context
- **Confidence Boosting**: Provides guidance when user confidence is low
- **Momentum Maintenance**: Keeps conversations engaging and on-track
- **Purpose-Driven Decisions**: Prioritizes actions that align with conversation goals

## Architecture

### Core Components

1. **ConversationManager** (`conversation_manager.py`)
   - Analyzes conversation context and purpose alignment
   - Determines optimal next actions
   - Manages conversation flow and redirection

2. **Structured Prompts** (`prompts.py`)
   - Centralized prompt management
   - Purpose-aware conversation analysis
   - Context-driven action selection
   - Unified intent classification and action determination

3. **Schemas** (`schemas.py`)
   - Type-safe data structures
   - Conversation context models
   - Action response validation

### Purpose Alignment Evaluation

The agent evaluates conversation alignment using:

```python
# Confidence levels
0.0-0.3: Off-track (not addressing purpose)
0.4-0.6: Partially on-track (some relevance)
0.7-0.8: Mostly on-track (good alignment)
0.9-1.0: Highly focused (excellent alignment)
```

### Action Selection Logic

- **High Confidence (>0.7)**: Focus on gathering missing information
- **Low Confidence (<0.4)**: Prioritize redirecting to purpose
- **Moderate Confidence (0.4-0.7)**: Balance purpose alignment and information gathering

### Unified Next Action Determination

The agent uses a single LLM call to:
- **Classify User Intent**: Distinguish between smalltalk and task-oriented conversation
- **Select Optimal Action**: Choose the best next action based on intent and context
- **Provide Reasoning**: Explain why the action was chosen
- **Assess Confidence**: Provide confidence scores for decision quality

**Benefits:**
- **Consistency**: Intent and action decisions are made together
- **Efficiency**: Reduces from 2-3 LLM calls to 1 unified call
- **Better Reasoning**: Full context consideration for nuanced decisions
- **Structured Output**: Type-safe responses with validation

## Usage Example

```python
from cogents.agents.askura_agent import ConversationManager
from cogents.agents.askura_agent.schemas import AskuraConfig

# Configure the agent
config = AskuraConfig(
    conversation_purposes=["travel planning"],
    enable_style_adaptation=True,
    enable_sentiment_analysis=True
)

# Create conversation manager
manager = ConversationManager(config=config, llm_client=llm_client)

# Analyze conversation context
context = manager.analyze_conversation_context(state)

# Check purpose alignment
print(f"On-track confidence: {context.conversation_on_track_confidence}")
print(f"Conversation purpose: {context.conversation_purpose}")

# Determine next action
next_action = manager.determine_next_action(state, context)
```

## Configuration

### AskuraConfig Options

- `conversation_purposes`: List of conversation goals
- `enable_style_adaptation`: Adapt to user communication style
- `enable_sentiment_analysis`: Monitor user sentiment
- `enable_confidence_boosting`: Provide guidance for low confidence
- `max_conversation_turns`: Limit conversation length
- `information_slots`: Define required information to collect

## Examples

See `examples/determine_next_action_example.py` for a complete demonstration of:
- Unified intent classification and action selection
- Different conversation scenarios (smalltalk, task, ready to summarize)
- Benefits of unified approach
- Fallback handling

See `examples/conversation_purpose_alignment_example.py` for a complete demonstration of:
- Purpose alignment evaluation
- Conversation context analysis
- Next action selection
- Different confidence scenarios

## Benefits

1. **Purpose-Driven**: Maintains focus on conversation goals
2. **Adaptive**: Adjusts to user communication preferences
3. **Intelligent**: Uses LLM-powered analysis for better decisions
4. **Reliable**: Structured completion with retry logic
5. **Maintainable**: Centralized prompts and clear architecture
