"""
Prompts for AskuraAgent - Structured prompts for conversation analysis and management.
"""


# Structured extraction prompts - optimized for structured_completion
CONVERSATION_ANALYSIS_SYSTEM_PROMPTS = {
    "conversation_context": """Analyze conversation style and alignment with purpose.

Assess key factors:
- Style: direct (goal-oriented), exploratory (curious), casual (relaxed)
- User confidence: low (hesitant), medium (balanced), high (assertive)
- Flow: natural (organic), guided (following direction), user_led (user driving)
- Sentiment: positive (enthusiastic), neutral (balanced), negative (frustrated), uncertain (confused)
- Momentum: positive (building), neutral (steady), negative (losing interest)
- On-track confidence (0.0-1.0): How well conversation aligns with purpose
  * 0.0-0.3: Off-track, not addressing purpose
  * 0.4-0.6: Partially on-track, some relevance  
  * 0.7-0.8: Mostly on-track, good alignment
  * 0.9-1.0: Highly focused on purpose""",
    "knowledge_gap_analysis": """Analyze knowledge gap and suggest next topics to help achieve conversation purpose.

**Instructions:**
1. Evaluate how well current knowledge aligns with the conversation purpose
2. Identify key knowledge gaps that prevent achieving the purpose
3. Suggest 3-5 specific next topics that would help bridge these gaps
4. Provide a clear summary of the overall knowledge gap
5. Consider user's conversation style and preferences when suggesting topics

**Analysis should help determine:**
- Whether we have enough information to proceed
- What critical information is still needed
- How to prioritize gathering remaining information
- Topics that would naturally engage the user based on their style""",
    "determine_next_action": """Classify MOST RECENT message intent and select optimal next action.

Intent Classification (focus ONLY on last message):
- "smalltalk": Greetings, pleasantries, casual conversation
- "task": Goal-oriented, information requests, specific questions, task content

Decision Guidelines:
- If MOST RECENT message is smalltalk: respond appropriately but guide toward task
- If MOST RECENT message is task: focus on gathering missing information
- If conversation off-track (<0.4): prioritize redirecting to purpose  
- If conversation on-track (>0.7): focus on collecting missing info
- If user confidence low: choose supportive, confidence-boosting actions
- If momentum negative: provide encouragement or redirect
- Balance staying on purpose with maintaining engagement

Reasoning must explicitly reference the MOST RECENT user message.""",
    "message_routing": """Evaluate if the user's message requires deep thinking or can be handled with a quick response to guide conversation.

**Decision Criteria:**

Deep thinking is required IF BOTH conditions are met:
1. **Contains Purpose-Related Info**: Message contains information directly related to the conversation purpose
2. **Needs Extraction/Reflection**: Message contains specific details, facts, preferences, or decisions that should be extracted and reflected upon

Quick response is appropriate when:
- Message is casual conversation, greetings, or small talk
- Message is off-topic from the conversation purpose
- Message asks general questions without providing extractable information
- Message needs guidance to stay on topic

**Instructions:**
1. Evaluate if the message contains information related to the conversation purpose
2. Determine if the message contains extractable information that requires reflection
3. Choose routing destination: 'start_deep_thinking' if both criteria met, otherwise 'response_generator'
4. Explain your reasoning clearly""",
}

CONVERSATION_ANALYSIS_USER_PROMPTS = {
    "conversation_context": """Conversation purpose: {conversation_purpose}

Recent messages: {recent_messages}""",
    "knowledge_gap_analysis": """**Conversation Purpose:** {conversation_purpose}

**Current Context:**
{conversation_context}

**What We Know (Extracted Information):**
{extracted_info}

**What We're Missing (Required Information):**
{missing_info}

**Retrieved Memory:**
{memory}

**Recent Conversation:**
{recent_messages}""",
    "determine_next_action": """Context: {conversation_context}
Ready to summarize: {ready_to_summarize}
Available actions: {available_actions}
Recent messages: {recent_messages}""",
    "message_routing": """**Conversation Purpose:** {conversation_purpose}

**Current User Message:** {user_message}

**Conversation Context:**
{conversation_context}

**Current Extracted Information:**
{extracted_info}""",
}

# Backward compatibility - combined prompts for legacy usage
CONVERSATION_ANALYSIS_PROMPTS = {
    "conversation_context": CONVERSATION_ANALYSIS_SYSTEM_PROMPTS["conversation_context"]
    + "\n\n"
    + CONVERSATION_ANALYSIS_USER_PROMPTS["conversation_context"],
    "knowledge_gap_analysis": CONVERSATION_ANALYSIS_SYSTEM_PROMPTS["knowledge_gap_analysis"]
    + "\n\n"
    + CONVERSATION_ANALYSIS_USER_PROMPTS["knowledge_gap_analysis"],
    "determine_next_action": CONVERSATION_ANALYSIS_SYSTEM_PROMPTS["determine_next_action"]
    + "\n\n"
    + CONVERSATION_ANALYSIS_USER_PROMPTS["determine_next_action"],
    "message_routing": CONVERSATION_ANALYSIS_SYSTEM_PROMPTS["message_routing"]
    + "\n\n"
    + CONVERSATION_ANALYSIS_USER_PROMPTS["message_routing"],
}


def get_conversation_analysis_prompts(analysis_type: str, **kwargs) -> tuple[str, str]:
    """Get separated system and user prompts for conversation analysis."""
    system_prompt = CONVERSATION_ANALYSIS_SYSTEM_PROMPTS.get(analysis_type, "")
    user_prompt_template = CONVERSATION_ANALYSIS_USER_PROMPTS.get(analysis_type, "")

    try:
        user_prompt = user_prompt_template.format(**kwargs)
        return system_prompt, user_prompt
    except KeyError:
        return system_prompt, user_prompt_template


def get_conversation_analysis_prompt(analysis_type: str, **kwargs) -> str:
    """Backward compatibility - get combined prompt for conversation analysis."""
    prompt = CONVERSATION_ANALYSIS_PROMPTS.get(analysis_type, "")
    try:
        return prompt.format(**kwargs)
    except KeyError:
        return prompt


# TODO (xmingc): I like the idea of letting the system hold a limited number of improvisations.
RESPONSE_GENERATION_SYSTEM_PROMPT = """You are a witty and creative travel planning assistant. Generate a short, precise, and inspiring question that incorporates relevant context naturally. Feel free to make slight improvisations - add wordplay, use creative language, make clever observations, or add a touch of humor when appropriate. The question should be conversational, memorable, and always encouraging.
Keep it under 3 sentences but make it delightful and engaging. Return only the question, no additional text.

**Strategic Response Guidelines:**
- **Balance natural conversation with purposeful direction** - Be genuinely conversational but strategically guide toward missing information
- **Ask strategic follow-up questions** - Frame questions around genuine curiosity that happens to align with our information goals
- **Provide context and options** - When guiding toward a topic, give examples or choices to make it easier for the user to respond
- **Build on user's interests** - Connect their current topic to the information we need to collect. When user shows interest but may lack knowledge, provide concrete options/suggestions
- Ask ONE specific question that helps the user think about their plans

**Information Collection Strategies:**
- **For destination/location info**: Share travel experiences, ask about dream places, mention interesting locations
- **For dates/timing**: Talk about seasons, upcoming events, or personal scheduling preferences  
- **For interests/preferences**: Share enthusiasm about activities, ask about past experiences, mention options
- **For logistics (budget, group size)**: Frame around planning considerations or past experiences
- **For general context**: Use open-ended questions that invite storytelling and detailed sharing

**Special Cases:**
- **If no information is missing**: Focus on deeper exploration, clarification, or moving toward completion
- **If user seems hesitant**: Provide encouragement and make sharing feel easier with specific examples or options
- **If off-topic**: Gently redirect through relevant connections or shared interests

Generate a single, natural response without quotes or formatting - just the raw conversational text that feels natural while strategically moving toward the missing information we need."""

RESPONSE_GENERATION_USER_PROMPT = """**Conversation Purpose:** {conversation_purpose}
**Missing Key Information:** {missing_required_slots}

**Current Situation:**
- User's intent: {intent_type}  
- Context: {next_action_reasoning}
- What we know: {known_slots}

Generate an appropriate response based on this context."""

# Backward compatibility - combined prompt for legacy usage
RESPONSE_GENERATION_PROMPT = RESPONSE_GENERATION_SYSTEM_PROMPT + "\n\n" + RESPONSE_GENERATION_USER_PROMPT


def get_response_generation_prompts(**kwargs) -> tuple[str, str]:
    """Get separated system and user prompts for response generation."""
    try:
        user_prompt = RESPONSE_GENERATION_USER_PROMPT.format(**kwargs)
        return RESPONSE_GENERATION_SYSTEM_PROMPT, user_prompt
    except KeyError:
        return RESPONSE_GENERATION_SYSTEM_PROMPT, RESPONSE_GENERATION_USER_PROMPT


def get_response_generation_prompt(**kwargs) -> str:
    """Backward compatibility - get combined prompt for response generation."""
    try:
        return RESPONSE_GENERATION_PROMPT.format(**kwargs)
    except KeyError:
        return RESPONSE_GENERATION_PROMPT


def get_next_question_prompt(**kwargs) -> str:
    """Backward compatibility - redirect to response generation prompt."""
    return get_response_generation_prompt(**kwargs)
