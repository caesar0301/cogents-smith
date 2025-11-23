"""
Prompts for the DeepResearchAgent agent.
"""

query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries for research. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Each query should be LESS than 40 characters.
- Rewritten queries should be in the same language as the original query.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format:
- Format your response as a JSON object with ALL three of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: Research the latest developments in renewable energy
```json
{{
    "rationale": "To gather comprehensive information about renewable energy developments, we need current data on technological advances, market trends, and policy updates. These queries target the specific information needed for thorough research.",
    "query": ["renewable energy development 2024", "solar power technology advances", "renewable energy market trends"]
}}
```

Context: {research_topic}"""


reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- The follow-up query should be less than 40 characters.
- The follow-up query should be in the same language as the original query.
- Focus on gathering comprehensive and accurate information relevant to the research topic.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about recent developments and current market conditions", // "" if is_sufficient is true
    "follow_up_queries": ["example follow-up query"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""


answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step.
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- You MUST include all the citations (if available in the summaries) in the answer correctly.
- DO NOT mention summary indicators in the answer.
- Structure the answer logically and comprehensively.
- Include specific details, facts, and current information when available.
- Provide actionable insights and practical information when relevant.

User Context:
- {research_topic}

Summaries:
{summaries}"""
