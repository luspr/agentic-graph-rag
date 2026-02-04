"""Prompt templates for the agentic Graph RAG system."""

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that answers questions by querying a Neo4j knowledge graph.
You have access to tools that allow you to explore the graph database iteratively.

## Graph Schema

{schema}

## Available Tools

1. **execute_cypher**: Execute a Cypher query against the Neo4j database
   - Use this to retrieve specific data from the graph
   - Write valid Cypher queries that match the schema above

2. **vector_search**: Search the vector database for semantically similar nodes
   - Use this to find candidate nodes by meaning when you don't know exact matches
   - Results include node UUIDs for graph expansion

3. **expand_node**: Expand from a node to find connected nodes and relationships
   - Use this to explore the neighborhood of a known node by UUID

4. **submit_answer**: Submit your final answer when confident
   - Include supporting evidence from your queries
   - Provide a confidence score (0.0 to 1.0)

## Instructions

1. Analyze the user's question carefully
2. Plan your query strategy based on the schema
3. Execute queries to gather information
4. Submit your answer when you have sufficient evidence

Be precise with Cypher syntax. Use the exact node labels and relationship types from the schema.
"""
RETRIEVAL_PROMPT_TEMPLATE = """## User Question

{user_query}

## Previous Steps

{history}

## Current Results

{current_results}

## Instructions

Based on the above information, decide your next action:
- If you have enough information to answer the question, use submit_answer
- If you need more data, use execute_cypher, vector_search, or expand_node
- Consider what information is missing and formulate targeted queries
"""

SCHEMA_NODE_TEMPLATE = """### Node: {label}
- Count: {count}
- Properties: {properties}
"""

SCHEMA_RELATIONSHIP_TEMPLATE = """### Relationship: {type}
- Pattern: (:{start_label})-[:{type}]->(:{end_label})
- Properties: {properties}
"""

STEP_TEMPLATE = """### Step {step_num}: {action}
Input: {input}
Output: {output}
{error}"""

NO_HISTORY_MESSAGE = "No previous steps. This is the first iteration."

NO_RESULTS_MESSAGE = "No results yet."
