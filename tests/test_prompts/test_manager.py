"""Unit tests for PromptManager."""

import pytest

from agentic_graph_rag.graph.base import GraphSchema, NodeType, RelationshipType
from agentic_graph_rag.prompts.manager import PromptContext, PromptManager
from agentic_graph_rag.retriever.base import RetrievalStep


@pytest.fixture
def prompt_manager() -> PromptManager:
    """Create a PromptManager instance."""
    return PromptManager()


@pytest.fixture
def sample_schema() -> GraphSchema:
    """Create a sample graph schema for testing."""
    return GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                properties={"name": "STRING", "born": "INTEGER"},
                count=100,
            ),
            NodeType(
                label="Movie",
                properties={
                    "title": "STRING",
                    "released": "INTEGER",
                    "tagline": "STRING",
                },
                count=50,
            ),
        ],
        relationship_types=[
            RelationshipType(
                type="ACTED_IN",
                start_label="Person",
                end_label="Movie",
                properties={"roles": "LIST"},
            ),
            RelationshipType(
                type="DIRECTED",
                start_label="Person",
                end_label="Movie",
                properties={},
            ),
        ],
    )


@pytest.fixture
def empty_schema() -> GraphSchema:
    """Create an empty graph schema for testing."""
    return GraphSchema(node_types=[], relationship_types=[])


# --- build_system_prompt tests ---


def test_build_system_prompt_includes_node_types(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_system_prompt includes node type information."""
    prompt = prompt_manager.build_system_prompt(sample_schema)

    assert "Person" in prompt
    assert "Movie" in prompt
    assert "name: STRING" in prompt
    assert "title: STRING" in prompt


def test_build_system_prompt_includes_relationship_types(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_system_prompt includes relationship type information."""
    prompt = prompt_manager.build_system_prompt(sample_schema)

    assert "ACTED_IN" in prompt
    assert "DIRECTED" in prompt
    assert (
        "(Person)-[ACTED_IN]->(Movie)" in prompt
        or "(:Person)-[:ACTED_IN]->(:Movie)" in prompt
    )


def test_build_system_prompt_includes_node_counts(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_system_prompt includes node counts."""
    prompt = prompt_manager.build_system_prompt(sample_schema)

    assert "100" in prompt
    assert "50" in prompt


def test_build_system_prompt_includes_tools(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_system_prompt includes available tools."""
    prompt = prompt_manager.build_system_prompt(sample_schema)

    assert "execute_cypher" in prompt
    assert "vector_search" in prompt
    assert "expand_node" in prompt
    assert "submit_answer" in prompt


def test_build_system_prompt_includes_instructions(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_system_prompt includes instructions."""
    prompt = prompt_manager.build_system_prompt(sample_schema)

    assert "Cypher" in prompt
    assert "query" in prompt.lower()


def test_build_system_prompt_empty_schema(
    prompt_manager: PromptManager,
    empty_schema: GraphSchema,
) -> None:
    """build_system_prompt handles empty schema."""
    prompt = prompt_manager.build_system_prompt(empty_schema)

    assert "No schema information available" in prompt or "execute_cypher" in prompt


# --- build_retrieval_prompt tests ---


def test_build_retrieval_prompt_includes_user_query(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_retrieval_prompt includes the user query."""
    context = PromptContext(
        user_query="Who directed The Matrix?",
        schema=sample_schema,
        history=[],
    )

    prompt = prompt_manager.build_retrieval_prompt(context)

    assert "Who directed The Matrix?" in prompt


def test_build_retrieval_prompt_includes_history(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_retrieval_prompt includes retrieval history."""
    history = [
        RetrievalStep(
            action="cypher_query",
            input={"query": "MATCH (m:Movie) RETURN m.title"},
            output={"records": [{"title": "The Matrix"}]},
            error=None,
        ),
    ]
    context = PromptContext(
        user_query="Who directed The Matrix?",
        schema=sample_schema,
        history=history,
    )

    prompt = prompt_manager.build_retrieval_prompt(context)

    assert "cypher_query" in prompt
    assert "The Matrix" in prompt


def test_build_retrieval_prompt_no_history(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_retrieval_prompt handles empty history."""
    context = PromptContext(
        user_query="Who directed The Matrix?",
        schema=sample_schema,
        history=[],
    )

    prompt = prompt_manager.build_retrieval_prompt(context)

    assert "No previous steps" in prompt or "first iteration" in prompt.lower()


def test_build_retrieval_prompt_shows_error_in_history(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_retrieval_prompt shows errors from history steps."""
    history = [
        RetrievalStep(
            action="cypher_query",
            input={"query": "INVALID CYPHER"},
            output={},
            error="Syntax error at position 0",
        ),
    ]
    context = PromptContext(
        user_query="Find movies",
        schema=sample_schema,
        history=history,
    )

    prompt = prompt_manager.build_retrieval_prompt(context)

    assert "Syntax error" in prompt


def test_build_retrieval_prompt_multiple_steps(
    prompt_manager: PromptManager,
    sample_schema: GraphSchema,
) -> None:
    """build_retrieval_prompt handles multiple history steps."""
    history = [
        RetrievalStep(
            action="cypher_query",
            input={"query": "MATCH (m:Movie) RETURN m"},
            output={"records": [{"title": "The Matrix"}]},
            error=None,
        ),
        RetrievalStep(
            action="cypher_query",
            input={"query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN p.name"},
            output={"records": [{"name": "Lana Wachowski"}]},
            error=None,
        ),
    ]
    context = PromptContext(
        user_query="Who directed The Matrix?",
        schema=sample_schema,
        history=history,
    )

    prompt = prompt_manager.build_retrieval_prompt(context)

    assert "Step 1" in prompt
    assert "Step 2" in prompt


# --- format_results tests ---


def test_format_results_empty_list(prompt_manager: PromptManager) -> None:
    """format_results handles empty result list."""
    result = prompt_manager.format_results([])

    assert "No results" in result or result == "No results yet."


def test_format_results_single_record(prompt_manager: PromptManager) -> None:
    """format_results formats a single record."""
    results = [{"name": "Alice", "age": 30}]

    formatted = prompt_manager.format_results(results)

    assert "Record 1" in formatted
    assert "name" in formatted
    assert "Alice" in formatted
    assert "age" in formatted
    assert "30" in formatted


def test_format_results_multiple_records(prompt_manager: PromptManager) -> None:
    """format_results formats multiple records."""
    results = [
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Charlie"},
    ]

    formatted = prompt_manager.format_results(results)

    assert "Record 1" in formatted
    assert "Record 2" in formatted
    assert "Record 3" in formatted
    assert "Alice" in formatted
    assert "Bob" in formatted
    assert "Charlie" in formatted


def test_format_results_nested_dict(prompt_manager: PromptManager) -> None:
    """format_results handles nested dictionaries."""
    results = [{"movie": {"title": "The Matrix", "year": 1999}}]

    formatted = prompt_manager.format_results(results)

    assert "movie" in formatted
    assert "The Matrix" in formatted


def test_format_results_list_value(prompt_manager: PromptManager) -> None:
    """format_results handles list values."""
    results = [{"roles": ["Neo", "Trinity"]}]

    formatted = prompt_manager.format_results(results)

    assert "roles" in formatted
    assert "Neo" in formatted


def test_format_results_long_list_truncated(prompt_manager: PromptManager) -> None:
    """format_results truncates long lists."""
    results = [{"items": list(range(20))}]

    formatted = prompt_manager.format_results(results)

    # Should indicate the count rather than showing all items
    assert "items" in formatted


# --- PromptContext tests ---


def test_prompt_context_creation(sample_schema: GraphSchema) -> None:
    """PromptContext can be created with required fields."""
    context = PromptContext(
        user_query="Test query",
        schema=sample_schema,
        history=[],
    )

    assert context.user_query == "Test query"
    assert context.schema == sample_schema
    assert context.history == []
    assert context.examples is None


def test_prompt_context_with_examples(sample_schema: GraphSchema) -> None:
    """PromptContext can include examples."""
    examples = [{"query": "Find all movies", "cypher": "MATCH (m:Movie) RETURN m"}]
    context = PromptContext(
        user_query="Test query",
        schema=sample_schema,
        history=[],
        examples=examples,
    )

    assert context.examples == examples


# --- Schema formatting edge cases ---


def test_build_system_prompt_node_without_properties(
    prompt_manager: PromptManager,
) -> None:
    """build_system_prompt handles nodes with no properties."""
    schema = GraphSchema(
        node_types=[
            NodeType(label="Tag", properties={}, count=10),
        ],
        relationship_types=[],
    )

    prompt = prompt_manager.build_system_prompt(schema)

    assert "Tag" in prompt
    assert "none" in prompt.lower() or "Properties" in prompt


def test_build_system_prompt_relationship_without_properties(
    prompt_manager: PromptManager,
) -> None:
    """build_system_prompt handles relationships with no properties."""
    schema = GraphSchema(
        node_types=[
            NodeType(label="Person", properties={}, count=10),
            NodeType(label="Movie", properties={}, count=5),
        ],
        relationship_types=[
            RelationshipType(
                type="LIKES",
                start_label="Person",
                end_label="Movie",
                properties={},
            ),
        ],
    )

    prompt = prompt_manager.build_system_prompt(schema)

    assert "LIKES" in prompt
    assert "Person" in prompt
    assert "Movie" in prompt
