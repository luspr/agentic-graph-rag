"""Unit tests for the SessionManager module."""

from datetime import datetime


from agentic_graph_rag.agent.session import Session, SessionManager, SessionMessage


# --- SessionMessage tests ---


def test_session_message_has_required_fields() -> None:
    """SessionMessage has role and content fields."""
    msg = SessionMessage(
        role="user",
        content="What is The Matrix?",
    )

    assert msg.role == "user"
    assert msg.content == "What is The Matrix?"


def test_session_message_trace_id_defaults_none() -> None:
    """SessionMessage trace_id defaults to None."""
    msg = SessionMessage(
        role="assistant",
        content="The Matrix is a movie.",
    )

    assert msg.trace_id is None


def test_session_message_accepts_trace_id() -> None:
    """SessionMessage accepts optional trace_id."""
    msg = SessionMessage(
        role="assistant",
        content="Answer here",
        trace_id="trace-123",
    )

    assert msg.trace_id == "trace-123"


# --- Session tests ---


def test_session_has_required_fields() -> None:
    """Session has session_id field."""
    session = Session(session_id="session-abc")

    assert session.session_id == "session-abc"


def test_session_messages_default_empty() -> None:
    """Session messages defaults to empty list."""
    session = Session(session_id="test")

    assert session.messages == []


def test_session_created_at_defaults_to_now() -> None:
    """Session created_at defaults to current time."""
    before = datetime.now()
    session = Session(session_id="test")
    after = datetime.now()

    assert before <= session.created_at <= after


def test_session_accepts_custom_created_at() -> None:
    """Session accepts custom created_at timestamp."""
    custom_time = datetime(2024, 1, 15, 10, 30, 0)
    session = Session(session_id="test", created_at=custom_time)

    assert session.created_at == custom_time


# --- SessionManager.create_session() tests ---


def test_create_session_returns_session() -> None:
    """create_session returns a Session object."""
    manager = SessionManager()

    session = manager.create_session()

    assert isinstance(session, Session)


def test_create_session_generates_unique_ids() -> None:
    """create_session generates unique session IDs."""
    manager = SessionManager()

    session1 = manager.create_session()
    session2 = manager.create_session()

    assert session1.session_id != session2.session_id


def test_create_session_sets_created_at() -> None:
    """create_session sets the created_at timestamp."""
    manager = SessionManager()
    before = datetime.now()

    session = manager.create_session()

    after = datetime.now()
    assert before <= session.created_at <= after


def test_create_session_starts_with_empty_messages() -> None:
    """create_session creates session with empty messages list."""
    manager = SessionManager()

    session = manager.create_session()

    assert session.messages == []


def test_create_session_stores_session() -> None:
    """create_session stores the session for later retrieval."""
    manager = SessionManager()

    session = manager.create_session()

    assert manager.get_session(session.session_id) == session


# --- SessionManager.add_message() tests ---


def test_add_message_appends_to_session() -> None:
    """add_message appends a message to the session."""
    manager = SessionManager()
    session = manager.create_session()

    manager.add_message(session, "user", "Hello")

    assert len(session.messages) == 1
    assert session.messages[0].content == "Hello"


def test_add_message_sets_role() -> None:
    """add_message sets the message role."""
    manager = SessionManager()
    session = manager.create_session()

    manager.add_message(session, "user", "Question")
    manager.add_message(session, "assistant", "Answer")

    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"


def test_add_message_accepts_trace_id() -> None:
    """add_message accepts optional trace_id."""
    manager = SessionManager()
    session = manager.create_session()

    manager.add_message(session, "assistant", "Answer", trace_id="trace-456")

    assert session.messages[0].trace_id == "trace-456"


def test_add_message_preserves_order() -> None:
    """add_message preserves message order."""
    manager = SessionManager()
    session = manager.create_session()

    manager.add_message(session, "user", "First")
    manager.add_message(session, "assistant", "Second")
    manager.add_message(session, "user", "Third")

    contents = [m.content for m in session.messages]
    assert contents == ["First", "Second", "Third"]


# --- SessionManager.get_context_messages() tests ---


def test_get_context_messages_returns_list() -> None:
    """get_context_messages returns a list."""
    manager = SessionManager()
    session = manager.create_session()

    result = manager.get_context_messages(session)

    assert isinstance(result, list)


def test_get_context_messages_returns_empty_for_empty_session() -> None:
    """get_context_messages returns empty list for empty session."""
    manager = SessionManager()
    session = manager.create_session()

    result = manager.get_context_messages(session)

    assert result == []


def test_get_context_messages_formats_as_dicts() -> None:
    """get_context_messages returns messages as dicts with role and content."""
    manager = SessionManager()
    session = manager.create_session()
    manager.add_message(session, "user", "Hello")
    manager.add_message(session, "assistant", "Hi there")

    result = manager.get_context_messages(session)

    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1] == {"role": "assistant", "content": "Hi there"}


def test_get_context_messages_respects_max_messages() -> None:
    """get_context_messages returns only the most recent max_messages."""
    manager = SessionManager()
    session = manager.create_session()
    for i in range(10):
        manager.add_message(session, "user", f"Message {i}")

    result = manager.get_context_messages(session, max_messages=3)

    assert len(result) == 3
    assert result[0]["content"] == "Message 7"
    assert result[1]["content"] == "Message 8"
    assert result[2]["content"] == "Message 9"


def test_get_context_messages_returns_all_when_under_limit() -> None:
    """get_context_messages returns all messages when under max_messages."""
    manager = SessionManager()
    session = manager.create_session()
    manager.add_message(session, "user", "First")
    manager.add_message(session, "assistant", "Second")

    result = manager.get_context_messages(session, max_messages=10)

    assert len(result) == 2


def test_get_context_messages_default_limit_is_ten() -> None:
    """get_context_messages defaults to max_messages=10."""
    manager = SessionManager()
    session = manager.create_session()
    for i in range(15):
        manager.add_message(session, "user", f"Message {i}")

    result = manager.get_context_messages(session)

    assert len(result) == 10


def test_get_context_messages_zero_limit_returns_empty() -> None:
    """get_context_messages with max_messages=0 returns empty list."""
    manager = SessionManager()
    session = manager.create_session()
    manager.add_message(session, "user", "Hello")

    result = manager.get_context_messages(session, max_messages=0)

    assert result == []


def test_get_context_messages_does_not_include_trace_id() -> None:
    """get_context_messages output does not include trace_id."""
    manager = SessionManager()
    session = manager.create_session()
    manager.add_message(session, "assistant", "Answer", trace_id="trace-789")

    result = manager.get_context_messages(session)

    assert "trace_id" not in result[0]


# --- SessionManager.get_session() tests ---


def test_get_session_returns_session_by_id() -> None:
    """get_session retrieves session by ID."""
    manager = SessionManager()
    session = manager.create_session()

    retrieved = manager.get_session(session.session_id)

    assert retrieved == session


def test_get_session_returns_none_for_unknown_id() -> None:
    """get_session returns None for unknown IDs."""
    manager = SessionManager()

    assert manager.get_session("nonexistent") is None


# --- SessionManager.get_message_count() tests ---


def test_get_message_count_returns_zero_for_empty() -> None:
    """get_message_count returns 0 for empty session."""
    manager = SessionManager()
    session = manager.create_session()

    assert manager.get_message_count(session) == 0


def test_get_message_count_returns_correct_count() -> None:
    """get_message_count returns the number of messages."""
    manager = SessionManager()
    session = manager.create_session()
    manager.add_message(session, "user", "One")
    manager.add_message(session, "assistant", "Two")
    manager.add_message(session, "user", "Three")

    assert manager.get_message_count(session) == 3


# --- SessionManager.clear_session() tests ---


def test_clear_session_removes_all_messages() -> None:
    """clear_session removes all messages from session."""
    manager = SessionManager()
    session = manager.create_session()
    manager.add_message(session, "user", "Hello")
    manager.add_message(session, "assistant", "Hi")

    manager.clear_session(session)

    assert session.messages == []


def test_clear_session_preserves_session_metadata() -> None:
    """clear_session preserves session_id and created_at."""
    manager = SessionManager()
    session = manager.create_session()
    original_id = session.session_id
    original_time = session.created_at
    manager.add_message(session, "user", "Hello")

    manager.clear_session(session)

    assert session.session_id == original_id
    assert session.created_at == original_time


# --- SessionManager.delete_session() tests ---


def test_delete_session_removes_session() -> None:
    """delete_session removes the session from storage."""
    manager = SessionManager()
    session = manager.create_session()
    session_id = session.session_id

    manager.delete_session(session_id)

    assert manager.get_session(session_id) is None


def test_delete_session_returns_true_on_success() -> None:
    """delete_session returns True when session is deleted."""
    manager = SessionManager()
    session = manager.create_session()

    result = manager.delete_session(session.session_id)

    assert result is True


def test_delete_session_returns_false_for_unknown() -> None:
    """delete_session returns False for unknown session IDs."""
    manager = SessionManager()

    result = manager.delete_session("nonexistent")

    assert result is False


# --- Multi-session tests ---


def test_multiple_sessions_are_independent() -> None:
    """Messages added to one session don't affect others."""
    manager = SessionManager()
    session1 = manager.create_session()
    session2 = manager.create_session()

    manager.add_message(session1, "user", "Message for session 1")
    manager.add_message(session2, "user", "Message for session 2")

    assert len(session1.messages) == 1
    assert len(session2.messages) == 1
    assert session1.messages[0].content == "Message for session 1"
    assert session2.messages[0].content == "Message for session 2"


def test_manager_tracks_multiple_sessions() -> None:
    """SessionManager can track multiple sessions simultaneously."""
    manager = SessionManager()

    sessions = [manager.create_session() for _ in range(5)]

    for session in sessions:
        assert manager.get_session(session.session_id) is not None


def test_deleting_one_session_does_not_affect_others() -> None:
    """Deleting one session doesn't affect other sessions."""
    manager = SessionManager()
    session1 = manager.create_session()
    session2 = manager.create_session()

    manager.delete_session(session1.session_id)

    assert manager.get_session(session1.session_id) is None
    assert manager.get_session(session2.session_id) == session2
