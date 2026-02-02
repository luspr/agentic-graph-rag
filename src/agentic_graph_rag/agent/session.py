"""Session manager for maintaining conversation history."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass
class SessionMessage:
    """A message in the session history."""

    role: str  # "user" or "assistant"
    content: str
    trace_id: str | None = None


@dataclass
class Session:
    """A user session with conversation history."""

    session_id: str
    messages: list[SessionMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class SessionManager:
    """Manages session state and conversation history.

    The session manager maintains conversation history across multiple user queries
    within a session. This enables multi-turn conversations where context from
    previous exchanges can inform the agent's responses.

    Sessions are stored in memory and do not persist across application restarts.

    Example:
        manager = SessionManager()
        session = manager.create_session()
        manager.add_message(session, "user", "What movies did Tom Hanks act in?")
        manager.add_message(session, "assistant", "Tom Hanks acted in...", trace_id="abc")
        context = manager.get_context_messages(session, max_messages=10)
    """

    def __init__(self) -> None:
        """Initialize the session manager."""
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new session with a unique ID.

        Returns:
            A new Session object ready to store messages.
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            messages=[],
            created_at=datetime.now(),
        )
        self._sessions[session_id] = session
        return session

    def add_message(
        self,
        session: Session,
        role: str,
        content: str,
        trace_id: str | None = None,
    ) -> None:
        """Add a message to the session history.

        Args:
            session: The session to add the message to.
            role: The role of the message sender ("user" or "assistant").
            content: The message content.
            trace_id: Optional trace ID linking to detailed execution trace.
        """
        message = SessionMessage(
            role=role,
            content=content,
            trace_id=trace_id,
        )
        session.messages.append(message)

    def get_context_messages(
        self,
        session: Session,
        max_messages: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent messages formatted for LLM context.

        Returns messages in the format expected by LLM APIs (list of dicts
        with 'role' and 'content' keys).

        Args:
            session: The session to get messages from.
            max_messages: Maximum number of messages to return. Returns the
                most recent messages if the session has more.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        # Get the most recent messages up to max_messages
        recent_messages = session.messages[-max_messages:] if max_messages > 0 else []

        return [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The Session if found, None otherwise.
        """
        return self._sessions.get(session_id)

    def get_message_count(self, session: Session) -> int:
        """Get the number of messages in a session.

        Args:
            session: The session to count messages for.

        Returns:
            The number of messages in the session.
        """
        return len(session.messages)

    def clear_session(self, session: Session) -> None:
        """Clear all messages from a session.

        Args:
            session: The session to clear.
        """
        session.messages.clear()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if the session was deleted, False if not found.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
