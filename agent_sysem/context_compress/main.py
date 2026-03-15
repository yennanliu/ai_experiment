"""
Context Compression & Summarization for Long-Running AI Conversations

This module implements 5 progressive strategies for managing token usage:
1. Sliding Window - Keep only recent N turns
2. Progressive Summarization - Compress older conversations into summaries
3. Hierarchical Summarization - Multi-layer structure across sessions
4. Semantic Compression - Extract meaning units (facts/decisions/questions)
5. Hybrid Compression - Production system combining all strategies

Reference: https://yennj12.js.org/yennj12_blog_V4/posts/context-compression-summarization-guide-zh/
"""

import anthropic
import json
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime
from pathlib import Path

# Initialize Anthropic client
client = anthropic.Anthropic()


# =============================================================================
# Strategy 1: Sliding Window Manager
# =============================================================================

@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window context management."""
    max_turns: int = 10  # Maximum conversation turns to keep
    max_tokens: int = 8000  # Maximum estimated tokens
    tokens_per_char: float = 0.4  # Rough estimation ratio


class SlidingWindowManager:
    """
    Simple sliding window that keeps only the most recent N turns.
    Oldest messages are dropped when limits are exceeded.
    """

    def __init__(self, config: Optional[SlidingWindowConfig] = None):
        self.config = config or SlidingWindowConfig()
        self.messages: list[dict] = []
        self._stats = {"total_messages": 0, "dropped_messages": 0}

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return int(len(text) * self.config.tokens_per_char)

    def _total_tokens(self) -> int:
        """Calculate total estimated tokens in current context."""
        return sum(
            self._estimate_tokens(m["content"])
            for m in self.messages
        )

    def add_message(self, role: str, content: str):
        """Add a message and trim if needed."""
        self.messages.append({"role": role, "content": content})
        self._stats["total_messages"] += 1
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Remove oldest messages if limits exceeded."""
        # Trim by turn count (each turn = 2 messages: user + assistant)
        while len(self.messages) > self.config.max_turns * 2:
            self.messages.pop(0)
            self._stats["dropped_messages"] += 1

        # Trim by token count
        while self._total_tokens() > self.config.max_tokens and len(self.messages) > 2:
            self.messages.pop(0)
            self._stats["dropped_messages"] += 1

    def get_messages(self) -> list[dict]:
        """Get current message context."""
        return self.messages.copy()

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            **self._stats,
            "current_messages": len(self.messages),
            "estimated_tokens": self._total_tokens()
        }


# =============================================================================
# Strategy 2: Progressive Summarization
# =============================================================================

@dataclass
class ProgressiveSummaryConfig:
    """Configuration for progressive summarization."""
    summarize_threshold: int = 5  # Turns before summarization
    recent_turns_to_keep: int = 3  # Recent turns kept complete
    max_summary_tokens: int = 500  # Max tokens for summary


@dataclass
class ConversationState:
    """State for progressive summarization."""
    summary: str = ""  # Running summary of older conversations
    recent_messages: list = field(default_factory=list)  # Recent complete messages
    total_turns: int = 0


class ProgressiveSummarizer:
    """
    Compresses older conversations into summaries while keeping
    recent messages complete for context continuity.
    """

    def __init__(self, config: Optional[ProgressiveSummaryConfig] = None):
        self.config = config or ProgressiveSummaryConfig()
        self.state = ConversationState()

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages into text for summarization."""
        return "\n".join(
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        )

    def _create_summary_prompt(self, messages_text: str, existing_summary: str) -> str:
        """Create prompt for generating summary."""
        if existing_summary:
            return f"""Please update this conversation summary with new content.

Existing summary:
{existing_summary}

New conversation to incorporate:
{messages_text}

Generate an updated summary that:
1. Preserves key decisions, facts, and context
2. Removes redundant information
3. Maintains chronological flow
4. Stays under 500 tokens

Updated summary:"""
        else:
            return f"""Please summarize this conversation, focusing on:
1. Key decisions made
2. Important facts established
3. User preferences and requirements
4. Ongoing tasks or questions

Conversation:
{messages_text}

Summary:"""

    def _generate_summary(self, messages: list[dict]) -> str:
        """Use LLM to generate summary of messages."""
        messages_text = self._format_messages_for_summary(messages)
        prompt = self._create_summary_prompt(messages_text, self.state.summary)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=self.config.max_summary_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def add_turn(self, user_message: str, assistant_message: str):
        """Add a conversation turn and check if summarization needed."""
        self.state.recent_messages.append({"role": "user", "content": user_message})
        self.state.recent_messages.append({"role": "assistant", "content": assistant_message})
        self.state.total_turns += 1
        self._check_and_summarize()

    def _check_and_summarize(self):
        """Check if summarization threshold reached and summarize if needed."""
        recent_turn_count = len(self.state.recent_messages) // 2

        if recent_turn_count > self.config.summarize_threshold:
            # Keep only the most recent turns
            keep_count = self.config.recent_turns_to_keep * 2
            messages_to_summarize = self.state.recent_messages[:-keep_count]

            # Generate new summary
            self.state.summary = self._generate_summary(messages_to_summarize)

            # Keep only recent messages
            self.state.recent_messages = self.state.recent_messages[-keep_count:]

    def get_context(self) -> str:
        """Get the current context (summary + recent)."""
        parts = []
        if self.state.summary:
            parts.append(f"[Previous conversation summary]\n{self.state.summary}")
        if self.state.recent_messages:
            parts.append("[Recent conversation]\n" +
                        self._format_messages_for_summary(self.state.recent_messages))
        return "\n\n".join(parts)

    def get_messages(self) -> list[dict]:
        """Get messages formatted for API call."""
        messages = []

        if self.state.summary:
            messages.append({
                "role": "user",
                "content": f"[Context from previous conversation]\n{self.state.summary}\n\nPlease continue based on this context."
            })
            messages.append({
                "role": "assistant",
                "content": "I understand the context. Please continue with your question."
            })

        messages.extend(self.state.recent_messages)
        return messages


# =============================================================================
# Strategy 3: Hierarchical Memory
# =============================================================================

@dataclass
class HierarchicalSummaryConfig:
    """Configuration for hierarchical memory management."""
    session_summary_threshold: int = 10  # Turns before session summary
    max_session_summary_tokens: int = 500
    max_cross_session_facts: int = 10
    profile_path: str = "user_profile.json"


@dataclass
class UserProfile:
    """Long-term user profile across sessions."""
    name: Optional[str] = None
    role: Optional[str] = None
    preferences: list[str] = field(default_factory=list)
    expertise: list[str] = field(default_factory=list)
    ongoing_projects: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)  # Cross-session facts

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "preferences": self.preferences,
            "expertise": self.expertise,
            "ongoing_projects": self.ongoing_projects,
            "facts": self.facts
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(**data)


@dataclass
class SessionState:
    """State for current session."""
    session_id: str
    started_at: str
    summary: str = ""
    recent_messages: list = field(default_factory=list)
    total_turns: int = 0


class HierarchicalMemoryManager:
    """
    Multi-layer memory structure:
    - Layer 1: Recent messages (complete)
    - Layer 2: Session summary
    - Layer 3: Cross-session facts
    - Layer 4: User profile
    """

    def __init__(self, config: Optional[HierarchicalSummaryConfig] = None):
        self.config = config or HierarchicalSummaryConfig()
        self.user_profile = self._load_user_profile()
        self.session = SessionState(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat()
        )

    def _load_user_profile(self) -> UserProfile:
        """Load user profile from file or create new."""
        profile_path = Path(self.config.profile_path)
        if profile_path.exists():
            data = json.loads(profile_path.read_text())
            return UserProfile.from_dict(data)
        return UserProfile()

    def _save_user_profile(self):
        """Save user profile to file."""
        profile_path = Path(self.config.profile_path)
        profile_path.write_text(json.dumps(self.user_profile.to_dict(), indent=2))

    def _update_user_profile(self, messages: list[dict]):
        """Extract and update user profile from messages."""
        messages_text = "\n".join(
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        )

        prompt = f"""Analyze this conversation and extract user profile information.
Return JSON with any new information found:
- name: User's name if mentioned
- role: User's job/role if mentioned
- preferences: List of stated preferences
- expertise: Areas of expertise mentioned
- ongoing_projects: Projects being worked on
- facts: Important facts to remember across sessions

Current profile: {json.dumps(self.user_profile.to_dict())}

Conversation:
{messages_text}

Return only new/updated information as JSON (empty object if nothing new):"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            # Parse response and merge with existing profile
            text = response.content[0].text
            # Try to extract JSON from the response
            if "{" in text:
                json_str = text[text.find("{"):text.rfind("}") + 1]
                updates = json.loads(json_str)

                if updates.get("name"):
                    self.user_profile.name = updates["name"]
                if updates.get("role"):
                    self.user_profile.role = updates["role"]
                for pref in updates.get("preferences", []):
                    if pref not in self.user_profile.preferences:
                        self.user_profile.preferences.append(pref)
                for exp in updates.get("expertise", []):
                    if exp not in self.user_profile.expertise:
                        self.user_profile.expertise.append(exp)
                for proj in updates.get("ongoing_projects", []):
                    if proj not in self.user_profile.ongoing_projects:
                        self.user_profile.ongoing_projects.append(proj)
                for fact in updates.get("facts", []):
                    if fact not in self.user_profile.facts:
                        self.user_profile.facts.append(fact)
                        # Keep only max facts
                        if len(self.user_profile.facts) > self.config.max_cross_session_facts:
                            self.user_profile.facts.pop(0)

                self._save_user_profile()
        except (json.JSONDecodeError, KeyError):
            pass  # Ignore parsing errors

    def _generate_session_summary(self, messages: list[dict]) -> str:
        """Generate summary for session."""
        messages_text = "\n".join(
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        )

        prompt = f"""Summarize this conversation segment, focusing on:
1. Key decisions and conclusions
2. Tasks completed or in progress
3. Important context for continuation

Previous summary: {self.session.summary if self.session.summary else 'None'}

New messages:
{messages_text}

Updated summary:"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=self.config.max_session_summary_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def add_turn(self, user_message: str, assistant_message: str):
        """Add a conversation turn."""
        self.session.recent_messages.append({"role": "user", "content": user_message})
        self.session.recent_messages.append({"role": "assistant", "content": assistant_message})
        self.session.total_turns += 1

        # Check if we need to summarize
        if self.session.total_turns % self.config.session_summary_threshold == 0:
            self._update_user_profile(self.session.recent_messages)
            self.session.summary = self._generate_session_summary(self.session.recent_messages)
            # Keep only last 3 turns (6 messages)
            self.session.recent_messages = self.session.recent_messages[-6:]

    def get_context(self) -> str:
        """Get full hierarchical context."""
        parts = []

        # User profile context
        if self.user_profile.name or self.user_profile.preferences:
            profile_text = "User Profile:\n"
            if self.user_profile.name:
                profile_text += f"- Name: {self.user_profile.name}\n"
            if self.user_profile.role:
                profile_text += f"- Role: {self.user_profile.role}\n"
            if self.user_profile.preferences:
                profile_text += f"- Preferences: {', '.join(self.user_profile.preferences)}\n"
            if self.user_profile.expertise:
                profile_text += f"- Expertise: {', '.join(self.user_profile.expertise)}\n"
            if self.user_profile.ongoing_projects:
                profile_text += f"- Projects: {', '.join(self.user_profile.ongoing_projects)}\n"
            if self.user_profile.facts:
                profile_text += f"- Key facts: {'; '.join(self.user_profile.facts)}\n"
            parts.append(profile_text)

        # Session summary
        if self.session.summary:
            parts.append(f"Session Summary:\n{self.session.summary}")

        return "\n\n".join(parts)

    def get_messages(self) -> list[dict]:
        """Get messages for API call."""
        messages = []

        context = self.get_context()
        if context:
            messages.append({
                "role": "user",
                "content": f"[Context]\n{context}\n\nPlease continue based on above context."
            })
            messages.append({
                "role": "assistant",
                "content": "I understand the context. How can I help you?"
            })

        messages.extend(self.session.recent_messages)
        return messages


# =============================================================================
# Strategy 4: Semantic Compression
# =============================================================================

class ImportanceLevel(Enum):
    """Importance levels for semantic units."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SemanticUnit:
    """A semantic unit extracted from conversation."""
    content: str
    unit_type: str  # "fact", "decision", "question", "preference", "context"
    importance: ImportanceLevel
    turn_created: int
    related_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.unit_type,
            "importance": self.importance.value,
            "turn": self.turn_created,
            "entities": self.related_entities
        }

    @classmethod
    def from_dict(cls, data: dict, turn: int) -> "SemanticUnit":
        return cls(
            content=data.get("content", ""),
            unit_type=data.get("type", "context"),
            importance=ImportanceLevel(data.get("importance", "medium")),
            turn_created=turn,
            related_entities=data.get("entities", [])
        )


@dataclass
class SemanticConfig:
    """Configuration for semantic compression."""
    max_units: int = 20
    extraction_interval: int = 3  # Extract every N turns
    recent_turns_to_keep: int = 2


class SemanticCompressor:
    """
    Extracts meaning units (facts, decisions, questions) from conversation
    rather than keeping full text.
    """

    def __init__(self, config: Optional[SemanticConfig] = None):
        self.config = config or SemanticConfig()
        self.semantic_units: list[SemanticUnit] = []
        self.recent_messages: list[dict] = []
        self.turn_count: int = 0

    def _extract_semantic_units(self, messages: list[dict]) -> list[SemanticUnit]:
        """Extract semantic units from messages using LLM."""
        messages_text = "\n".join(
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        )

        prompt = f"""Analyze this conversation and extract semantic units.
Return a JSON array where each unit has:
- content: The key information (concise)
- type: One of "fact", "decision", "question", "preference", "context"
- importance: One of "critical", "high", "medium", "low"
- entities: List of related entities (names, topics, etc.)

Focus on information that would be needed to continue the conversation.
Skip greetings, acknowledgments, and redundant information.

Conversation:
{messages_text}

Return only the JSON array:"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        units = []
        try:
            text = response.content[0].text
            if "[" in text:
                json_str = text[text.find("["):text.rfind("]") + 1]
                data = json.loads(json_str)
                for item in data:
                    units.append(SemanticUnit.from_dict(item, self.turn_count))
        except (json.JSONDecodeError, KeyError):
            pass

        return units

    def _prune_units(self):
        """Remove low-importance units if exceeding max."""
        if len(self.semantic_units) <= self.config.max_units:
            return

        # Sort by importance and recency
        importance_order = {
            ImportanceLevel.CRITICAL: 4,
            ImportanceLevel.HIGH: 3,
            ImportanceLevel.MEDIUM: 2,
            ImportanceLevel.LOW: 1
        }

        self.semantic_units.sort(
            key=lambda u: (importance_order[u.importance], u.turn_created),
            reverse=True
        )
        self.semantic_units = self.semantic_units[:self.config.max_units]

    def add_turn(self, user_message: str, assistant_message: str):
        """Add a conversation turn."""
        self.recent_messages.append({"role": "user", "content": user_message})
        self.recent_messages.append({"role": "assistant", "content": assistant_message})
        self.turn_count += 1

        # Extract semantic units periodically
        if self.turn_count % self.config.extraction_interval == 0:
            keep_count = self.config.recent_turns_to_keep * 2
            messages_to_process = self.recent_messages[:-keep_count] if len(self.recent_messages) > keep_count else []

            if messages_to_process:
                new_units = self._extract_semantic_units(messages_to_process)
                self.semantic_units.extend(new_units)
                self._prune_units()

                # Keep only recent messages
                self.recent_messages = self.recent_messages[-keep_count:]

    def get_context(self) -> str:
        """Get semantic context."""
        if not self.semantic_units:
            return ""

        # Group by type
        grouped = {}
        for unit in self.semantic_units:
            if unit.unit_type not in grouped:
                grouped[unit.unit_type] = []
            grouped[unit.unit_type].append(unit)

        parts = ["[Extracted Knowledge]"]
        for unit_type, units in grouped.items():
            parts.append(f"\n{unit_type.upper()}:")
            for u in units:
                importance = f"[{u.importance.value}]" if u.importance != ImportanceLevel.MEDIUM else ""
                parts.append(f"  - {u.content} {importance}")

        return "\n".join(parts)

    def get_messages(self) -> list[dict]:
        """Get messages for API call."""
        messages = []

        context = self.get_context()
        if context:
            messages.append({
                "role": "user",
                "content": f"{context}\n\nPlease use this knowledge context for our conversation."
            })
            messages.append({
                "role": "assistant",
                "content": "I have the context. How can I help you?"
            })

        messages.extend(self.recent_messages)
        return messages


# =============================================================================
# Strategy 5: Hybrid Compression
# =============================================================================

class MessageType(Enum):
    """Message classification types."""
    FACT = "fact"
    QUESTION = "question"
    INSTRUCTION = "instruction"
    CHITCHAT = "chitchat"
    FEEDBACK = "feedback"


@dataclass
class HybridConfig:
    """Configuration for hybrid compression."""
    compression_interval: int = 5
    max_recent_messages: int = 6  # 3 turns
    max_semantic_units: int = 20
    max_session_summary_tokens: int = 500
    max_facts: int = 10
    importance_threshold: float = 0.5


@dataclass
class HybridMemory:
    """Memory structure for hybrid compression."""
    recent: list = field(default_factory=list)  # Recent complete messages
    session_summary: str = ""  # Current session summary
    semantic_units: list = field(default_factory=list)  # Extracted semantic units
    long_term_facts: list = field(default_factory=list)  # Cross-session facts


class HybridCompressionManager:
    """
    Production-ready hybrid system combining all strategies:
    - Message classification
    - Importance-based retention
    - Progressive summarization
    - Semantic extraction
    - Long-term memory
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.memory = HybridMemory()
        self.turn_count: int = 0

    def _classify_message(self, message: str) -> tuple[MessageType, float]:
        """Classify message type and importance."""
        prompt = f"""Classify this message and rate its importance for future context.

Message: "{message}"

Return JSON with:
- type: One of "fact", "question", "instruction", "chitchat", "feedback"
- importance: Float 0-1 (1 = critical for future context)
- reason: Brief explanation

JSON response:"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            if "{" in text:
                json_str = text[text.find("{"):text.rfind("}") + 1]
                data = json.loads(json_str)
                msg_type = MessageType(data.get("type", "chitchat"))
                importance = float(data.get("importance", 0.5))
                return msg_type, importance
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return MessageType.CHITCHAT, 0.5

    def _update_session_summary(self, messages: list[dict]) -> str:
        """Update session summary with new messages."""
        messages_text = "\n".join(
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        )

        prompt = f"""Update the session summary with new conversation content.

Current summary:
{self.memory.session_summary if self.memory.session_summary else "No previous summary."}

New messages:
{messages_text}

Generate updated summary (keep it concise, ~200 words):"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=self.config.max_session_summary_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _extract_units(self, messages: list[dict]) -> list[dict]:
        """Extract semantic units from messages."""
        messages_text = "\n".join(
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        )

        prompt = f"""Extract key semantic units from this conversation.
Return JSON array with: content, type (fact|decision|preference|task), importance (0-1)

Conversation:
{messages_text}

JSON array:"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            if "[" in text:
                json_str = text[text.find("["):text.rfind("]") + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, KeyError):
            pass
        return []

    def _perform_compression(self):
        """Perform full compression cycle."""
        # Get messages to compress (exclude recent)
        if len(self.memory.recent) <= self.config.max_recent_messages:
            return

        messages_to_compress = self.memory.recent[:-self.config.max_recent_messages]

        # Update session summary
        self.memory.session_summary = self._update_session_summary(messages_to_compress)

        # Extract semantic units
        new_units = self._extract_units(messages_to_compress)
        for unit in new_units:
            if unit.get("importance", 0) >= self.config.importance_threshold:
                self.memory.semantic_units.append(unit)

        # Prune semantic units
        self.memory.semantic_units.sort(
            key=lambda u: u.get("importance", 0), reverse=True
        )
        self.memory.semantic_units = self.memory.semantic_units[:self.config.max_semantic_units]

        # Keep only recent messages
        self.memory.recent = self.memory.recent[-self.config.max_recent_messages:]

    def add_turn(self, user_message: str, assistant_message: str):
        """Add a conversation turn with classification."""
        self.turn_count += 1

        # Classify user message
        msg_type, importance = self._classify_message(user_message)

        # Add to recent messages
        self.memory.recent.append({
            "role": "user",
            "content": user_message,
            "_type": msg_type.value,
            "_importance": importance
        })
        self.memory.recent.append({
            "role": "assistant",
            "content": assistant_message
        })

        # Check for long-term facts (high importance facts)
        if msg_type == MessageType.FACT and importance >= 0.8:
            if user_message not in self.memory.long_term_facts:
                self.memory.long_term_facts.append(user_message)
                if len(self.memory.long_term_facts) > self.config.max_facts:
                    self.memory.long_term_facts.pop(0)

        # Perform compression at intervals
        if self.turn_count % self.config.compression_interval == 0:
            self._perform_compression()

    def get_context(self) -> str:
        """Get full hybrid context."""
        parts = []

        # Long-term facts
        if self.memory.long_term_facts:
            parts.append("Long-term Facts:\n" + "\n".join(f"- {f}" for f in self.memory.long_term_facts))

        # Session summary
        if self.memory.session_summary:
            parts.append(f"Session Summary:\n{self.memory.session_summary}")

        # Semantic units
        if self.memory.semantic_units:
            units_text = "Key Information:\n"
            for u in self.memory.semantic_units:
                units_text += f"- [{u.get('type', 'info')}] {u.get('content', '')}\n"
            parts.append(units_text)

        return "\n\n".join(parts)

    def get_messages(self) -> list[dict]:
        """Get messages for API call."""
        messages = []

        context = self.get_context()
        if context:
            messages.append({
                "role": "user",
                "content": f"[Context]\n{context}\n\nContinue based on this context."
            })
            messages.append({
                "role": "assistant",
                "content": "I have the context. How can I help?"
            })

        # Add recent messages (strip internal metadata)
        for msg in self.memory.recent:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages

    def get_stats(self) -> dict:
        """Get compression statistics."""
        return {
            "turn_count": self.turn_count,
            "recent_messages": len(self.memory.recent),
            "semantic_units": len(self.memory.semantic_units),
            "long_term_facts": len(self.memory.long_term_facts),
            "has_session_summary": bool(self.memory.session_summary)
        }


# =============================================================================
# Example Usage
# =============================================================================

def demo_sliding_window():
    """Demo sliding window manager."""
    print("\n=== Sliding Window Demo ===")
    manager = SlidingWindowManager(SlidingWindowConfig(max_turns=3))

    for i in range(5):
        manager.add_message("user", f"This is user message {i+1}")
        manager.add_message("assistant", f"This is assistant response {i+1}")

    print(f"Stats: {manager.get_stats()}")
    print(f"Messages kept: {len(manager.get_messages())}")


def demo_progressive_summarization():
    """Demo progressive summarization."""
    print("\n=== Progressive Summarization Demo ===")
    summarizer = ProgressiveSummarizer(ProgressiveSummaryConfig(
        summarize_threshold=3,
        recent_turns_to_keep=2
    ))

    # Simulate conversation
    turns = [
        ("What is Python?", "Python is a high-level programming language."),
        ("What are its main features?", "Python features include readability, dynamic typing, and extensive libraries."),
        ("Can you show me a hello world?", "print('Hello, World!')"),
        ("How about a loop?", "for i in range(5): print(i)"),
    ]

    for user, assistant in turns:
        summarizer.add_turn(user, assistant)
        print(f"Turn added. Summary exists: {bool(summarizer.state.summary)}")

    print(f"\nContext:\n{summarizer.get_context()}")


def demo_hybrid_compression():
    """Demo hybrid compression system."""
    print("\n=== Hybrid Compression Demo ===")
    manager = HybridCompressionManager(HybridConfig(compression_interval=2))

    turns = [
        ("My name is Alice and I'm a data scientist", "Nice to meet you, Alice! How can I help with your data science work?"),
        ("I'm working on a machine learning project", "That sounds interesting! What kind of ML project?"),
        ("It's a classification task for customer churn", "Customer churn prediction is valuable. What data do you have?"),
    ]

    for user, assistant in turns:
        manager.add_turn(user, assistant)

    print(f"Stats: {manager.get_stats()}")
    print(f"\nContext:\n{manager.get_context()}")


if __name__ == "__main__":
    print("Context Compression & Summarization Strategies")
    print("=" * 50)

    # Sliding window demo (no API required)
    demo_sliding_window()

    # LLM-based demos (require ANTHROPIC_API_KEY)
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            demo_progressive_summarization()
            demo_hybrid_compression()
        except (anthropic.AuthenticationError, anthropic.APIConnectionError) as e:
            print(f"\nAPI Error: {e}")
            print("Check your ANTHROPIC_API_KEY configuration.")
    else:
        print("\nNote: Set ANTHROPIC_API_KEY environment variable to run LLM-based demos.")
        print("(Progressive Summarization, Hybrid Compression)")
