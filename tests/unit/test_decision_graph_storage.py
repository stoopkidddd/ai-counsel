"""Unit tests for decision graph storage layer."""
import sqlite3
from datetime import datetime

import pytest

from decision_graph.schema import (DecisionNode, DecisionSimilarity,
                                   ParticipantStance)
from decision_graph.storage import DecisionGraphStorage


@pytest.fixture
def storage():
    """Provide in-memory storage for testing."""
    storage = DecisionGraphStorage(db_path=":memory:")
    yield storage
    storage.close()


@pytest.fixture
def sample_decision_node():
    """Create a sample DecisionNode for testing."""
    return DecisionNode(
        question="Should we adopt TypeScript?",
        timestamp=datetime.now(),
        consensus="Yes, with gradual migration",
        convergence_status="converged",
        participants=["opus@claude", "gpt-4@codex"],
        transcript_path="transcripts/typescript_decision.md",
    )


@pytest.fixture
def sample_participant_stance(sample_decision_node):
    """Create a sample ParticipantStance for testing."""
    return ParticipantStance(
        decision_id=sample_decision_node.id,
        participant="opus@claude",
        vote_option="Gradual Migration",
        confidence=0.85,
        rationale="Minimizes disruption while gaining benefits",
        final_position="I recommend gradual TypeScript adoption",
    )


class TestDecisionGraphStorageInitialization:
    """Tests for storage initialization."""

    def test_storage_initializes_with_memory_db(self):
        """Test that storage can initialize with in-memory database."""
        storage = DecisionGraphStorage(db_path=":memory:")
        assert storage is not None
        assert storage.db_path == ":memory:"
        storage.close()

    def test_storage_initializes_with_file_path(self, tmp_path):
        """Test that storage can initialize with file path."""
        db_file = tmp_path / "test.db"
        storage = DecisionGraphStorage(db_path=str(db_file))
        assert storage is not None
        assert storage.db_path == str(db_file)
        storage.close()

    def test_storage_creates_parent_directory_if_missing(self, tmp_path):
        """Test that storage automatically creates parent directories for the database file.

        This test ensures that first-time users don't get "readonly database" errors
        when the parent directory doesn't exist. Regression test for GitHub issue
        where deleted database files couldn't be recreated.
        """
        # Create a nested path where intermediate directories don't exist
        db_file = tmp_path / "data" / "subdir" / "test.db"
        parent_dir = db_file.parent

        # Verify parent directory doesn't exist yet
        assert not parent_dir.exists()

        # Initialize storage - should create parent directories
        storage = DecisionGraphStorage(db_path=str(db_file))

        # Verify parent directory was created
        assert parent_dir.exists()
        assert parent_dir.is_dir()

        # Verify database file was created and is functional
        assert db_file.exists()
        assert db_file.is_file()

        # Verify we can actually write to the database
        cursor = storage.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        assert len(tables) > 0  # Should have created tables

        storage.close()

    def test_storage_creates_tables_on_init(self, storage):
        """Test that tables are created during initialization."""
        cursor = storage.conn.cursor()

        # Check decision_nodes table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='decision_nodes'
        """
        )
        assert cursor.fetchone() is not None

        # Check participant_stances table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='participant_stances'
        """
        )
        assert cursor.fetchone() is not None

        # Check decision_similarities table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='decision_similarities'
        """
        )
        assert cursor.fetchone() is not None

    def test_storage_creates_indexes(self, storage):
        """Test that indexes are created during initialization."""
        cursor = storage.conn.cursor()

        # Get all indexes
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='index'
        """
        )
        indexes = [row[0] for row in cursor.fetchall()]

        # Check expected indexes exist
        assert "idx_decision_timestamp" in indexes
        assert "idx_participant_decision" in indexes
        assert "idx_similarity_source" in indexes
        assert "idx_similarity_score" in indexes

    def test_storage_enables_foreign_keys(self, storage):
        """Test that foreign key constraints are enabled."""
        cursor = storage.conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        assert result[0] == 1  # 1 means enabled

    def test_storage_context_manager(self):
        """Test storage works with context manager."""
        with DecisionGraphStorage(db_path=":memory:") as storage:
            assert storage is not None
            assert storage._conn is not None
        # Connection should be closed after context
        assert storage._conn is None


class TestDecisionNodeCRUD:
    """Tests for DecisionNode CRUD operations."""

    def test_save_decision_node(self, storage, sample_decision_node):
        """Test saving a decision node."""
        saved_id = storage.save_decision_node(sample_decision_node)
        assert saved_id == sample_decision_node.id
        assert isinstance(saved_id, str)

    def test_save_decision_node_returns_id(self, storage):
        """Test that save returns the node ID."""
        node = DecisionNode(
            question="Test question",
            timestamp=datetime.now(),
            consensus="Test consensus",
            convergence_status="converged",
            participants=["p1"],
            transcript_path="/path",
        )
        result = storage.save_decision_node(node)
        assert result == node.id

    def test_get_decision_node_by_id(self, storage, sample_decision_node):
        """Test retrieving a decision node by ID."""
        storage.save_decision_node(sample_decision_node)

        retrieved = storage.get_decision_node(sample_decision_node.id)
        assert retrieved is not None
        assert retrieved.id == sample_decision_node.id
        assert retrieved.question == sample_decision_node.question
        assert retrieved.consensus == sample_decision_node.consensus
        assert retrieved.convergence_status == sample_decision_node.convergence_status
        assert retrieved.participants == sample_decision_node.participants

    def test_get_decision_node_missing_returns_none(self, storage):
        """Test that retrieving non-existent node returns None."""
        retrieved = storage.get_decision_node("nonexistent-id")
        assert retrieved is None

    def test_get_decision_node_preserves_timestamp(self, storage):
        """Test that timestamp is correctly preserved."""
        timestamp = datetime(2024, 10, 20, 15, 30, 45)
        node = DecisionNode(
            question="Q",
            timestamp=timestamp,
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        # Compare as strings since we serialize to ISO format
        assert retrieved.timestamp.replace(microsecond=0) == timestamp.replace(
            microsecond=0
        )

    def test_get_decision_node_preserves_participants_list(self, storage):
        """Test that participants list is correctly preserved."""
        node = DecisionNode(
            question="Q",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=["model-a", "model-b", "model-c"],
            transcript_path="t",
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        assert retrieved.participants == ["model-a", "model-b", "model-c"]
        assert isinstance(retrieved.participants, list)

    def test_get_decision_node_preserves_winning_option(self, storage):
        """Test that optional winning_option is preserved."""
        node = DecisionNode(
            question="Q",
            timestamp=datetime.now(),
            consensus="C",
            winning_option="Option A",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        assert retrieved.winning_option == "Option A"

    def test_get_decision_node_preserves_metadata(self, storage):
        """Test that metadata dict is preserved."""
        metadata = {"custom": "value", "rounds": 3}
        node = DecisionNode(
            question="Q",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
            metadata=metadata,
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        assert retrieved.metadata == metadata
        assert retrieved.metadata["custom"] == "value"

    def test_save_duplicate_id_raises_error(self, storage, sample_decision_node):
        """Test that saving node with duplicate ID raises IntegrityError."""
        storage.save_decision_node(sample_decision_node)

        with pytest.raises(sqlite3.IntegrityError):
            storage.save_decision_node(sample_decision_node)

    def test_get_all_decisions_empty(self, storage):
        """Test getting all decisions when database is empty."""
        decisions = storage.get_all_decisions()
        assert decisions == []

    def test_get_all_decisions_returns_all(self, storage):
        """Test getting all decisions."""
        # Create and save multiple decisions
        nodes = []
        for i in range(5):
            node = DecisionNode(
                question=f"Question {i}",
                timestamp=datetime.now(),
                consensus=f"Consensus {i}",
                convergence_status="converged",
                participants=[f"p{i}"],
                transcript_path=f"t{i}",
            )
            storage.save_decision_node(node)
            nodes.append(node)

        all_decisions = storage.get_all_decisions(limit=10)
        assert len(all_decisions) == 5

    def test_get_all_decisions_ordered_by_timestamp(self, storage):
        """Test that decisions are ordered by timestamp (newest first)."""
        # Create nodes with different timestamps
        node1 = DecisionNode(
            question="First",
            timestamp=datetime(2024, 10, 20, 10, 0, 0),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        node2 = DecisionNode(
            question="Second",
            timestamp=datetime(2024, 10, 20, 11, 0, 0),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        node3 = DecisionNode(
            question="Third",
            timestamp=datetime(2024, 10, 20, 12, 0, 0),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )

        # Save in random order
        storage.save_decision_node(node2)
        storage.save_decision_node(node1)
        storage.save_decision_node(node3)

        # Retrieve
        decisions = storage.get_all_decisions()
        assert len(decisions) == 3
        assert decisions[0].question == "Third"  # Newest first
        assert decisions[1].question == "Second"
        assert decisions[2].question == "First"

    def test_get_all_decisions_respects_limit(self, storage):
        """Test that limit parameter works correctly."""
        # Create 10 decisions
        for i in range(10):
            node = DecisionNode(
                question=f"Q{i}",
                timestamp=datetime.now(),
                consensus="C",
                convergence_status="converged",
                participants=[],
                transcript_path="t",
            )
            storage.save_decision_node(node)

        # Request with limit
        decisions = storage.get_all_decisions(limit=5)
        assert len(decisions) == 5

    def test_get_all_decisions_respects_offset(self, storage):
        """Test that offset parameter works for pagination."""
        # Create 10 decisions with distinct questions
        for i in range(10):
            node = DecisionNode(
                question=f"Question {i:02d}",
                timestamp=datetime(2024, 10, 20, 10, i, 0),
                consensus="C",
                convergence_status="converged",
                participants=[],
                transcript_path="t",
            )
            storage.save_decision_node(node)

        # Get first page
        page1 = storage.get_all_decisions(limit=5, offset=0)
        assert len(page1) == 5

        # Get second page
        page2 = storage.get_all_decisions(limit=5, offset=5)
        assert len(page2) == 5

        # Ensure no overlap
        page1_ids = {d.id for d in page1}
        page2_ids = {d.id for d in page2}
        assert page1_ids.isdisjoint(page2_ids)


class TestParticipantStanceCRUD:
    """Tests for ParticipantStance CRUD operations."""

    def test_save_participant_stance(
        self, storage, sample_decision_node, sample_participant_stance
    ):
        """Test saving a participant stance."""
        storage.save_decision_node(sample_decision_node)
        row_id = storage.save_participant_stance(sample_participant_stance)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_save_stance_requires_valid_decision_id(self, storage):
        """Test that saving stance with invalid decision_id raises error."""
        stance = ParticipantStance(
            decision_id="nonexistent-decision-id",
            participant="p",
            final_position="pos",
        )

        with pytest.raises(sqlite3.IntegrityError):
            storage.save_participant_stance(stance)

    def test_get_participant_stances_empty(self, storage, sample_decision_node):
        """Test getting stances when none exist."""
        storage.save_decision_node(sample_decision_node)
        stances = storage.get_participant_stances(sample_decision_node.id)
        assert stances == []

    def test_get_participant_stances(self, storage, sample_decision_node):
        """Test retrieving stances for a decision."""
        storage.save_decision_node(sample_decision_node)

        # Create multiple stances
        stance1 = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="opus@claude",
            final_position="Position 1",
        )
        stance2 = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="gpt-4@codex",
            final_position="Position 2",
        )

        storage.save_participant_stance(stance1)
        storage.save_participant_stance(stance2)

        stances = storage.get_participant_stances(sample_decision_node.id)
        assert len(stances) == 2
        participants = [s.participant for s in stances]
        assert "opus@claude" in participants
        assert "gpt-4@codex" in participants

    def test_get_participant_stances_preserves_vote_data(
        self, storage, sample_decision_node
    ):
        """Test that vote-related fields are preserved."""
        storage.save_decision_node(sample_decision_node)

        stance = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="opus@claude",
            vote_option="Option A",
            confidence=0.85,
            rationale="Best option for scalability",
            final_position="I recommend Option A",
        )
        storage.save_participant_stance(stance)

        retrieved = storage.get_participant_stances(sample_decision_node.id)
        assert len(retrieved) == 1
        assert retrieved[0].vote_option == "Option A"
        assert retrieved[0].confidence == 0.85
        assert retrieved[0].rationale == "Best option for scalability"
        assert retrieved[0].final_position == "I recommend Option A"

    def test_get_participant_stances_handles_none_values(
        self, storage, sample_decision_node
    ):
        """Test that None values in optional fields are handled correctly."""
        storage.save_decision_node(sample_decision_node)

        stance = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="p",
            final_position="pos",
            # vote_option, confidence, rationale all None
        )
        storage.save_participant_stance(stance)

        retrieved = storage.get_participant_stances(sample_decision_node.id)
        assert len(retrieved) == 1
        assert retrieved[0].vote_option is None
        assert retrieved[0].confidence is None
        assert retrieved[0].rationale is None

    def test_get_participant_stances_ordered_by_participant(
        self, storage, sample_decision_node
    ):
        """Test that stances are ordered by participant name."""
        storage.save_decision_node(sample_decision_node)

        # Save in non-alphabetical order
        stance_z = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="z-model",
            final_position="pos",
        )
        stance_a = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="a-model",
            final_position="pos",
        )
        stance_m = ParticipantStance(
            decision_id=sample_decision_node.id,
            participant="m-model",
            final_position="pos",
        )

        storage.save_participant_stance(stance_z)
        storage.save_participant_stance(stance_a)
        storage.save_participant_stance(stance_m)

        retrieved = storage.get_participant_stances(sample_decision_node.id)
        assert len(retrieved) == 3
        assert retrieved[0].participant == "a-model"
        assert retrieved[1].participant == "m-model"
        assert retrieved[2].participant == "z-model"

    def test_multiple_stances_per_decision(self, storage):
        """Test saving multiple stances for the same decision."""
        node = DecisionNode(
            question="Q",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=["p1", "p2", "p3"],
            transcript_path="t",
        )
        storage.save_decision_node(node)

        # Save stances for 3 participants
        for i in range(3):
            stance = ParticipantStance(
                decision_id=node.id,
                participant=f"participant-{i}",
                final_position=f"Position {i}",
            )
            storage.save_participant_stance(stance)

        retrieved = storage.get_participant_stances(node.id)
        assert len(retrieved) == 3


class TestDecisionSimilarityCRUD:
    """Tests for DecisionSimilarity CRUD operations."""

    def test_save_similarity(self, storage):
        """Test saving a similarity relationship."""
        # Create two decisions
        node1 = DecisionNode(
            question="Q1",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        node2 = DecisionNode(
            question="Q2",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        storage.save_decision_node(node1)
        storage.save_decision_node(node2)

        # Save similarity
        similarity = DecisionSimilarity(
            source_id=node1.id,
            target_id=node2.id,
            similarity_score=0.75,
        )
        storage.save_similarity(similarity)

        # Should not raise an error

    def test_save_similarity_requires_valid_source_id(
        self, storage, sample_decision_node
    ):
        """Test that invalid source_id raises error."""
        storage.save_decision_node(sample_decision_node)

        similarity = DecisionSimilarity(
            source_id="nonexistent-id",
            target_id=sample_decision_node.id,
            similarity_score=0.5,
        )

        with pytest.raises(sqlite3.IntegrityError):
            storage.save_similarity(similarity)

    def test_save_similarity_requires_valid_target_id(
        self, storage, sample_decision_node
    ):
        """Test that invalid target_id raises error."""
        storage.save_decision_node(sample_decision_node)

        similarity = DecisionSimilarity(
            source_id=sample_decision_node.id,
            target_id="nonexistent-id",
            similarity_score=0.5,
        )

        with pytest.raises(sqlite3.IntegrityError):
            storage.save_similarity(similarity)

    def test_save_similarity_upsert_behavior(self, storage):
        """Test that saving similarity with same IDs updates the score."""
        # Create two decisions
        node1 = DecisionNode(
            question="Q1",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        node2 = DecisionNode(
            question="Q2",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        storage.save_decision_node(node1)
        storage.save_decision_node(node2)

        # Save initial similarity
        sim1 = DecisionSimilarity(
            source_id=node1.id,
            target_id=node2.id,
            similarity_score=0.5,
        )
        storage.save_similarity(sim1)

        # Update with new score
        sim2 = DecisionSimilarity(
            source_id=node1.id,
            target_id=node2.id,
            similarity_score=0.9,
        )
        storage.save_similarity(sim2)

        # Retrieve and verify updated score
        similar = storage.get_similar_decisions(node1.id, threshold=0.0)
        assert len(similar) == 1
        assert similar[0][1] == 0.9  # Updated score

    def test_get_similar_decisions_empty(self, storage, sample_decision_node):
        """Test getting similar decisions when none exist."""
        storage.save_decision_node(sample_decision_node)
        similar = storage.get_similar_decisions(sample_decision_node.id)
        assert similar == []

    def test_get_similar_decisions_returns_matches(self, storage):
        """Test retrieving similar decisions."""
        # Create three decisions
        nodes = []
        for i in range(3):
            node = DecisionNode(
                question=f"Q{i}",
                timestamp=datetime.now(),
                consensus="C",
                convergence_status="converged",
                participants=[],
                transcript_path="t",
            )
            storage.save_decision_node(node)
            nodes.append(node)

        # Create similarities from node0 to others
        sim1 = DecisionSimilarity(
            source_id=nodes[0].id,
            target_id=nodes[1].id,
            similarity_score=0.9,
        )
        sim2 = DecisionSimilarity(
            source_id=nodes[0].id,
            target_id=nodes[2].id,
            similarity_score=0.8,
        )
        storage.save_similarity(sim1)
        storage.save_similarity(sim2)

        # Retrieve
        similar = storage.get_similar_decisions(nodes[0].id, threshold=0.7)
        assert len(similar) == 2

    def test_get_similar_decisions_respects_threshold(self, storage):
        """Test that threshold filters results correctly."""
        # Create nodes
        nodes = []
        for i in range(4):
            node = DecisionNode(
                question=f"Q{i}",
                timestamp=datetime.now(),
                consensus="C",
                convergence_status="converged",
                participants=[],
                transcript_path="t",
            )
            storage.save_decision_node(node)
            nodes.append(node)

        # Create similarities with different scores
        storage.save_similarity(
            DecisionSimilarity(
                source_id=nodes[0].id, target_id=nodes[1].id, similarity_score=0.9
            )
        )
        storage.save_similarity(
            DecisionSimilarity(
                source_id=nodes[0].id, target_id=nodes[2].id, similarity_score=0.5
            )
        )
        storage.save_similarity(
            DecisionSimilarity(
                source_id=nodes[0].id, target_id=nodes[3].id, similarity_score=0.3
            )
        )

        # High threshold
        similar_high = storage.get_similar_decisions(nodes[0].id, threshold=0.8)
        assert len(similar_high) == 1
        assert similar_high[0][1] == 0.9

        # Low threshold
        similar_low = storage.get_similar_decisions(nodes[0].id, threshold=0.2)
        assert len(similar_low) == 3

    def test_get_similar_decisions_ordered_by_score(self, storage):
        """Test that results are ordered by similarity score (highest first)."""
        # Create nodes
        nodes = []
        for i in range(4):
            node = DecisionNode(
                question=f"Q{i}",
                timestamp=datetime.now(),
                consensus="C",
                convergence_status="converged",
                participants=[],
                transcript_path="t",
            )
            storage.save_decision_node(node)
            nodes.append(node)

        # Save in non-sorted order
        storage.save_similarity(
            DecisionSimilarity(
                source_id=nodes[0].id, target_id=nodes[1].id, similarity_score=0.5
            )
        )
        storage.save_similarity(
            DecisionSimilarity(
                source_id=nodes[0].id, target_id=nodes[2].id, similarity_score=0.9
            )
        )
        storage.save_similarity(
            DecisionSimilarity(
                source_id=nodes[0].id, target_id=nodes[3].id, similarity_score=0.7
            )
        )

        similar = storage.get_similar_decisions(nodes[0].id, threshold=0.0)
        assert len(similar) == 3
        assert similar[0][1] == 0.9  # Highest
        assert similar[1][1] == 0.7  # Middle
        assert similar[2][1] == 0.5  # Lowest

    def test_get_similar_decisions_respects_limit(self, storage):
        """Test that limit parameter works correctly."""
        # Create nodes
        nodes = []
        for i in range(6):
            node = DecisionNode(
                question=f"Q{i}",
                timestamp=datetime.now(),
                consensus="C",
                convergence_status="converged",
                participants=[],
                transcript_path="t",
            )
            storage.save_decision_node(node)
            nodes.append(node)

        # Create 5 similarities
        for i in range(1, 6):
            storage.save_similarity(
                DecisionSimilarity(
                    source_id=nodes[0].id,
                    target_id=nodes[i].id,
                    similarity_score=0.8,
                )
            )

        # Request with limit
        similar = storage.get_similar_decisions(nodes[0].id, threshold=0.7, limit=3)
        assert len(similar) == 3

    def test_get_similar_decisions_returns_decision_nodes(self, storage):
        """Test that results include full DecisionNode objects."""
        # Create nodes
        node1 = DecisionNode(
            question="Original question",
            timestamp=datetime.now(),
            consensus="C1",
            convergence_status="converged",
            participants=["p1"],
            transcript_path="t1",
        )
        node2 = DecisionNode(
            question="Similar question",
            timestamp=datetime.now(),
            consensus="C2",
            convergence_status="refining",
            participants=["p2"],
            transcript_path="t2",
        )
        storage.save_decision_node(node1)
        storage.save_decision_node(node2)

        # Create similarity
        storage.save_similarity(
            DecisionSimilarity(
                source_id=node1.id,
                target_id=node2.id,
                similarity_score=0.85,
            )
        )

        # Retrieve
        similar = storage.get_similar_decisions(node1.id, threshold=0.8)
        assert len(similar) == 1

        retrieved_node, score = similar[0]
        assert isinstance(retrieved_node, DecisionNode)
        assert retrieved_node.question == "Similar question"
        assert retrieved_node.consensus == "C2"
        assert score == 0.85


class TestStorageEdgeCases:
    """Tests for edge cases and error handling."""

    def test_close_connection(self, storage):
        """Test that close() closes the connection."""
        assert storage._conn is not None
        storage.close()
        assert storage._conn is None

    def test_close_idempotent(self, storage):
        """Test that calling close() multiple times is safe."""
        storage.close()
        storage.close()  # Should not raise error
        assert storage._conn is None

    def test_empty_participants_list(self, storage):
        """Test saving node with empty participants list."""
        node = DecisionNode(
            question="Q",
            timestamp=datetime.now(),
            consensus="C",
            convergence_status="converged",
            participants=[],
            transcript_path="t",
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        assert retrieved.participants == []

    def test_long_text_fields(self, storage):
        """Test handling of very long text fields."""
        long_text = "A" * 10000
        node = DecisionNode(
            question=long_text,
            timestamp=datetime.now(),
            consensus=long_text,
            convergence_status="converged",
            participants=[],
            transcript_path=long_text,
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        assert len(retrieved.question) == 10000
        assert len(retrieved.consensus) == 10000

    def test_unicode_text_fields(self, storage):
        """Test handling of unicode characters."""
        node = DecisionNode(
            question="Should we use 日本語 in our app?",
            timestamp=datetime.now(),
            consensus="Yes, with UTF-8 encoding 🎉",
            convergence_status="converged",
            participants=["模型-A", "模型-B"],
            transcript_path="/transcripts/unicode_test.md",
        )
        storage.save_decision_node(node)

        retrieved = storage.get_decision_node(node.id)
        assert "日本語" in retrieved.question
        assert "🎉" in retrieved.consensus
        assert "模型-A" in retrieved.participants

    def test_transaction_rollback_on_error(self, storage):
        """Test that transaction rolls back on error."""
        # This should fail due to missing required field
        try:
            with storage.transaction() as conn:
                conn.execute("INSERT INTO decision_nodes (id) VALUES (?)", ("test-id",))
        except sqlite3.IntegrityError:
            pass  # Expected

        # Verify nothing was saved
        retrieved = storage.get_decision_node("test-id")
        assert retrieved is None


class TestThreadingMigration:
    """Tests for the conversation-threading schema migration (Plan B / B2)."""

    def test_migration_adds_thread_columns_idempotently(self, tmp_path):
        """Initialising twice over the same file is safe and produces the threading columns."""
        db_path = str(tmp_path / "graph.db")
        s1 = DecisionGraphStorage(db_path)
        cols = {row[1] for row in s1.conn.execute("PRAGMA table_info(decision_nodes)").fetchall()}
        assert "thread_id" in cols
        assert "parent_decision_id" in cols
        s1.close()

        # Re-open without errors
        s2 = DecisionGraphStorage(db_path)
        cols2 = {row[1] for row in s2.conn.execute("PRAGMA table_info(decision_nodes)").fetchall()}
        assert cols == cols2

    def test_re_init_does_not_duplicate_schema_version_rows(self, tmp_path):
        """Opening a fully-migrated DB twice doesn't insert duplicate version rows."""
        db_path = str(tmp_path / "graph.db")
        s1 = DecisionGraphStorage(db_path)
        rows1 = s1.conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        s1.close()

        s2 = DecisionGraphStorage(db_path)
        rows2 = s2.conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        assert rows1 == rows2

    def test_migration_upgrades_legacy_db(self, tmp_path):
        """A pre-existing DB without threading columns gets ALTER TABLE'd in place."""
        db_path = str(tmp_path / "legacy.db")
        # Create a database with the OLD schema (no thread columns)
        legacy = sqlite3.connect(db_path)
        legacy.execute(
            """
            CREATE TABLE decision_nodes (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                consensus TEXT NOT NULL,
                winning_option TEXT,
                convergence_status TEXT NOT NULL,
                participants TEXT NOT NULL,
                transcript_path TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        legacy.execute(
            "INSERT INTO decision_nodes VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "legacy-1",
                "old question",
                "2026-01-01T00:00:00",
                "old consensus",
                None,
                "converged",
                '["x@y"]',
                "/t",
                None,
            ),
        )
        legacy.commit()
        legacy.close()

        # Storage migration must add the columns
        storage = DecisionGraphStorage(db_path)
        cols = {row[1] for row in storage.conn.execute("PRAGMA table_info(decision_nodes)").fetchall()}
        assert "thread_id" in cols
        assert "parent_decision_id" in cols

        # Legacy data is preserved with NULL threading values
        legacy_node = storage.get_decision_node("legacy-1")
        assert legacy_node is not None
        assert legacy_node.thread_id is None
        assert legacy_node.parent_decision_id is None
        assert legacy_node.question == "old question"

    def test_thread_indexes_created(self, storage):
        """Threading indexes exist on decision_nodes."""
        cursor = storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='decision_nodes'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        assert "idx_decision_thread" in index_names
        assert "idx_decision_parent" in index_names

    def test_set_thread_id_backfills_existing_decision(self, storage):
        """set_thread_id updates an existing decision in place."""
        node = DecisionNode(
            id="root-1",
            question="q",
            timestamp=datetime.now(),
            consensus="c",
            convergence_status="converged",
            participants=["x@y"],
            transcript_path="/t",
        )
        storage.save_decision_node(node)
        assert storage.get_decision_node("root-1").thread_id is None

        storage.set_thread_id("root-1", "thr-xyz")
        assert storage.get_decision_node("root-1").thread_id == "thr-xyz"

    def test_get_decisions_by_thread_returns_oldest_first(self, storage):
        """get_decisions_by_thread orders by timestamp ascending."""
        from datetime import timedelta

        base = datetime(2026, 1, 1, 0, 0, 0)
        for i, ts_offset in enumerate([2, 0, 1]):  # save out of order
            node = DecisionNode(
                id=f"d{i}",
                question=f"q{i}",
                timestamp=base + timedelta(minutes=ts_offset),
                consensus="c",
                convergence_status="converged",
                participants=["x@y"],
                transcript_path="/t",
                thread_id="thr-A",
            )
            storage.save_decision_node(node)

        chain = storage.get_decisions_by_thread("thr-A")
        # Sorted ascending by timestamp: ts=0 (d1), ts=1 (d2), ts=2 (d0)
        assert [d.id for d in chain] == ["d1", "d2", "d0"]


class TestSchemaVersioning:
    """Tests for the versioned migration framework (schema_version table)."""

    def test_fresh_db_records_all_migration_versions(self, storage):
        """A new in-memory DB lands at CURRENT_SCHEMA_VERSION with all steps recorded."""
        from decision_graph.storage import CURRENT_SCHEMA_VERSION

        rows = storage.conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        applied = [r[0] for r in rows]
        assert applied == list(range(1, CURRENT_SCHEMA_VERSION + 1))

    def test_schema_version_table_has_applied_at_timestamps(self, storage):
        """Each recorded version has a non-null ISO timestamp."""
        from datetime import datetime as dt

        rows = storage.conn.execute(
            "SELECT version, applied_at FROM schema_version"
        ).fetchall()
        assert rows
        for version, applied_at in rows:
            assert applied_at is not None
            # Must round-trip through fromisoformat
            dt.fromisoformat(applied_at)

    def test_legacy_v01_db_migrates_through_all_versions(self, tmp_path):
        """A pre-versioning DB (no schema_version, no threading) catches up cleanly."""
        from decision_graph.storage import CURRENT_SCHEMA_VERSION

        db_path = str(tmp_path / "legacy_v01.db")
        legacy = sqlite3.connect(db_path)
        legacy.execute(
            """
            CREATE TABLE decision_nodes (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, timestamp TEXT NOT NULL,
                consensus TEXT NOT NULL, winning_option TEXT, convergence_status TEXT NOT NULL,
                participants TEXT NOT NULL, transcript_path TEXT NOT NULL, metadata TEXT
            )
            """
        )
        legacy.commit()
        legacy.close()

        storage = DecisionGraphStorage(db_path)

        applied = [
            r[0] for r in storage.conn.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall()
        ]
        assert applied == list(range(1, CURRENT_SCHEMA_VERSION + 1))

        cols = {row[1] for row in storage.conn.execute("PRAGMA table_info(decision_nodes)").fetchall()}
        assert "thread_id" in cols
        assert "parent_decision_id" in cols

    def test_handrolled_v02_db_reconciles_to_current_version(self, tmp_path):
        """A DB with threading columns but no schema_version (upgraded by the
        original hand-rolled migration) is reconciled, not re-migrated."""
        db_path = str(tmp_path / "handrolled.db")
        legacy = sqlite3.connect(db_path)
        legacy.execute(
            """
            CREATE TABLE decision_nodes (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, timestamp TEXT NOT NULL,
                consensus TEXT NOT NULL, winning_option TEXT, convergence_status TEXT NOT NULL,
                participants TEXT NOT NULL, transcript_path TEXT NOT NULL, metadata TEXT,
                thread_id TEXT, parent_decision_id TEXT
            )
            """
        )
        legacy.commit()
        legacy.close()

        storage = DecisionGraphStorage(db_path)
        applied = [
            r[0] for r in storage.conn.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall()
        ]
        # Must include both versions, not error on duplicate ALTER TABLE
        assert applied == [1, 2]

    def test_db_newer_than_code_logs_warning_and_proceeds(self, tmp_path, caplog):
        """If schema_version exceeds code's CURRENT_SCHEMA_VERSION (downgrade
        scenario), the storage logs a warning and continues without erroring."""
        import logging

        from decision_graph.storage import CURRENT_SCHEMA_VERSION

        db_path = str(tmp_path / "future.db")
        # Build a DB that's "from the future" — schema_version says v3, but
        # code only knows up to CURRENT_SCHEMA_VERSION (today: 2).
        future = sqlite3.connect(db_path)
        future.execute(
            """
            CREATE TABLE decision_nodes (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, timestamp TEXT NOT NULL,
                consensus TEXT NOT NULL, winning_option TEXT, convergence_status TEXT NOT NULL,
                participants TEXT NOT NULL, transcript_path TEXT NOT NULL, metadata TEXT,
                thread_id TEXT, parent_decision_id TEXT
            )
            """
        )
        future.execute(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
            )
            """
        )
        future_v = CURRENT_SCHEMA_VERSION + 1
        future.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, '2099-01-01')",
            (future_v,),
        )
        future.commit()
        future.close()

        with caplog.at_level(logging.WARNING):
            storage = DecisionGraphStorage(db_path)

        assert any(
            "newer than this code supports" in record.message
            for record in caplog.records
        )
        # Existing version row preserved; nothing forcibly added
        rows = storage.conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        assert future_v in {r[0] for r in rows}
