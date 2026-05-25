"""FaceStore 단위 테스트 (in-memory SQLite 사용)."""
import numpy as np
import pytest
from modules.face_store import FaceStore


@pytest.fixture
def store(tmp_path):
    s = FaceStore(db_path=tmp_path / "test_faces.db")
    yield s
    s.close()


def _dummy_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(128)
    return v / np.linalg.norm(v)


class TestStudentManagement:
    def test_register_and_list(self, store):
        store.register_student("s001", "홍길동", consent=True)
        students = store.list_students()
        assert len(students) == 1
        assert students[0]["student_id"] == "s001"
        assert students[0]["consent"] is True

    def test_register_idempotent(self, store):
        store.register_student("s001", "홍길동")
        store.register_student("s001", "홍길동")
        assert len(store.list_students()) == 1

    def test_default_consent_false(self, store):
        store.register_student("s002", "이순신")
        assert store.has_consent("s002") is False

    def test_set_consent(self, store):
        store.register_student("s003", "강감찬", consent=False)
        store.set_consent("s003", True)
        assert store.has_consent("s003") is True

    def test_has_consent_unknown_student(self, store):
        assert store.has_consent("nonexistent") is False


class TestEmbeddingStorage:
    def test_add_and_get_embedding(self, store):
        store.register_student("s001", "홍길동", consent=True)
        emb = _dummy_emb(1)
        store.add_embedding("s001", emb)
        retrieved = store.get_embeddings("s001")
        assert len(retrieved) == 1
        np.testing.assert_allclose(retrieved[0], emb)

    def test_add_embedding_without_consent_raises(self, store):
        store.register_student("s002", "이순신", consent=False)
        with pytest.raises(PermissionError):
            store.add_embedding("s002", _dummy_emb())

    def test_multiple_embeddings_per_student(self, store):
        store.register_student("s001", "홍길동", consent=True)
        for i in range(3):
            store.add_embedding("s001", _dummy_emb(i))
        assert len(store.get_embeddings("s001")) == 3

    def test_get_all_embeddings_excludes_no_consent(self, store):
        store.register_student("s001", "동의함", consent=True)
        store.register_student("s002", "미동의", consent=False)
        store.add_embedding("s001", _dummy_emb(1))
        # s002는 consent False → add_embedding 자체가 PermissionError이므로
        # get_all_embeddings는 s001만 반환해야 함
        all_embs = store.get_all_embeddings()
        assert "s001" in all_embs
        assert "s002" not in all_embs

    def test_delete_student_embeddings(self, store):
        store.register_student("s001", "홍길동", consent=True)
        store.add_embedding("s001", _dummy_emb(1))
        store.delete_student_embeddings("s001")
        assert store.get_embeddings("s001") == []


class TestSimilarity:
    def test_same_embedding_similarity_is_one(self):
        emb = _dummy_emb(42)
        assert FaceStore.similarity(emb, emb) == pytest.approx(1.0)

    def test_different_embeddings_similarity_less_than_one(self):
        a, b = _dummy_emb(1), _dummy_emb(2)
        assert FaceStore.similarity(a, b) < 1.0

    def test_zero_vector_returns_zero(self):
        zero = np.zeros(128)
        assert FaceStore.similarity(zero, _dummy_emb()) == 0.0


class TestBestMatch:
    def test_best_match_returns_correct_student(self, store):
        store.register_student("s001", "홍길동", consent=True)
        store.register_student("s002", "이순신", consent=True)
        emb1, emb2 = _dummy_emb(10), _dummy_emb(20)
        store.add_embedding("s001", emb1)
        store.add_embedding("s002", emb2)
        assert store.best_match(emb1, threshold=0.5) == "s001"
        assert store.best_match(emb2, threshold=0.5) == "s002"

    def test_best_match_below_threshold_returns_none(self, store):
        store.register_student("s001", "홍길동", consent=True)
        store.add_embedding("s001", _dummy_emb(10))
        query = _dummy_emb(99)
        result = store.best_match(query, threshold=0.99)
        assert result is None

    def test_best_match_empty_store_returns_none(self, store):
        assert store.best_match(_dummy_emb()) is None
