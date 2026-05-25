"""
얼굴 임베딩 저장소.

- 얼굴 원본 이미지는 절대 저장하지 않는다.
- 128차원 임베딩 벡터만 로컬 SQLite에 보관한다.
- consent=True 인 학생에 한해 임베딩이 활성화된다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import face_recognition


_DB_VERSION = 1


class FaceStore:
    """학생 얼굴 임베딩을 SQLite에 저장·조회한다."""

    def __init__(self, db_path: str | Path = "data/sessions/faces.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id  TEXT PRIMARY KEY,
                name        TEXT,
                consent     INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL REFERENCES students(student_id),
                embedding   BLOB NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emb_student
            ON embeddings(student_id)
        """)
        self._conn.commit()

    # ── 학생 관리 ──────────────────────────────────────────────────────────

    def register_student(self, student_id: str, name: str, consent: bool = False):
        self._conn.execute(
            "INSERT OR IGNORE INTO students (student_id, name, consent) VALUES (?, ?, ?)",
            (student_id, name, int(consent)),
        )
        self._conn.commit()

    def set_consent(self, student_id: str, consent: bool):
        self._conn.execute(
            "UPDATE students SET consent = ? WHERE student_id = ?",
            (int(consent), student_id),
        )
        self._conn.commit()

    def has_consent(self, student_id: str) -> bool:
        row = self._conn.execute(
            "SELECT consent FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return bool(row and row[0])

    def list_students(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT student_id, name, consent FROM students ORDER BY student_id"
        ).fetchall()
        return [{"student_id": r[0], "name": r[1], "consent": bool(r[2])} for r in rows]

    # ── 임베딩 관리 ────────────────────────────────────────────────────────

    def add_embedding(self, student_id: str, embedding: np.ndarray):
        """128차원 float64 벡터를 저장한다. consent 미확인 시 예외."""
        if not self.has_consent(student_id):
            raise PermissionError(
                f"학생 {student_id!r}의 동의가 없습니다. 임베딩 저장 불가."
            )
        blob = embedding.astype(np.float64).tobytes()
        self._conn.execute(
            "INSERT INTO embeddings (student_id, embedding) VALUES (?, ?)",
            (student_id, blob),
        )
        self._conn.commit()

    def get_embeddings(self, student_id: str) -> list[np.ndarray]:
        rows = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE student_id = ?", (student_id,)
        ).fetchall()
        return [np.frombuffer(r[0], dtype=np.float64) for r in rows]

    def get_all_embeddings(self) -> dict[str, list[np.ndarray]]:
        """consent=True 학생의 모든 임베딩을 반환한다."""
        rows = self._conn.execute("""
            SELECT e.student_id, e.embedding
            FROM embeddings e
            JOIN students s ON e.student_id = s.student_id
            WHERE s.consent = 1
        """).fetchall()
        result: dict[str, list[np.ndarray]] = {}
        for sid, blob in rows:
            result.setdefault(sid, []).append(np.frombuffer(blob, dtype=np.float64))
        return result

    def delete_student_embeddings(self, student_id: str):
        self._conn.execute(
            "DELETE FROM embeddings WHERE student_id = ?", (student_id,)
        )
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ── 정적 유틸리티 ──────────────────────────────────────────────────────

    @staticmethod
    def encode_face(face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        BGR 크롭 이미지 → 128차원 임베딩. 얼굴 미감지 시 None.
        face_recognition은 RGB 입력을 기대한다.
        """
        rgb = face_img[:, :, ::-1]  # BGR → RGB
        encs = face_recognition.face_encodings(rgb)
        return encs[0] if encs else None

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 (0~1). 값이 클수록 같은 얼굴."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def best_match(
        self,
        query: np.ndarray,
        threshold: float = 0.55,
    ) -> Optional[str]:
        """
        저장된 모든 임베딩과 비교해 가장 유사한 student_id를 반환.
        threshold 미만이면 None (미등록 인물).
        """
        best_sid, best_sim = None, -1.0
        for sid, embs in self.get_all_embeddings().items():
            for emb in embs:
                sim = self.similarity(query, emb)
                if sim > best_sim:
                    best_sim, best_sid = sim, sid
        if best_sim >= threshold:
            return best_sid
        return None
