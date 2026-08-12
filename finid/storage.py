"""Temporary SQLite-backed run state for bounded-memory identification."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


class ResultStore:
    """Disk-backed run state so memory use does not grow with the image tree."""

    def __init__(self) -> None:
        handle, name = tempfile.mkstemp(prefix="finid-", suffix=".sqlite3")
        os.close(handle)
        self.path = Path(name)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE directories (
                path TEXT PRIMARY KEY,
                parent_path TEXT,
                jpeg_count INTEGER NOT NULL DEFAULT 0,
                processed_count INTEGER NOT NULL DEFAULT 0,
                detected_fins INTEGER NOT NULL DEFAULT 0,
                accepted_images INTEGER NOT NULL DEFAULT 0,
                rejected_images INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE source_images (
                path TEXT PRIMARY KEY,
                directory_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX source_images_directory ON source_images(directory_path);
            CREATE TABLE skipped_files (
                directory_path TEXT NOT NULL,
                filename TEXT NOT NULL
            );
            CREATE TABLE failures (
                directory_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE TABLE accepted_images (
                id INTEGER PRIMARY KEY,
                directory_path TEXT NOT NULL,
                filename TEXT NOT NULL
            );
            CREATE INDEX accepted_images_directory ON accepted_images(directory_path);
            CREATE TABLE accepted_fins (
                image_id INTEGER NOT NULL,
                identity TEXT NOT NULL,
                score REAL NOT NULL,
                score_type TEXT NOT NULL,
                detection_confidence REAL NOT NULL,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL
            );
            """
        )

    def add_directory(self, path: Path, parent: Path | None) -> None:
        """Add one discovered directory and optional parent."""

        self.connection.execute(
            "INSERT OR IGNORE INTO directories(path, parent_path) VALUES (?, ?)",
            (str(path), str(parent) if parent is not None else None),
        )

    def add_image(self, path: Path) -> None:
        """Add one pending source image and increment its directory count."""

        self.connection.execute(
            "INSERT INTO source_images(path, directory_path, filename) VALUES (?, ?, ?)",
            (str(path), str(path.parent), path.name),
        )
        self.connection.execute(
            "UPDATE directories SET jpeg_count = jpeg_count + 1 WHERE path = ?",
            (str(path.parent),),
        )

    def add_skipped(self, directory: Path, filename: str) -> None:
        """Record one unsupported file in a directory."""

        directory = directory.resolve()
        self.connection.execute(
            "INSERT INTO skipped_files(directory_path, filename) VALUES (?, ?)",
            (str(directory), filename),
        )
        self.connection.execute(
            "UPDATE directories SET skipped_count = skipped_count + 1 WHERE path = ?",
            (str(directory),),
        )

    def finish_inventory(self) -> None:
        """Commit all inventory rows."""

        self.connection.commit()

    def record_result(
        self,
        image_path: Path,
        detected_fins: int,
        accepted: Iterable[dict[str, object]],
    ) -> None:
        """Record completed detections and accepted identifications."""

        image_path = image_path.resolve()
        accepted_rows = list(accepted)
        directory = str(image_path.parent)
        state = "accepted" if accepted_rows else "rejected"
        self.connection.execute(
            "UPDATE source_images SET state = ? WHERE path = ?",
            (state, str(image_path)),
        )
        self.connection.execute(
            """
            UPDATE directories
            SET processed_count = processed_count + 1,
                detected_fins = detected_fins + ?,
                accepted_images = accepted_images + ?,
                rejected_images = rejected_images + ?
            WHERE path = ?
            """,
            (
                detected_fins,
                1 if accepted_rows else 0,
                0 if accepted_rows else 1,
                directory,
            ),
        )
        if accepted_rows:
            cursor = self.connection.execute(
                "INSERT INTO accepted_images(directory_path, filename) VALUES (?, ?)",
                (directory, image_path.name),
            )
            image_id = int(cursor.lastrowid)
            self.connection.executemany(
                """
                INSERT INTO accepted_fins(
                    image_id, identity, score, score_type, detection_confidence,
                    x1, y1, x2, y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        image_id,
                        str(row["identity"]),
                        float(row["score"]),
                        str(row["score_type"]),
                        float(row["detection_confidence"]),
                        int(row["x1"]),
                        int(row["y1"]),
                        int(row["x2"]),
                        int(row["y2"]),
                    )
                    for row in accepted_rows
                ],
            )
        self.connection.commit()

    def record_failure(self, image_path: Path, message: str) -> None:
        """Record one failed source image and its error message."""

        image_path = image_path.resolve()
        self.connection.execute(
            "UPDATE source_images SET state = 'failed' WHERE path = ?",
            (str(image_path),),
        )
        self.connection.execute(
            """
            UPDATE directories
            SET processed_count = processed_count + 1,
                failure_count = failure_count + 1
            WHERE path = ?
            """,
            (str(image_path.parent),),
        )
        self.connection.execute(
            "INSERT INTO failures(directory_path, filename, message) VALUES (?, ?, ?)",
            (str(image_path.parent), image_path.name, message),
        )
        self.connection.commit()

    def mark_pending_failed(self, message: str) -> int:
        """Mark every pending image failed and return the affected count."""

        rows = list(
            self.connection.execute(
                "SELECT path FROM source_images WHERE state = 'pending' ORDER BY path"
            )
        )
        for row in rows:
            self.record_failure(Path(row["path"]), message)
        return len(rows)

    def directories(self) -> Iterator[sqlite3.Row]:
        """Yield all directory summary rows in path order."""

        yield from self.connection.execute("SELECT * FROM directories ORDER BY path")

    def directory_paths(self) -> list[Path]:
        """Return all inventoried directory paths."""

        return [Path(row["path"]) for row in self.directories()]

    def has_directory(self, path: Path) -> bool:
        """Return whether a directory exists in the inventory."""

        path = path.resolve()
        row = self.connection.execute(
            "SELECT 1 FROM directories WHERE path = ?",
            (str(path),),
        ).fetchone()
        return row is not None

    def child_directories(self, path: Path) -> Iterator[Path]:
        """Yield direct inventoried children of a directory."""

        path = path.resolve()
        for row in self.connection.execute(
            """
            SELECT path FROM directories WHERE parent_path = ?
            ORDER BY path COLLATE NOCASE, path
            """,
            (str(path),),
        ):
            yield Path(row["path"])

    def accepted_images(self, directory: Path) -> Iterator[sqlite3.Row]:
        """Yield accepted images in one directory."""

        directory = directory.resolve()
        yield from self.connection.execute(
            """
            SELECT id, filename FROM accepted_images
            WHERE directory_path = ? ORDER BY filename COLLATE NOCASE, filename
            """,
            (str(directory),),
        )

    def accepted_fins(self, image_id: int) -> Iterator[sqlite3.Row]:
        """Yield accepted fin identifications for one image."""

        yield from self.connection.execute(
            """
            SELECT * FROM accepted_fins WHERE image_id = ?
            ORDER BY score DESC, identity
            """,
            (image_id,),
        )

    def skipped_files(self, directory: Path) -> Iterator[str]:
        """Yield skipped filenames in one directory."""

        directory = directory.resolve()
        for row in self.connection.execute(
            """
            SELECT filename FROM skipped_files WHERE directory_path = ?
            ORDER BY filename COLLATE NOCASE, filename
            """,
            (str(directory),),
        ):
            yield str(row["filename"])

    def failures(self, directory: Path) -> Iterator[sqlite3.Row]:
        """Yield failed filenames and messages in one directory."""

        directory = directory.resolve()
        yield from self.connection.execute(
            """
            SELECT filename, message FROM failures WHERE directory_path = ?
            ORDER BY filename COLLATE NOCASE, filename
            """,
            (str(directory),),
        )

    def total_images(self) -> int:
        """Return the total number of inventoried JPEG images."""

        row = self.connection.execute(
            "SELECT COALESCE(SUM(jpeg_count), 0) AS total FROM directories"
        ).fetchone()
        return int(row["total"])

    def processed_images(self) -> int:
        """Return the total number of completed image rows."""

        row = self.connection.execute(
            "SELECT COALESCE(SUM(processed_count), 0) AS total FROM directories"
        ).fetchone()
        return int(row["total"])

    def directory_count(self) -> int:
        """Return the number of inventoried directories."""

        row = self.connection.execute("SELECT COUNT(*) AS total FROM directories").fetchone()
        return int(row["total"])

    def close(self) -> None:
        """Close SQLite and remove all temporary database files."""

        if self.connection is not None:
            self.connection.close()
            self.connection = None  # type: ignore[assignment]
        for candidate in (
            self.path,
            Path(str(self.path) + "-wal"),
            Path(str(self.path) + "-shm"),
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
