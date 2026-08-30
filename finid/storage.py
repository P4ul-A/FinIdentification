"""Temporary SQLite-backed state for encounter-aware identification runs."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


def _path_key(path: Path) -> str:
    """Return a stable absolute database key without redundant filesystem I/O.

    Parameters:
        path: Path to normalize for SQLite storage or lookup.

    Returns:
        Absolute path text. Already-absolute paths are returned without a
        filesystem-resolving system call.
    """
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else path.resolve())


class ResultStore:
    """Keep inventory, detections, identities, and assignments on disk."""

    COMMIT_INTERVAL = 500

    def __init__(self) -> None:
        handle, name = tempfile.mkstemp(prefix="finid-", suffix=".sqlite3")
        os.close(handle)
        self.path = Path(name)
        self._writes_since_commit = 0
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE encounters (
                path TEXT PRIMARY KEY, relative_path TEXT NOT NULL,
                output_path TEXT, image_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE scan_directories (
                path TEXT PRIMARY KEY,
                parent_path TEXT,
                name TEXT NOT NULL,
                child_count INTEGER NOT NULL DEFAULT 0,
                direct_jpeg_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX scan_directories_parent ON scan_directories(parent_path);
            CREATE TABLE scan_images (
                path TEXT PRIMARY KEY
            );
            CREATE INDEX scan_images_order ON scan_images(
                path COLLATE NOCASE, path
            );
            CREATE TABLE source_images (
                id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                encounter_path TEXT NOT NULL, relative_path TEXT NOT NULL,
                filename TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                cluster_category TEXT, cluster_side TEXT, copied_filename TEXT,
                failure_message TEXT, primary_identity TEXT,
                primary_identity_score REAL
            );
            CREATE INDEX source_images_pending ON source_images(
                state, path COLLATE NOCASE, path
            );
            CREATE INDEX source_images_encounter_order ON source_images(
                encounter_path, relative_path COLLATE NOCASE, relative_path
            );
            CREATE INDEX source_images_category_order ON source_images(
                encounter_path, cluster_category,
                relative_path COLLATE NOCASE, relative_path
            );
            CREATE INDEX source_images_identity_order ON source_images(
                encounter_path, primary_identity,
                relative_path COLLATE NOCASE, relative_path
            );
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL, class_name TEXT NOT NULL, side TEXT,
                confidence REAL NOT NULL, selected INTEGER NOT NULL,
                x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL, y2 INTEGER NOT NULL
            );
            CREATE INDEX detections_image ON detections(image_id);
            CREATE TABLE identities (
                id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL,
                detection_id INTEGER, identity TEXT NOT NULL, score REAL NOT NULL,
                score_type TEXT NOT NULL, UNIQUE(image_id, identity)
            );
            CREATE INDEX identities_image ON identities(image_id);
            CREATE INDEX identities_ranked ON identities(
                image_id, score DESC, identity
            );
            CREATE TABLE skipped_undated (path TEXT PRIMARY KEY);
            """
        )

    def _commit_periodically(self) -> None:
        """Bound transaction size while avoiding a disk sync for every image."""
        self._writes_since_commit += 1
        if self._writes_since_commit >= self.COMMIT_INTERVAL:
            self.commit()

    def commit(self) -> None:
        """Commit pending state changes and reset the write counter."""
        self.connection.commit()
        self._writes_since_commit = 0

    def add_encounter(self, path: Path, relative_path: Path) -> None:
        """Add an encounter and its mirrored path.

        Parameters:
            path: Absolute source encounter root.
            relative_path: Mirrored path beneath the selected input root.

        Returns:
            None.
        """
        self.connection.execute(
            "INSERT OR IGNORE INTO encounters(path, relative_path) VALUES (?, ?)",
            (_path_key(path), relative_path.as_posix()),
        )
        self._commit_periodically()

    def add_scanned_directory(self, path: Path, parent: Path | None) -> None:
        """Record a directory found during the single filesystem scan."""
        self.connection.execute(
            """INSERT OR IGNORE INTO scan_directories(path, parent_path, name)
               VALUES (?, ?, ?)""",
            (str(path), str(parent) if parent is not None else None, path.name),
        )
        if parent is not None:
            self.connection.execute(
                "UPDATE scan_directories SET child_count = child_count + 1 WHERE path = ?",
                (str(parent),),
            )
        self._commit_periodically()

    def add_scanned_image(self, path: Path) -> None:
        """Record a JPEG found during the single filesystem scan."""
        self.connection.execute(
            "INSERT INTO scan_images(path) VALUES (?)",
            (str(path),),
        )
        self.connection.execute(
            """UPDATE scan_directories SET direct_jpeg_count = direct_jpeg_count + 1
               WHERE path = ?""",
            (str(path.parent),),
        )
        self._commit_periodically()

    def scanned_directories(self) -> Iterator[sqlite3.Row]:
        """Yield scanned directory metadata without loading it into memory."""
        yield from self.connection.execute(
            "SELECT * FROM scan_directories ORDER BY path COLLATE NOCASE, path"
        )

    def scanned_children(self, parent: Path) -> Iterator[sqlite3.Row]:
        """Yield direct scanned child directories."""
        yield from self.connection.execute(
            """SELECT * FROM scan_directories WHERE parent_path = ?
               ORDER BY path COLLATE NOCASE, path""",
            (str(parent),),
        )

    def group_sibling_directories(self) -> Iterator[Path]:
        """Yield GROUP-prefixed directories with at least one GROUP sibling."""
        yield from (
            Path(row["path"])
            for row in self.connection.execute(
                """SELECT child.path FROM scan_directories AS child
                   JOIN (
                       SELECT parent_path FROM scan_directories
                       WHERE name LIKE 'GROUP%'
                       GROUP BY parent_path HAVING COUNT(*) >= 2
                   ) AS grouped ON grouped.parent_path = child.parent_path
                   WHERE child.name LIKE 'GROUP%'
                   ORDER BY child.path COLLATE NOCASE, child.path"""
            )
        )

    def scanned_images(self) -> Iterator[sqlite3.Row]:
        """Yield scanned JPEG rows in deterministic path order."""
        yield from self.connection.execute(
            "SELECT * FROM scan_images ORDER BY path COLLATE NOCASE, path"
        )

    def discard_scan_state(self) -> None:
        """Drop provisional inventory tables after encounter assignment."""
        self.connection.executescript(
            "DROP TABLE scan_images; DROP TABLE scan_directories;"
        )
        self.commit()

    def set_encounter_output(self, path: Path, output_path: Path) -> None:
        """Record the report/output root for an encounter."""
        self.connection.execute(
            "UPDATE encounters SET output_path = ? WHERE path = ?",
            (str(output_path), _path_key(path)),
        )
        self._commit_periodically()

    def add_image(self, path: Path, encounter: Path) -> None:
        """Add a source image owned by an encounter."""
        path_text = _path_key(path)
        encounter_text = _path_key(encounter)
        relative_path = Path(path_text).relative_to(Path(encounter_text)).as_posix()
        self.connection.execute(
            """INSERT INTO source_images(path, encounter_path, relative_path, filename)
               VALUES (?, ?, ?, ?)""",
            (path_text, encounter_text, relative_path, path.name),
        )
        self.connection.execute(
            "UPDATE encounters SET image_count = image_count + 1 WHERE path = ?",
            (encounter_text,),
        )
        self._commit_periodically()

    def add_skipped_undated(self, path: Path) -> None:
        """Record a JPEG excluded for lacking a dated encounter ancestor."""
        self.connection.execute(
            "INSERT OR IGNORE INTO skipped_undated(path) VALUES (?)", (_path_key(path),)
        )
        self._commit_periodically()

    def finish_inventory(self) -> None:
        """Commit inventoried rows."""
        self.commit()

    def pending_paths(self) -> Iterator[Path]:
        """Yield pending source paths in deterministic order."""
        for row in self.connection.execute(
            "SELECT path FROM source_images WHERE state = 'pending' ORDER BY path COLLATE NOCASE, path"
        ):
            yield Path(row["path"])

    def _image_id(self, image_path: Path) -> int | None:
        """Return an image ID, resolving platform path aliases only on a miss.

        Parameters:
            image_path: Source image path to find.

        Returns:
            Database image ID, or ``None`` when the image was not inventoried.
        """
        key = _path_key(image_path)
        row = self.connection.execute(
            "SELECT id FROM source_images WHERE path = ?", (key,)
        ).fetchone()
        if row is None:
            resolved_key = str(Path(image_path).expanduser().resolve())
            if resolved_key != key:
                row = self.connection.execute(
                    "SELECT id FROM source_images WHERE path = ?", (resolved_key,)
                ).fetchone()
        return int(row["id"]) if row is not None else None

    def record_result(
        self, image_path: Path, detections: Iterable[dict[str, object]],
        identities: Iterable[dict[str, object]], category: str, side: str | None,
    ) -> None:
        """Persist detections, identities, and the image's single assignment."""
        image_id = self._image_id(image_path)
        if image_id is None:
            raise KeyError(f"Image was not inventoried: {image_path}")
        detection_ids: list[int] = []
        for detection in detections:
            cursor = self.connection.execute(
                """INSERT INTO detections(
                    image_id, class_id, class_name, side, confidence, selected,
                    x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (image_id, int(detection["class_id"]), str(detection["class_name"]),
                 detection.get("side"), float(detection["confidence"]),
                 1 if detection.get("selected") else 0, int(detection["x1"]),
                 int(detection["y1"]), int(detection["x2"]), int(detection["y2"])),
            )
            detection_ids.append(int(cursor.lastrowid))
        best: dict[str, dict[str, object]] = {}
        for identity in identities:
            name = str(identity["identity"])
            if name not in best or float(identity["score"]) > float(best[name]["score"]):
                best[name] = identity
        for identity in best.values():
            index = int(identity.get("detection_index", -1))
            detection_id = detection_ids[index] if 0 <= index < len(detection_ids) else None
            self.connection.execute(
                """INSERT INTO identities(image_id, detection_id, identity, score, score_type)
                   VALUES (?, ?, ?, ?, ?)""",
                (image_id, detection_id, str(identity["identity"]),
                 float(identity["score"]), str(identity["score_type"])),
            )
        primary = min(
            best.values(),
            key=lambda item: (-float(item["score"]), str(item["identity"])),
            default=None,
        )
        self.connection.execute(
            """UPDATE source_images SET state = 'processed', cluster_category = ?,
               cluster_side = ?, primary_identity = ?, primary_identity_score = ?
               WHERE id = ?""",
            (
                category,
                side,
                str(primary["identity"]) if primary is not None else None,
                float(primary["score"]) if primary is not None else None,
                image_id,
            ),
        )
        self._commit_periodically()

    def record_failure(self, image_path: Path, message: str) -> None:
        """Record a failed image as a Rest assignment."""
        image_id = self._image_id(image_path)
        if image_id is None:
            return
        self.connection.execute(
            """UPDATE source_images SET state = 'failed', cluster_category = 'Rest',
               cluster_side = NULL, failure_message = ? WHERE id = ?""",
            (message, image_id),
        )
        self._commit_periodically()

    def mark_pending_failed(self, message: str) -> int:
        """Mark pending rows as failed and return their count."""
        cursor = self.connection.execute(
            """UPDATE source_images SET state = 'failed', cluster_category = 'Rest',
               cluster_side = NULL, failure_message = ? WHERE state = 'pending'""", (message,)
        )
        self.commit()
        return int(cursor.rowcount)

    def set_copied_filename(self, image_id: int, filename: str) -> None:
        """Record a copied flattened filename."""
        self.connection.execute(
            "UPDATE source_images SET copied_filename = ? WHERE id = ?", (filename, image_id)
        )
        self._commit_periodically()

    def encounters(self) -> Iterator[sqlite3.Row]:
        """Yield encounter rows in source path order."""
        yield from self.connection.execute("SELECT * FROM encounters ORDER BY path")

    def encounter_images(self, encounter: Path, category: str | None = None) -> Iterator[sqlite3.Row]:
        """Yield completed encounter images, optionally in one category."""
        sql = "SELECT * FROM source_images WHERE encounter_path = ? AND state != 'pending'"
        values: list[object] = [_path_key(encounter)]
        if category is not None:
            sql += " AND cluster_category = ?"
            values.append(category)
        sql += " ORDER BY relative_path COLLATE NOCASE, relative_path"
        yield from self.connection.execute(sql, values)

    def encounter_category_count(self, encounter: Path, category: str) -> int:
        """Return the number of completed encounter images in a category."""
        return int(
            self.connection.execute(
                """SELECT COUNT(*) FROM source_images
                   WHERE encounter_path = ? AND state != 'pending'
                   AND cluster_category = ?""",
                (_path_key(encounter), category),
            ).fetchone()[0]
        )

    def encounter_counts(self, encounter: Path) -> dict[str, int]:
        """Return all report category and failure counts in one query.

        Parameters:
            encounter: Source encounter root.

        Returns:
            Counts for Total, IDed, FinSaddle, Eyes, Rest, and Failures.
        """
        counts = {
            "Total": 0,
            "IDed": 0,
            "FinSaddle": 0,
            "Eyes": 0,
            "Rest": 0,
            "Failures": 0,
        }
        for row in self.connection.execute(
            """SELECT cluster_category, COUNT(*) AS image_count,
                      SUM(failure_message IS NOT NULL) AS failure_count
               FROM source_images
               WHERE encounter_path = ? AND state != 'pending'
               GROUP BY cluster_category""",
            (_path_key(encounter),),
        ):
            category = str(row["cluster_category"])
            image_count = int(row["image_count"])
            if category in counts:
                counts[category] = image_count
            counts["Total"] += image_count
            counts["Failures"] += int(row["failure_count"] or 0)
        return counts

    def encounter_failure_count(self, encounter: Path) -> int:
        """Return the number of failed images in an encounter."""
        return int(
            self.connection.execute(
                """SELECT COUNT(*) FROM source_images
                   WHERE encounter_path = ? AND failure_message IS NOT NULL""",
                (_path_key(encounter),),
            ).fetchone()[0]
        )

    def primary_identity_groups(self, encounter: Path) -> Iterator[sqlite3.Row]:
        """Yield primary identities and image counts for an encounter."""
        yield from self.connection.execute(
            """SELECT primary_identity AS identity, COUNT(*) AS image_count
               FROM source_images WHERE encounter_path = ?
                 AND cluster_category = 'IDed' AND primary_identity IS NOT NULL
               GROUP BY primary_identity
               ORDER BY primary_identity COLLATE NOCASE, primary_identity""",
            (_path_key(encounter),),
        )

    def identified_images(self, encounter: Path, primary_identity: str) -> Iterator[sqlite3.Row]:
        """Yield identified images whose highest-scoring identity matches a value."""
        yield from self.connection.execute(
            """SELECT * FROM source_images WHERE encounter_path = ?
                 AND cluster_category = 'IDed' AND primary_identity = ?
               ORDER BY relative_path COLLATE NOCASE, relative_path""",
            (_path_key(encounter), primary_identity),
        )

    def detections(self, image_id: int) -> Iterator[sqlite3.Row]:
        """Yield raw detections for an image."""
        yield from self.connection.execute(
            "SELECT * FROM detections WHERE image_id = ? ORDER BY confidence DESC, id", (image_id,)
        )

    def identities(self, image_id: int) -> Iterator[sqlite3.Row]:
        """Yield accepted identities for an image by score."""
        yield from self.connection.execute(
            "SELECT * FROM identities WHERE image_id = ? ORDER BY score DESC, identity", (image_id,)
        )

    def total_images(self) -> int:
        """Return inventoried dated JPEG count."""
        return int(self.connection.execute("SELECT COUNT(*) FROM source_images").fetchone()[0])

    def processed_images(self) -> int:
        """Return completed or failed image count."""
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM source_images WHERE state != 'pending'"
        ).fetchone()[0])

    def encounter_count(self) -> int:
        """Return discovered encounter count."""
        return int(self.connection.execute("SELECT COUNT(*) FROM encounters").fetchone()[0])

    def skipped_undated_count(self) -> int:
        """Return excluded undated JPEG count."""
        return int(self.connection.execute("SELECT COUNT(*) FROM skipped_undated").fetchone()[0])

    def skipped_undated_paths(self, limit: int = 10) -> list[Path]:
        """Return representative excluded paths."""
        return [Path(row[0]) for row in self.connection.execute(
            "SELECT path FROM skipped_undated ORDER BY path LIMIT ?", (limit,)
        )]

    def clustered_images(self) -> int:
        """Return number of rows with a copied filename."""
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM source_images WHERE copied_filename IS NOT NULL"
        ).fetchone()[0])

    def close(self) -> None:
        """Close and remove temporary database files."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None  # type: ignore[assignment]
        for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
