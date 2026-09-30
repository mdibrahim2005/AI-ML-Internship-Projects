import sqlite3
import json
from datetime import datetime


DATABASE = "resume_ranker.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ranking_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT NOT NULL,
            candidate_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            score REAL,
            recommendation TEXT,
            skill_score REAL,
            text_score REAL,
            title_score REAL,
            experience_score REAL,
            resume_skills TEXT,
            job_skills TEXT,
            matching_skills TEXT,
            missing_skills TEXT,
            FOREIGN KEY (session_id)
                REFERENCES ranking_sessions(id)
        )
    """)

    connection.commit()
    connection.close()


def create_session(job_description, candidate_count):

    connection = get_connection()
    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO ranking_sessions
        (
            job_description,
            candidate_count,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            job_description,
            candidate_count,
            created_at
        )
    )

    session_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return session_id


def save_candidate_result(session_id, result):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO candidate_results
        (
            session_id,
            filename,
            score,
            recommendation,
            skill_score,
            text_score,
            title_score,
            experience_score,
            resume_skills,
            job_skills,
            matching_skills,
            missing_skills
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            result.get("filename", ""),
            result.get("score", 0),
            result.get("recommendation", ""),
            result.get("skill_score", 0),
            result.get("text_score", 0),
            result.get("title_score", 0),
            result.get("experience_score", 0),
            json.dumps(result.get("resume_skills", [])),
            json.dumps(result.get("job_skills", [])),
            json.dumps(result.get("matching_skills", [])),
            json.dumps(result.get("missing_skills", []))
        )
    )

    connection.commit()
    connection.close()


def get_all_sessions():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM ranking_sessions
        ORDER BY id DESC
    """)

    sessions = cursor.fetchall()

    connection.close()

    return sessions


def get_session(session_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM ranking_sessions
        WHERE id = ?
        """,
        (session_id,)
    )

    session = cursor.fetchone()

    connection.close()

    return session


def get_session_results(session_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM candidate_results
        WHERE session_id = ?
        ORDER BY score DESC
        """,
        (session_id,)
    )

    rows = cursor.fetchall()

    connection.close()

    results = []

    for row in rows:

        results.append({
            "filename": row["filename"],
            "score": row["score"],
            "recommendation": row["recommendation"],
            "skill_score": row["skill_score"],
            "text_score": row["text_score"],
            "title_score": row["title_score"],
            "experience_score": row["experience_score"],
            "resume_skills": json.loads(
                row["resume_skills"] or "[]"
            ),
            "job_skills": json.loads(
                row["job_skills"] or "[]"
            ),
            "matching_skills": json.loads(
                row["matching_skills"] or "[]"
            ),
            "missing_skills": json.loads(
                row["missing_skills"] or "[]"
            )
        })

    return results