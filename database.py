"""SQLite storage so every analysis run is saved and can be reopened."""
import json
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "reviewshield.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT, name TEXT, platform TEXT,
            total INTEGER, suspicious INTEGER, trust REAL,
            summary TEXT, results TEXT)""")


def save_run(name, platform, summary, results):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (created,name,platform,total,suspicious,trust,summary,results)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), name, platform,
             summary["total"], summary["suspicious"], summary["trust_score"],
             json.dumps(summary), json.dumps(results)))
        return cur.lastrowid


def get_run(run_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    return {"id": r["id"], "created": r["created"], "name": r["name"],
            "platform": r["platform"], "summary": json.loads(r["summary"]),
            "results": json.loads(r["results"])}


def list_runs(limit=50):
    with _conn() as c:
        rows = c.execute("SELECT id,created,name,platform,total,suspicious,trust "
                         "FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_run(run_id):
    with _conn() as c:
        c.execute("DELETE FROM runs WHERE id=?", (run_id,))


def overall_stats():
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) n, COALESCE(SUM(total),0) t, "
                      "COALESCE(SUM(suspicious),0) s, AVG(trust) tr FROM runs").fetchone()
    return {"runs": r["n"], "reviews": r["t"], "suspicious": r["s"],
            "avg_trust": round(r["tr"], 1) if r["tr"] is not None else None}
