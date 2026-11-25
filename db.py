"""SQLite data access for the online exam proctoring project."""
import sqlite3, json, os, math
from datetime import datetime
from contextlib import contextmanager
DB_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"exam_system.db")

# InsightFace buffalo_l ArcFace embeddings are 512-dimensional. Keep this map
# explicit so a wrong model/vector combination cannot silently enter the DB.
EXPECTED_EMBEDDING_DIMS = {"buffalo_l": 512}

def _validated_embedding(features_vector, embedding_model="buffalo_l"):
    model = str(embedding_model or "").strip()
    if not model:
        raise ValueError("Tên embedding_model không được để trống.")
    if features_vector is None or isinstance(features_vector, (str, bytes)):
        raise ValueError("Embedding phải là một vector số, không được rỗng.")
    try:
        values = list(features_vector)
    except TypeError as exc:
        raise ValueError("Embedding phải là một vector 1 chiều.") from exc
    if not values:
        raise ValueError("Embedding không được rỗng.")
    vals = []
    for value in values:
        if isinstance(value, (list, tuple, dict)):
            raise ValueError("Embedding phải là vector 1 chiều.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding chứa phần tử không phải số.") from exc
        if not math.isfinite(number):
            raise ValueError("Embedding chứa NaN hoặc vô cực.")
        vals.append(number)
    expected = EXPECTED_EMBEDDING_DIMS.get(model.lower())
    if expected is not None and len(vals) != expected:
        raise ValueError(f"Embedding của model {model} phải có {expected} chiều, hiện có {len(vals)}.")
    norm_sq = sum(x * x for x in vals)
    if not math.isfinite(norm_sq) or norm_sq < 1e-16:
        raise ValueError("Embedding có độ lớn bằng 0 hoặc không hợp lệ.")
    norm = math.sqrt(norm_sq)
    return [x / norm for x in vals], model



def get_connection():
    conn=sqlite3.connect(DB_PATH,timeout=15)
    conn.row_factory=sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


@contextmanager
def connection():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _columns(cur, table):
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db():
    with connection() as conn:
        cur=conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL UNIQUE,
            features TEXT NOT NULL, registered_at TEXT NOT NULL,
            embedding_model TEXT DEFAULT 'buffalo_l', embedding_dim INTEGER)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS exam_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER NOT NULL,
            started_at TEXT NOT NULL, ended_at TEXT, status TEXT DEFAULT 'active',
            FOREIGN KEY(candidate_id) REFERENCES candidates(id))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
            violation_type TEXT NOT NULL, message TEXT, evidence_path TEXT,
            created_at TEXT NOT NULL, severity TEXT DEFAULT 'violation',
            FOREIGN KEY(session_id) REFERENCES exam_sessions(id))""")
        # Backward-compatible additive migration for existing databases.
        ccols=_columns(cur,"candidates")
        if "embedding_model" not in ccols: cur.execute("ALTER TABLE candidates ADD COLUMN embedding_model TEXT DEFAULT 'buffalo_l'")
        if "embedding_dim" not in ccols: cur.execute("ALTER TABLE candidates ADD COLUMN embedding_dim INTEGER")
        vcols=_columns(cur,"violations")
        if "severity" not in vcols: cur.execute("ALTER TABLE violations ADD COLUMN severity TEXT DEFAULT 'violation'")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_candidate ON exam_sessions(candidate_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_violations_session_time ON violations(session_id, created_at)")


def candidate_exists(full_name):
    with connection() as conn:
        return conn.execute("SELECT 1 FROM candidates WHERE full_name=?",(full_name,)).fetchone() is not None


def add_candidate(full_name, features_vector, embedding_model="buffalo_l"):
    vals, embedding_model = _validated_embedding(features_vector, embedding_model)
    with connection() as conn:
        cur=conn.execute("INSERT INTO candidates(full_name,features,registered_at,embedding_model,embedding_dim) VALUES(?,?,?,?,?)",
                          (full_name,json.dumps(vals),datetime.now().isoformat(timespec="seconds"),embedding_model,len(vals)))
        return cur.lastrowid


def update_candidate_features(candidate_id, features_vector, embedding_model="buffalo_l"):
    vals, embedding_model = _validated_embedding(features_vector, embedding_model)
    with connection() as conn:
        cur=conn.execute("UPDATE candidates SET features=?, embedding_model=?, embedding_dim=? WHERE id=?",
                          (json.dumps(vals),embedding_model,len(vals),candidate_id))
        return cur.rowcount>0


def get_candidate_by_name(full_name):
    with connection() as conn:
        row=conn.execute("SELECT * FROM candidates WHERE full_name=?",(full_name,)).fetchone()
        if not row: return None
        item=dict(row)
        try: item["features"]=json.loads(item["features"])
        except Exception: item["features"]=[]
        return item


def get_all_candidates():
    with connection() as conn:
        rows=conn.execute("SELECT id,full_name,features,registered_at,embedding_model,embedding_dim FROM candidates ORDER BY id").fetchall()
    out=[]
    for row in rows:
        item=dict(row)
        try: item["features"]=json.loads(item["features"])
        except Exception: item["features"]=[]
        out.append(item)
    return out


def start_session(candidate_id):
    with connection() as conn:
        cur=conn.execute("INSERT INTO exam_sessions(candidate_id,started_at,status) VALUES(?,?,'active')",
                          (candidate_id,datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid


def end_session(session_id,status="completed"):
    with connection() as conn:
        conn.execute("UPDATE exam_sessions SET ended_at=?,status=? WHERE id=?",
                     (datetime.now().isoformat(timespec="seconds"),status,session_id))


def log_violation(session_id,violation_type,message,evidence_path=None,severity="violation"):
    with connection() as conn:
        cur=conn.execute("INSERT INTO violations(session_id,violation_type,message,evidence_path,created_at,severity) VALUES(?,?,?,?,?,?)",
                          (session_id,violation_type,message,evidence_path,datetime.now().isoformat(timespec="seconds"),severity))
        return cur.lastrowid


def get_violations_by_session(session_id):
    with connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM violations WHERE session_id=? ORDER BY created_at,id",(session_id,)).fetchall()]


def get_session_summary(session_id):
    with connection() as conn:
        row=conn.execute("SELECT es.*,c.full_name FROM exam_sessions es JOIN candidates c ON c.id=es.candidate_id WHERE es.id=?",(session_id,)).fetchone()
        return dict(row) if row else None


def get_all_sessions():
    with connection() as conn:
        rows=conn.execute("""SELECT es.id,es.candidate_id,es.started_at,es.ended_at,es.status,c.full_name,
                            COALESCE(SUM(CASE WHEN v.id IS NOT NULL AND COALESCE(v.severity,'violation')!='warning' THEN 1 ELSE 0 END),0) AS n_violations,
                            COALESCE(SUM(CASE WHEN v.severity='warning' THEN 1 ELSE 0 END),0) AS n_warnings
                            FROM exam_sessions es JOIN candidates c ON c.id=es.candidate_id
                            LEFT JOIN violations v ON v.session_id=es.id
                            GROUP BY es.id ORDER BY es.started_at DESC""").fetchall()
        return [dict(r) for r in rows]


def get_session_ids_by_candidate(candidate_id):
    with connection() as conn:
        return [r[0] for r in conn.execute("SELECT id FROM exam_sessions WHERE candidate_id=?",(candidate_id,)).fetchall()]


def delete_violation(violation_id):
    with connection() as conn:
        cur=conn.execute("DELETE FROM violations WHERE id=?",(violation_id,))
        return cur.rowcount>0


def delete_session(session_id):
    with connection() as conn:
        conn.execute("DELETE FROM violations WHERE session_id=?",(session_id,))
        cur=conn.execute("DELETE FROM exam_sessions WHERE id=?",(session_id,))
        return cur.rowcount>0


def delete_candidate(candidate_id):
    with connection() as conn:
        ids=[r[0] for r in conn.execute("SELECT id FROM exam_sessions WHERE candidate_id=?",(candidate_id,)).fetchall()]
        for sid in ids: conn.execute("DELETE FROM violations WHERE session_id=?",(sid,))
        conn.execute("DELETE FROM exam_sessions WHERE candidate_id=?",(candidate_id,))
        cur=conn.execute("DELETE FROM candidates WHERE id=?",(candidate_id,))
        return cur.rowcount>0
