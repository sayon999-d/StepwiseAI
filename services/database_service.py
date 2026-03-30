import json
import hashlib
import secrets
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import bcrypt
import os
import tempfile

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    import sys
    vendor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vendor')
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        psycopg2 = None

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/career_regret_ai")
SQLITE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'career_ai.db')


class DatabaseService:

    def __init__(self, database_url: str = DATABASE_URL, sqlite_path: str = SQLITE_PATH):
        self.database_url = database_url
        self.sqlite_path = sqlite_path
        self.backend = None          
        self._pg_available = False

        if psycopg2 is not None:
            try:
                conn = psycopg2.connect(self.database_url)
                conn.close()
                self._pg_available = True
                self.backend = 'postgres'
            except Exception:
                self._pg_available = False

        if not self._pg_available:
            self.backend = 'sqlite'
            print(f"[DatabaseService] PostgreSQL unavailable — using SQLite at {self.sqlite_path}")

        self._initialize_database()

    @contextmanager
    def get_connection(self):
        if self.backend == 'postgres':
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = False
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _q(self, sql: str) -> str:
        if self.backend == 'postgres':
            return sql
        sql = sql.replace('%s', '?')
        sql = sql.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
        sql = sql.replace('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', 'TEXT DEFAULT CURRENT_TIMESTAMP')
        sql = sql.replace('TIMESTAMP', 'TEXT')
        sql = sql.replace('BOOLEAN DEFAULT FALSE', 'INTEGER DEFAULT 0')
        sql = sql.replace('BOOLEAN DEFAULT TRUE', 'INTEGER DEFAULT 1')
        sql = sql.replace('BOOLEAN', 'INTEGER')
        sql = sql.replace('= TRUE', '= 1')
        sql = sql.replace('= FALSE', '= 0')
        sql = sql.replace('ILIKE', 'LIKE')
        sql = sql.replace("CURRENT_DATE - INTERVAL '30 days'", "datetime('now', '-30 days')")
        return sql

    def _is_integrity_error(self, exc: Exception) -> bool:
        if self.backend == 'postgres' and psycopg2:
            return isinstance(exc, psycopg2.IntegrityError)
        return isinstance(exc, sqlite3.IntegrityError)

    def _initialize_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    avatar_url TEXT,
                    bio TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    email_verified BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    preferences TEXT DEFAULT '{}',
                    theme TEXT DEFAULT 'dark',
                    notification_settings TEXT DEFAULT '{}'
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    refresh_token TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    device_info TEXT,
                    ip_address TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    decision_type TEXT,
                    status TEXT DEFAULT 'pending',
                    predicted_regret REAL,
                    actual_regret REAL,
                    confidence REAL,
                    emotions TEXT DEFAULT '[]',
                    factors TEXT DEFAULT '[]',
                    pros TEXT DEFAULT '[]',
                    cons TEXT DEFAULT '[]',
                    chosen_option TEXT,
                    alternatives TEXT DEFAULT '[]',
                    notes TEXT,
                    tags TEXT DEFAULT '[]',
                    template_id TEXT,
                    nlp_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decided_at TIMESTAMP,
                    follow_up_date TIMESTAMP,
                    outcome_recorded_at TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    actual_regret REAL,
                    satisfaction_score REAL,
                    outcome_notes TEXT,
                    surprises TEXT DEFAULT '[]',
                    lessons_learned TEXT,
                    would_do_again BOOLEAN,
                    time_since_decision_days INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS calendar_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    event_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    location TEXT,
                    reminder_minutes TEXT DEFAULT '[30, 1440]',
                    recurrence TEXT,
                    color_id TEXT DEFAULT '1',
                    google_event_id TEXT,
                    synced BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT,
                    doc_type TEXT DEFAULT 'general',
                    summary TEXT,
                    embedding_id TEXT,
                    size_bytes INTEGER,
                    char_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    preview TEXT,
                    messages TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    notification_type TEXT DEFAULT 'info',
                    action_url TEXT,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS user_analytics (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT DEFAULT '{}',
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS ai_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    message_id TEXT,
                    feedback_type TEXT,
                    rating INTEGER,
                    comment TEXT,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    backup_type TEXT DEFAULT 'full',
                    file_path TEXT,
                    size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_user ON calendar_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar_events(start_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')

            conn.commit()
            if self.backend == 'postgres':
                print(f"PostgreSQL database initialized: {self.database_url.split('@')[-1] if '@' in self.database_url else 'local'}")
            else:
                print(f"SQLite database initialized: {self.sqlite_path}")

    def _build_update_clause(self, updates: Dict[str, Any], allowed_fields: List[str]) -> Tuple[str, List[Any]]:
        safe_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        if not safe_updates:
            return "", []
        ph = "?" if self.backend == 'sqlite' else "%s"
        set_clause = ", ".join([f"{k} = {ph}" for k in safe_updates.keys()])
        return set_clause, list(safe_updates.values())

    def _row_to_dict(self, cursor, row):
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def create_user(self, email: str, username: str, password: str, full_name: str = None) -> Optional[Dict]:
        user_id = secrets.token_hex(16)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(self._q('''
                    INSERT INTO users (id, email, username, password_hash, full_name, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                '''), (user_id, email.lower(), username, password_hash, full_name, datetime.utcnow().isoformat()))
                conn.commit()
                return self.get_user_by_id(user_id)
            except Exception as e:
                if self._is_integrity_error(e):
                    conn.rollback()
                    return None
                raise

    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM users WHERE email = %s AND is_active = TRUE'), (email.lower(),))
            row = cursor.fetchone()

            if row:
                d = self._row_to_dict(cursor, row)
                if bcrypt.checkpw(password.encode('utf-8'), d['password_hash'].encode('utf-8')):
                    cursor.execute(self._q('UPDATE users SET last_login = %s WHERE id = %s'),
                                 (datetime.utcnow().isoformat(), d['id']))
                    conn.commit()
                    return d
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM users WHERE id = %s'), (user_id,))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row)

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM users WHERE email = %s'), (email.lower(),))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row)

    def update_user(self, user_id: str, **updates) -> bool:
        allowed_fields = ['full_name', 'bio', 'avatar_url', 'preferences', 'theme', 'notification_settings']
        set_clause, values = self._build_update_clause(updates, allowed_fields)

        if not set_clause:
            return False

        ph = "?" if self.backend == 'sqlite' else "%s"
        values.append(user_id)
        query = "UPDATE users SET " + set_clause + f" WHERE id = {ph}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        user = self.get_user_by_id(user_id)
        if not user:
            return False

        if not bcrypt.checkpw(old_password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return False

        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('UPDATE users SET password_hash = %s WHERE id = %s'), (new_hash, user_id))
            conn.commit()
            return True

    def create_session(self, user_id: str, device_info: str = None, ip_address: str = None,
                       expires_hours: int = 24) -> Dict:
        session_id = secrets.token_hex(16)
        token = secrets.token_urlsafe(64)
        refresh_token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO sessions (id, user_id, token, refresh_token, expires_at, device_info, ip_address)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''), (session_id, user_id, token, refresh_token, expires_at.isoformat(), device_info, ip_address))
            conn.commit()

        return {
            'session_id': session_id,
            'token': token,
            'refresh_token': refresh_token,
            'expires_at': expires_at.isoformat()
        }

    def validate_session(self, token: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                SELECT s.*, u.email, u.username, u.full_name
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = %s AND s.is_active = TRUE AND s.expires_at > %s
            '''), (token, datetime.utcnow().isoformat()))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row)

    def invalidate_session(self, token: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('UPDATE sessions SET is_active = FALSE WHERE token = %s'), (token,))
            conn.commit()
            return cursor.rowcount > 0

    def refresh_session(self, refresh_token: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM sessions WHERE refresh_token = %s AND is_active = TRUE'), (refresh_token,))
            row = cursor.fetchone()

            if row:
                d = self._row_to_dict(cursor, row)
                new_token = secrets.token_urlsafe(64)
                new_expires = datetime.utcnow() + timedelta(hours=24)
                cursor.execute(self._q('''
                    UPDATE sessions SET token = %s, expires_at = %s WHERE id = %s
                '''), (new_token, new_expires.isoformat(), d['id']))
                conn.commit()
                return {
                    'token': new_token,
                    'expires_at': new_expires.isoformat()
                }
            return None

    def create_decision(self, user_id: str, data: Dict) -> Dict:
        decision_id = secrets.token_hex(8)
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO decisions (
                    id, user_id, title, description, decision_type, status,
                    predicted_regret, confidence, emotions, factors, pros, cons,
                    chosen_option, alternatives, notes, tags, template_id, nlp_analysis,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''), (
                decision_id, user_id,
                data.get('title', ''),
                data.get('description', ''),
                data.get('decision_type', 'general'),
                data.get('status', 'pending'),
                data.get('predicted_regret'),
                data.get('confidence'),
                json.dumps(data.get('emotions', [])),
                json.dumps(data.get('factors', [])),
                json.dumps(data.get('pros', [])),
                json.dumps(data.get('cons', [])),
                data.get('chosen_option'),
                json.dumps(data.get('alternatives', [])),
                data.get('notes'),
                json.dumps(data.get('tags', [])),
                data.get('template_id'),
                json.dumps(data.get('nlp_analysis', {})),
                now, now
            ))
            conn.commit()

        return self.get_decision(user_id, decision_id)

    def get_decision(self, user_id: str, decision_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM decisions WHERE id = %s AND user_id = %s'), (decision_id, user_id))
            row = cursor.fetchone()
            if row:
                d = self._row_to_dict(cursor, row)
                for field in ['emotions', 'factors', 'pros', 'cons', 'alternatives', 'tags', 'nlp_analysis']:
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except:
                            pass
                return d
            return None

    def get_decisions(self, user_id: str, status: str = None, decision_type: str = None,
                      search: str = None, limit: int = 50, offset: int = 0,
                      sort_by: str = 'created_at', sort_order: str = 'desc') -> Tuple[List[Dict], int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = self._q('SELECT * FROM decisions WHERE user_id = %s')
            count_query = self._q('SELECT COUNT(*) as count FROM decisions WHERE user_id = %s')
            params = [user_id]

            ph = "?" if self.backend == 'sqlite' else "%s"

            if status:
                query += f' AND status = {ph}'
                count_query += f' AND status = {ph}'
                params.append(status)

            if decision_type:
                query += f' AND decision_type = {ph}'
                count_query += f' AND decision_type = {ph}'
                params.append(decision_type)

            if search:
                like_op = 'LIKE' if self.backend == 'sqlite' else 'ILIKE'
                query += f' AND (title {like_op} {ph} OR description {like_op} {ph} OR notes {like_op} {ph})'
                count_query += f' AND (title {like_op} {ph} OR description {like_op} {ph} OR notes {like_op} {ph})'
                search_param = f'%{search}%'
                params.extend([search_param, search_param, search_param])

            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]

            allowed_sort = ['created_at', 'updated_at', 'predicted_regret', 'title', 'decision_type']
            if sort_by not in allowed_sort:
                sort_by = 'created_at'
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'

            query += f' ORDER BY {sort_by} {sort_order} LIMIT {ph} OFFSET {ph}'
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            decisions = []
            for row in rows:
                d = self._row_to_dict(cursor, row)
                for field in ['emotions', 'factors', 'pros', 'cons', 'alternatives', 'tags', 'nlp_analysis']:
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except:
                            pass
                decisions.append(d)

            return decisions, total

    def update_decision(self, user_id: str, decision_id: str, data: Dict) -> bool:
        allowed_fields = ['title', 'description', 'decision_type', 'status', 'predicted_regret',
                         'actual_regret', 'confidence', 'emotions', 'factors', 'pros', 'cons',
                         'chosen_option', 'alternatives', 'notes', 'tags', 'decided_at', 'follow_up_date']

        updates = {}
        for k, v in data.items():
            if k in allowed_fields:
                if isinstance(v, (list, dict)):
                    updates[k] = json.dumps(v)
                else:
                    updates[k] = v

        if not updates:
            return False

        updates['updated_at'] = datetime.utcnow().isoformat()
        set_clause, values = self._build_update_clause(updates, allowed_fields + ['updated_at'])

        if not set_clause:
            return False

        ph = "?" if self.backend == 'sqlite' else "%s"
        values.extend([decision_id, user_id])
        query = "UPDATE decisions SET " + set_clause + f" WHERE id = {ph} AND user_id = {ph}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_decision(self, user_id: str, decision_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('DELETE FROM decisions WHERE id = %s AND user_id = %s'), (decision_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def record_outcome(self, user_id: str, decision_id: str, data: Dict) -> Dict:
        outcome_id = secrets.token_hex(8)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO decision_outcomes (
                    id, decision_id, user_id, actual_regret, satisfaction_score,
                    outcome_notes, surprises, lessons_learned, would_do_again, time_since_decision_days
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''), (
                outcome_id, decision_id, user_id,
                data.get('actual_regret'),
                data.get('satisfaction_score'),
                data.get('outcome_notes'),
                json.dumps(data.get('surprises', [])),
                data.get('lessons_learned'),
                data.get('would_do_again'),
                data.get('time_since_decision_days')
            ))

            if data.get('actual_regret') is not None:
                cursor.execute(self._q('''
                    UPDATE decisions SET actual_regret = %s, outcome_recorded_at = %s WHERE id = %s
                '''), (data['actual_regret'], datetime.utcnow().isoformat(), decision_id))

            conn.commit()

        return {'outcome_id': outcome_id, 'decision_id': decision_id}

    def create_calendar_event(self, user_id: str, data: Dict) -> Dict:
        event_id = secrets.token_hex(8)
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO calendar_events (
                    id, user_id, title, description, event_type, start_time, end_time,
                    location, reminder_minutes, recurrence, color_id, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''), (
                event_id, user_id,
                data.get('title'),
                data.get('description', ''),
                data.get('event_type', 'general'),
                data.get('start_time'),
                data.get('end_time'),
                data.get('location', ''),
                json.dumps(data.get('reminder_minutes', [30, 1440])),
                data.get('recurrence'),
                data.get('color_id', '1'),
                now, now
            ))
            conn.commit()

        return self.get_calendar_event(user_id, event_id)

    def get_calendar_event(self, user_id: str, event_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM calendar_events WHERE id = %s AND user_id = %s'), (event_id, user_id))
            row = cursor.fetchone()
            if row:
                d = self._row_to_dict(cursor, row)
                if d.get('reminder_minutes'):
                    try:
                        d['reminder_minutes'] = json.loads(d['reminder_minutes'])
                    except:
                        pass
                return d
            return None

    def get_calendar_events(self, user_id: str, start_date: str = None, end_date: str = None,
                           event_type: str = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = self._q('SELECT * FROM calendar_events WHERE user_id = %s')
            params = [user_id]
            ph = "?" if self.backend == 'sqlite' else "%s"

            if start_date:
                query += f' AND start_time >= {ph}'
                params.append(start_date)

            if end_date:
                query += f' AND start_time <= {ph}'
                params.append(end_date)

            if event_type:
                query += f' AND event_type = {ph}'
                params.append(event_type)

            query += ' ORDER BY start_time ASC'

            cursor.execute(query, params)
            events = []
            for row in cursor.fetchall():
                d = self._row_to_dict(cursor, row)
                if d.get('reminder_minutes'):
                    try:
                        d['reminder_minutes'] = json.loads(d['reminder_minutes'])
                    except:
                        pass
                events.append(d)
            return events

    def update_calendar_event(self, user_id: str, event_id: str, data: Dict) -> bool:
        allowed_fields = ['title', 'description', 'event_type', 'start_time', 'end_time',
                         'location', 'reminder_minutes', 'recurrence', 'color_id', 'synced']

        updates = {}
        for k, v in data.items():
            if k in allowed_fields:
                if isinstance(v, list):
                    updates[k] = json.dumps(v)
                else:
                    updates[k] = v

        if not updates:
            return False

        updates['updated_at'] = datetime.utcnow().isoformat()
        set_clause, values = self._build_update_clause(updates, allowed_fields + ['updated_at'])

        if not set_clause:
            return False

        ph = "?" if self.backend == 'sqlite' else "%s"
        values.extend([event_id, user_id])
        query = "UPDATE calendar_events SET " + set_clause + f" WHERE id = {ph} AND user_id = {ph}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_calendar_event(self, user_id: str, event_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('DELETE FROM calendar_events WHERE id = %s AND user_id = %s'), (event_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def add_document(self, user_id: str, filename: str, content: str, doc_type: str = 'general') -> Dict:
        doc_id = secrets.token_hex(8)
        summary = f"Document about {filename} with {len(content)} characters of specialized knowledge."

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO knowledge_documents (
                    id, user_id, filename, content, doc_type, summary, char_count, size_bytes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            '''), (doc_id, user_id, filename, content, doc_type, summary, len(content), len(content.encode('utf-8'))))
            conn.commit()

        return {
            'id': doc_id,
            'filename': filename,
            'type': doc_type,
            'chars': len(content),
            'summary': summary,
            'added_at': datetime.utcnow().isoformat()
        }

    def get_documents(self, user_id: str) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                SELECT id, filename, doc_type as type, char_count as chars, summary, created_at as added_at
                FROM knowledge_documents WHERE user_id = %s ORDER BY created_at DESC
            '''), (user_id,))
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('DELETE FROM knowledge_documents WHERE id = %s AND user_id = %s'), (doc_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def save_conversation(self, user_id: str, title: str, messages: List[Dict]) -> Dict:
        conv_id = secrets.token_hex(8)
        preview = messages[0].get('content', '')[:50] if messages else ''

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO conversations (id, user_id, title, preview, messages)
                VALUES (%s, %s, %s, %s, %s)
            '''), (conv_id, user_id, title, preview, json.dumps(messages)))
            conn.commit()

        return {'id': conv_id, 'title': title, 'preview': preview}

    def get_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                SELECT id, title, preview, created_at as timestamp
                FROM conversations WHERE user_id = %s ORDER BY created_at DESC LIMIT %s
            '''), (user_id, limit))
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]

    def get_conversation(self, user_id: str, conv_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM conversations WHERE id = %s AND user_id = %s'), (conv_id, user_id))
            row = cursor.fetchone()
            if row:
                d = self._row_to_dict(cursor, row)
                d['messages'] = json.loads(d.get('messages', '[]'))
                return d
            return None

    def delete_conversation(self, user_id: str, conv_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('DELETE FROM conversations WHERE id = %s AND user_id = %s'), (conv_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def create_notification(self, user_id: str, title: str, message: str,
                           notification_type: str = 'info', action_url: str = None) -> Dict:
        notif_id = secrets.token_hex(8)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO notifications (id, user_id, title, message, notification_type, action_url)
                VALUES (%s, %s, %s, %s, %s, %s)
            '''), (notif_id, user_id, title, message, notification_type, action_url))
            conn.commit()

        return {'id': notif_id, 'title': title, 'message': message, 'type': notification_type}

    def get_notifications(self, user_id: str, unread_only: bool = False, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = self._q('SELECT * FROM notifications WHERE user_id = %s')
            params = [user_id]
            ph = "?" if self.backend == 'sqlite' else "%s"

            if unread_only:
                query += ' AND is_read = 0' if self.backend == 'sqlite' else ' AND is_read = FALSE'

            query += f' ORDER BY created_at DESC LIMIT {ph}'
            params.append(limit)

            cursor.execute(query, params)
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]

    def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            val = 1 if self.backend == 'sqlite' else True
            cursor.execute(self._q('UPDATE notifications SET is_read = %s WHERE id = %s AND user_id = %s'),
                          (val, notification_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def mark_all_notifications_read(self, user_id: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.backend == 'sqlite':
                cursor.execute('UPDATE notifications SET is_read = 1 WHERE user_id = ? AND is_read = 0', (user_id,))
            else:
                cursor.execute('UPDATE notifications SET is_read = TRUE WHERE user_id = %s AND is_read = FALSE', (user_id,))
            conn.commit()
            return cursor.rowcount

    def record_metric(self, user_id: str, metric_name: str, value: float, metadata: Dict = None):
        metric_id = secrets.token_hex(8)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO user_analytics (id, user_id, metric_name, metric_value, metadata)
                VALUES (%s, %s, %s, %s, %s)
            '''), (metric_id, user_id, metric_name, value, json.dumps(metadata or {})))
            conn.commit()

    def get_analytics_summary(self, user_id: str) -> Dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                SELECT COUNT(*) as total_decisions,
                       AVG(predicted_regret) as avg_predicted_regret,
                       AVG(actual_regret) as avg_actual_regret,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_decisions
                FROM decisions WHERE user_id = %s
            '''), (user_id,))
            row = cursor.fetchone()
            decision_stats = self._row_to_dict(cursor, row) or {}

            cursor.execute(self._q('''
                SELECT decision_type, COUNT(*) as count
                FROM decisions WHERE user_id = %s
                GROUP BY decision_type ORDER BY count DESC LIMIT 5
            '''), (user_id,))
            decision_types = [self._row_to_dict(cursor, row) for row in cursor.fetchall()]

            # Use backend-appropriate date query
            if self.backend == 'sqlite':
                cursor.execute('''
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM decisions WHERE user_id = ? AND created_at > datetime('now', '-30 days')
                    GROUP BY DATE(created_at) ORDER BY date DESC
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM decisions WHERE user_id = %s AND created_at > CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY DATE(created_at) ORDER BY date DESC
                ''', (user_id,))
            activity = [self._row_to_dict(cursor, row) for row in cursor.fetchall()]

            return {
                'decision_stats': decision_stats,
                'decision_types': decision_types,
                'recent_activity': activity
            }

    def record_ai_feedback(self, user_id: str, message_id: str, feedback_type: str,
                          rating: int = None, comment: str = None, context: Dict = None):
        feedback_id = secrets.token_hex(8)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                INSERT INTO ai_feedback (id, user_id, message_id, feedback_type, rating, comment, context)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''), (feedback_id, user_id, message_id, feedback_type, rating, comment, json.dumps(context or {})))
            conn.commit()

    def export_user_data(self, user_id: str) -> Dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                SELECT id, email, username, full_name, bio, created_at, preferences, theme
                FROM users WHERE id = %s
            '''), (user_id,))
            row = cursor.fetchone()
            user = self._row_to_dict(cursor, row) if row else {}

            decisions, _ = self.get_decisions(user_id, limit=10000)
            events = self.get_calendar_events(user_id)
            documents = self.get_documents(user_id)
            conversations = self.get_conversations(user_id, limit=10000)

            return {
                'exported_at': datetime.utcnow().isoformat(),
                'version': '1.0',
                'user': user,
                'decisions': decisions,
                'calendar_events': events,
                'documents': documents,
                'conversations': conversations
            }

    def import_user_data(self, user_id: str, data: Dict) -> Dict:
        imported = {'decisions': 0, 'events': 0, 'documents': 0}

        for decision in data.get('decisions', []):
            try:
                decision.pop('id', None)
                self.create_decision(user_id, decision)
                imported['decisions'] += 1
            except:
                pass

        for event in data.get('calendar_events', []):
            try:
                event.pop('id', None)
                self.create_calendar_event(user_id, event)
                imported['events'] += 1
            except:
                pass

        return imported

try:
    db_service = DatabaseService()
except Exception as e:
    import warnings
    warnings.warn(f"DatabaseService failed to initialise: {e}")
    db_service = None
