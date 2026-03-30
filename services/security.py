import os
import time
import hashlib
import secrets
import hmac
import re
import ipaddress
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from threading import Lock
from functools import wraps
import json
import sqlite3
import asyncio

logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.WARNING)


class SecurityConfig:

    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY", "")
    if not SECRET_KEY:
        SECRET_KEY = secrets.token_urlsafe(64)
        print("SECURITY WARNING: Using auto-generated SECRET_KEY. Set JWT_SECRET_KEY in environment for production!")

    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_RPM", "30"))
    RATE_LIMIT_REQUESTS_PER_HOUR: int = int(os.getenv("RATE_LIMIT_RPH", "500"))
    RATE_LIMIT_BURST: int = 5

    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15

    MAX_REQUEST_SIZE: int = 1 * 1024 * 1024
    MAX_JSON_DEPTH: int = 10
    MAX_STRING_LENGTH: int = 10000
    MAX_ARRAY_LENGTH: int = 100

    IP_BLACKLIST: Set[str] = set(os.getenv("IP_BLACKLIST", "").split(",")) - {""}
    IP_WHITELIST: Set[str] = set(os.getenv("IP_WHITELIST", "127.0.0.1,::1").split(","))

    ALLOWED_ORIGINS: List[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
    ] or ["http://localhost:8000", "http://127.0.0.1:8000"]

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    TRUSTED_PROXIES: set = set(
        p.strip() for p in os.getenv("TRUSTED_PROXIES", "127.0.0.1").split(",") if p.strip()
    )

    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")
    if not ENCRYPTION_KEY and not DEBUG:
        print("SECURITY ERROR: ENCRYPTION_KEY must be set in production! Data encryption is compromised.")

    BLOCKED_PATTERNS: List[str] = [
        r"<script",
        r"javascript:",
        r"on\w+\s*=",
        r"data:text/html",
        r"vbscript:",
        r"\.\./",
        r"\.\.\\",
        r";\s*--",
        r"/\*.*\*/",
        r"union\s+select",
        r"insert\s+into",
        r"drop\s+table",
        r"delete\s+from",
        r"exec\s*\(",
        r"eval\s*\(",
        r"\$\{.*\}",
        r"\{\{.*\}\}",
        r"__proto__",
        r"constructor",
    ]



class SecurityException(Exception):
    def __init__(self, message: str, code: str = "SECURITY_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)

class RateLimitExceeded(SecurityException):
    def __init__(self, retry_after: int = 60):
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.", "RATE_LIMIT_EXCEEDED")
        self.retry_after = retry_after

class BruteForceDetected(SecurityException):
    def __init__(self, lockout_until: datetime):
        super().__init__("Too many failed attempts. Account temporarily locked.", "BRUTE_FORCE_DETECTED")
        self.lockout_until = lockout_until

class MaliciousInputDetected(SecurityException):
    def __init__(self, input_type: str = "unknown"):
        super().__init__(f"Potentially malicious input detected: {input_type}", "MALICIOUS_INPUT")

class UnauthorizedAccess(SecurityException):
    def __init__(self, resource: str = ""):
        super().__init__(f"Unauthorized access{' to ' + resource if resource else ''}", "UNAUTHORIZED")

class IPBlocked(SecurityException):
    def __init__(self, ip: str):
        super().__init__(f"IP address blocked: {ip}", "IP_BLOCKED")



class InputValidator:

    _blocked_patterns = [re.compile(p, re.IGNORECASE) for p in SecurityConfig.BLOCKED_PATTERNS]

    @classmethod
    def sanitize_string(cls, text: str, max_length: int = None) -> str:
        if not text:
            return ""

        max_len = max_length or SecurityConfig.MAX_STRING_LENGTH
        text = text[:max_len]

        text = text.replace('\x00', '')

        text = ''.join(c for c in text if c in '\n\t' or (ord(c) >= 32 and ord(c) != 127))

        text = (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
        )

        return text.strip()

    @classmethod
    def validate_input(cls, text: str, field_name: str = "input") -> Tuple[bool, str]:
        if not text:
            return True, ""

        text_lower = text.lower()

        for pattern in cls._blocked_patterns:
            if pattern.search(text_lower):
                security_logger.warning(f"Blocked malicious input in {field_name}: pattern match")
                return False, f"Invalid characters in {field_name}"

        return True, ""

    @classmethod
    def validate_email(cls, email: str) -> bool:
        if not email or len(email) > 254:
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @classmethod
    def validate_username(cls, username: str) -> bool:
        if not username or len(username) < 3 or len(username) > 30:
            return False
        pattern = r'^[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, username))

    @classmethod
    def validate_password_strength(cls, password: str) -> Tuple[bool, str]:
        if len(password) < 12:
            return False, "Password must be at least 12 characters"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain lowercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain a number"
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain a special character"
        return True, ""

    @classmethod
    def validate_json_depth(cls, obj: Any, current_depth: int = 0) -> bool:
        if current_depth > SecurityConfig.MAX_JSON_DEPTH:
            return False

        if isinstance(obj, dict):
            if len(obj) > SecurityConfig.MAX_ARRAY_LENGTH:
                return False
            return all(cls.validate_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if len(obj) > SecurityConfig.MAX_ARRAY_LENGTH:
                return False
            return all(cls.validate_json_depth(item, current_depth + 1) for item in obj)

        return True

    @classmethod
    def sanitize_path(cls, path: str) -> str:
        path = path.replace('..', '')
        path = path.replace('~', '')
        path = re.sub(r'[^a-zA-Z0-9_\-.]', '', path)
        return path



class PasswordService:

    ITERATIONS = 310000
    SALT_LENGTH = 32
    KEY_LENGTH = 64

    @classmethod
    def hash_password(cls, password: str) -> str:
        salt = secrets.token_bytes(cls.SALT_LENGTH)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            cls.ITERATIONS,
            dklen=cls.KEY_LENGTH
        )
        return f"{cls.ITERATIONS}${salt.hex()}${key.hex()}"

    @classmethod
    def verify_password(cls, password: str, stored_hash: str) -> bool:
        try:
            iterations_str, salt_hex, key_hex = stored_hash.split('$')
            iterations = int(iterations_str)
            salt = bytes.fromhex(salt_hex)
            stored_key = bytes.fromhex(key_hex)

            computed_key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                iterations,
                dklen=len(stored_key)
            )

            return hmac.compare_digest(computed_key, stored_key)
        except (ValueError, AttributeError):
            return False



@dataclass
class RateLimitBucket:
    tokens: float
    last_update: float
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    minute_window_start: float = 0
    hour_window_start: float = 0
    blocked_until: float = 0

class HardenedRateLimiter:

    def __init__(
        self,
        requests_per_minute: int = None,
        requests_per_hour: int = None,
        burst_size: int = None
    ):
        self.rpm = requests_per_minute or SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE
        self.rph = requests_per_hour or SecurityConfig.RATE_LIMIT_REQUESTS_PER_HOUR
        self.burst = burst_size or SecurityConfig.RATE_LIMIT_BURST
        self.buckets: Dict[str, RateLimitBucket] = {}
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        self.lock = Lock()

    def _get_bucket(self, identifier: str) -> RateLimitBucket:
        now = time.time()
        if identifier not in self.buckets:
            self.buckets[identifier] = RateLimitBucket(
                tokens=float(self.burst),
                last_update=now,
                minute_window_start=now,
                hour_window_start=now
            )
        return self.buckets[identifier]

    def _refill_tokens(self, bucket: RateLimitBucket, now: float):
        elapsed = now - bucket.last_update
        refill_rate = self.rpm / 60.0
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * refill_rate)
        bucket.last_update = now

    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        now = time.time()

        with self.lock:
            bucket = self._get_bucket(identifier)

            if bucket.blocked_until > now:
                return False, {
                    "allowed": False,
                    "retry_after": int(bucket.blocked_until - now),
                    "reason": "rate_limit_exceeded"
                }

            self._refill_tokens(bucket, now)

            if now - bucket.minute_window_start > 60:
                bucket.requests_this_minute = 0
                bucket.minute_window_start = now

            if now - bucket.hour_window_start > 3600:
                bucket.requests_this_hour = 0
                bucket.hour_window_start = now

            if bucket.tokens < 1:
                self.suspicious_ips[identifier] += 1
                return False, {
                    "allowed": False,
                    "retry_after": 1,
                    "reason": "burst_limit"
                }

            if bucket.requests_this_minute >= self.rpm:
                bucket.blocked_until = now + 60
                self.suspicious_ips[identifier] += 1
                return False, {
                    "allowed": False,
                    "retry_after": 60,
                    "reason": "minute_limit"
                }

            if bucket.requests_this_hour >= self.rph:
                bucket.blocked_until = now + 3600
                return False, {
                    "allowed": False,
                    "retry_after": 3600,
                    "reason": "hour_limit"
                }

            bucket.tokens -= 1
            bucket.requests_this_minute += 1
            bucket.requests_this_hour += 1

            return True, {
                "allowed": True,
                "remaining_minute": self.rpm - bucket.requests_this_minute,
                "remaining_burst": int(bucket.tokens)
            }

    def is_suspicious(self, identifier: str, threshold: int = 10) -> bool:
        return self.suspicious_ips.get(identifier, 0) >= threshold

    def block_ip(self, identifier: str, duration_seconds: int = 3600):
        with self.lock:
            bucket = self._get_bucket(identifier)
            bucket.blocked_until = time.time() + duration_seconds

    def cleanup_old_buckets(self, max_age_hours: int = 24):
        now = time.time()
        cutoff = now - (max_age_hours * 3600)

        with self.lock:
            to_remove = [
                k for k, v in self.buckets.items()
                if v.last_update < cutoff
            ]
            for key in to_remove:
                del self.buckets[key]



@dataclass
class LoginAttempt:
    timestamp: float
    success: bool
    ip_address: str

class BruteForceProtector:

    def __init__(
        self,
        max_attempts: int = None,
        lockout_minutes: int = None
    ):
        self.max_attempts = max_attempts or SecurityConfig.MAX_FAILED_LOGIN_ATTEMPTS
        self.lockout_duration = (lockout_minutes or SecurityConfig.LOCKOUT_DURATION_MINUTES) * 60
        self.attempts: Dict[str, List[LoginAttempt]] = defaultdict(list)
        self.lockouts: Dict[str, float] = {}
        self.lock = Lock()

    def _cleanup_old_attempts(self, identifier: str, window_seconds: int = 900):
        now = time.time()
        cutoff = now - window_seconds
        self.attempts[identifier] = [
            a for a in self.attempts[identifier] if a.timestamp > cutoff
        ]

    def record_attempt(self, identifier: str, success: bool, ip_address: str):
        with self.lock:
            self._cleanup_old_attempts(identifier)

            self.attempts[identifier].append(LoginAttempt(
                timestamp=time.time(),
                success=success,
                ip_address=ip_address
            ))

            if not success:
                failed_count = sum(1 for a in self.attempts[identifier] if not a.success)
                if failed_count >= self.max_attempts:
                    self.lockouts[identifier] = time.time() + self.lockout_duration
                    security_logger.warning(
                        f"Account locked due to brute force: {identifier} from {ip_address}"
                    )

    def is_locked(self, identifier: str) -> Tuple[bool, Optional[float]]:
        with self.lock:
            if identifier in self.lockouts:
                lockout_until = self.lockouts[identifier]
                if time.time() < lockout_until:
                    return True, lockout_until
                else:
                    del self.lockouts[identifier]
            return False, None

    def get_remaining_attempts(self, identifier: str) -> int:
        with self.lock:
            self._cleanup_old_attempts(identifier)
            failed = sum(1 for a in self.attempts[identifier] if not a.success)
            return max(0, self.max_attempts - failed)



class IPManager:

    def __init__(self):
        self.blacklist: Set[str] = set(SecurityConfig.IP_BLACKLIST)
        self.whitelist: Set[str] = set(SecurityConfig.IP_WHITELIST)
        self.dynamic_blocks: Dict[str, float] = {}
        self.lock = Lock()

    def is_valid_ip(self, ip: str) -> bool:
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def is_private_ip(self, ip: str) -> bool:
        try:
            return ipaddress.ip_address(ip).is_private
        except ValueError:
            return False

    def is_blocked(self, ip: str) -> bool:
        if ip in self.blacklist:
            return True

        with self.lock:
            if ip in self.dynamic_blocks:
                if time.time() < self.dynamic_blocks[ip]:
                    return True
                else:
                    del self.dynamic_blocks[ip]

        return False

    def is_whitelisted(self, ip: str) -> bool:
        return ip in self.whitelist

    def block_ip(self, ip: str, duration_hours: int = 24):
        with self.lock:
            self.dynamic_blocks[ip] = time.time() + (duration_hours * 3600)
        security_logger.warning(f"IP blocked: {ip} for {duration_hours} hours")

    def unblock_ip(self, ip: str):
        with self.lock:
            self.dynamic_blocks.pop(ip, None)

    def add_to_blacklist(self, ip: str):
        self.blacklist.add(ip)
        security_logger.warning(f"IP added to permanent blacklist: {ip}")



@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: str
    resource: str
    action: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

class AuditLogger:

    def __init__(self, max_events: int = 10000):
        self.events: List[AuditEvent] = []
        self.max_events = max_events
        self.lock = Lock()

    def log(
        self,
        event_type: str,
        ip_address: str,
        resource: str,
        action: str,
        success: bool,
        user_id: str = None,
        details: Dict = None
    ):
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            success=success,
            details=details or {}
        )

        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

        log_msg = f"[AUDIT] {event_type}: {action} on {resource} by {user_id or 'anonymous'} from {ip_address} - {'SUCCESS' if success else 'FAILED'}"
        if success:
            security_logger.info(log_msg)
        else:
            security_logger.warning(log_msg)

    def get_events(
        self,
        event_type: str = None,
        user_id: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        with self.lock:
            events = self.events.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_suspicious_activity(self, ip_address: str = None, hours: int = 24) -> List[AuditEvent]:
        since = datetime.utcnow() - timedelta(hours=hours)
        with self.lock:
            events = [
                e for e in self.events
                if not e.success and e.timestamp >= since
                and (ip_address is None or e.ip_address == ip_address)
            ]
        return events



class SecurityHeaders:

    @staticmethod
    def get_headers(is_html: bool = False) -> Dict[str, str]:
        headers = {
            "X-Frame-Options": "DENY",

            "X-Content-Type-Options": "nosniff",

            "X-XSS-Protection": "1; mode=block",

            "Referrer-Policy": "strict-origin-when-cross-origin",

            "Permissions-Policy": "geolocation=(), microphone=(self), camera=(self), payment=()",

            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",

            "X-Powered-By": "",
            "Server": "",
        }

        if is_html:
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://d3js.org https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
                "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )

        if not SecurityConfig.DEBUG:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return headers



class CSRFProtector:

    TOKEN_LENGTH = 64
    TOKEN_LIFETIME_SECONDS = 3600

    def __init__(self):
        self.tokens: Dict[str, Tuple[str, float]] = {}
        self.lock = Lock()

    def generate_token(self, session_id: str) -> str:
        token = secrets.token_urlsafe(self.TOKEN_LENGTH)
        with self.lock:
            self.tokens[session_id] = (token, time.time())
        return token

    def validate_token(self, session_id: str, token: str) -> bool:
        with self.lock:
            if session_id not in self.tokens:
                return False

            stored_token, created_at = self.tokens[session_id]

            if time.time() - created_at > self.TOKEN_LIFETIME_SECONDS:
                del self.tokens[session_id]
                return False

            return hmac.compare_digest(token, stored_token)

    def cleanup_expired(self):
        now = time.time()
        with self.lock:
            expired = [
                sid for sid, (_, created_at) in self.tokens.items()
                if now - created_at > self.TOKEN_LIFETIME_SECONDS
            ]
            for sid in expired:
                del self.tokens[sid]



class RequestValidator:

    DANGEROUS_USER_AGENTS = [
        "sqlmap", "nikto", "nmap", "masscan", "dirbuster",
        "gobuster", "hydra", "burp", "zaproxy"
    ]

    @classmethod
    def validate_user_agent(cls, user_agent: str) -> bool:
        if not user_agent:
            return False
        ua_lower = user_agent.lower()
        return not any(tool in ua_lower for tool in cls.DANGEROUS_USER_AGENTS)

    @classmethod
    def validate_content_type(cls, content_type: str, expected: str = "application/json") -> bool:
        if not content_type:
            return False
        return expected in content_type.lower()

    @classmethod
    def validate_request_size(cls, content_length: int) -> bool:
        return content_length <= SecurityConfig.MAX_REQUEST_SIZE

    @classmethod
    def detect_request_smuggling(cls, headers: Dict[str, str]) -> bool:
        te = headers.get("transfer-encoding", "").lower()
        cl = headers.get("content-length", "")

        if te and cl:
            return True

        if "chunked" in te and te != "chunked":
            return True

        return False



@dataclass
class SecureUser:
    id: str
    username: str
    email: str
    password_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_admin: bool = False
    api_key: Optional[str] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_login_count: int = 0
    locked_until: Optional[datetime] = None

class HardenedAuthService:
    @staticmethod
    def _resolve_db_path():
        primary = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'career_ai_users.db')
        try:
            import sqlite3 as _sq
            _conn = _sq.connect(primary)
            _conn.execute("SELECT 1")
            _conn.close()
            return primary
        except Exception:
            fallback = os.path.join('/tmp', 'career_ai_users.db')
            print(f"[AuthService] Primary DB path unavailable, using fallback: {fallback}")
            return fallback

    _DB_PATH = None

    def __init__(self):
        if self._DB_PATH is None:
            self._DB_PATH = self._resolve_db_path()
        self.users: Dict[str, SecureUser] = {}
        self.sessions: Dict[str, Tuple[str, float]] = {}
        self.api_keys: Dict[str, str] = {}
        self.brute_force = BruteForceProtector()
        self.rate_limiter = HardenedRateLimiter()
        self.audit = AuditLogger()
        self.lock = Lock()

        os.makedirs(os.path.dirname(self._DB_PATH), exist_ok=True)
        self._init_user_db()
        self._load_users_from_db()
        self._create_admin_account()

    def _init_user_db(self):
        conn = sqlite3.connect(self._DB_PATH)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_users (
                    username TEXT PRIMARY KEY,
                    id TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    is_admin INTEGER DEFAULT 0,
                    api_key TEXT,
                    created_at TEXT,
                    last_login TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_users_from_db(self):
        conn = sqlite3.connect(self._DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM auth_users").fetchall()
            for row in rows:
                user = SecureUser(
                    id=row["id"],
                    username=row["username"],
                    email=row["email"],
                    password_hash=row["password_hash"],
                    is_active=bool(row["is_active"]),
                    is_admin=bool(row["is_admin"]),
                    api_key=row["api_key"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
                )
                self.users[row["username"]] = user
                if user.api_key:
                    self.api_keys[user.api_key] = user.id
            if rows:
                security_logger.info(f"Loaded {len(rows)} persisted user(s) from database")
        finally:
            conn.close()

    def _persist_user(self, user: SecureUser):
        conn = sqlite3.connect(self._DB_PATH)
        try:
            conn.execute("""
                INSERT INTO auth_users (username, id, email, password_hash, is_active, is_admin, api_key, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    is_active     = excluded.is_active,
                    api_key       = excluded.api_key,
                    last_login    = excluded.last_login
            """, (
                user.username, user.id, user.email, user.password_hash,
                int(user.is_active), int(user.is_admin),
                user.api_key,
                user.created_at.isoformat() if user.created_at else datetime.utcnow().isoformat(),
                user.last_login.isoformat() if user.last_login else None,
            ))
            conn.commit()
        finally:
            conn.close()

    def _create_admin_account(self):
        if "admin" in self.users:
            return

        admin_password = os.getenv("ADMIN_PASSWORD")
        if not admin_password:
            admin_password = secrets.token_urlsafe(32)
            admin_cred_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data', '.admin_credentials'
            )
            try:
                os.makedirs(os.path.dirname(admin_cred_path), exist_ok=True)
                with open(admin_cred_path, 'w') as f:
                    f.write(f"admin_password={admin_password}\n")
                os.chmod(admin_cred_path, 0o600)
                security_logger.warning(
                    f"Auto-generated admin password written to {admin_cred_path} (chmod 600). "
                    "Set ADMIN_PASSWORD environment variable in production!"
                )
            except Exception as e:
                security_logger.error(f"Could not write admin credentials file: {e}")
                print("SECURITY: Admin password auto-generated. Set ADMIN_PASSWORD env var. Check data/.admin_credentials")

        admin = SecureUser(
            id="admin_001",
            username="admin",
            email="admin@system.local",
            password_hash=PasswordService.hash_password(admin_password),
            is_admin=True,
            api_key=secrets.token_urlsafe(48)
        )
        self.users["admin"] = admin
        self.api_keys[admin.api_key] = admin.id
        self._persist_user(admin)

    def register(
        self,
        username: str,
        email: str,
        password: str,
        ip_address: str
    ) -> Tuple[Optional[SecureUser], str]:
        if not InputValidator.validate_username(username):
            return None, "Invalid username. Use 3-30 alphanumeric characters."

        if not InputValidator.validate_email(email):
            return None, "Invalid email format."

        valid, msg = InputValidator.validate_password_strength(password)
        if not valid:
            return None, msg

        with self.lock:
            if username.lower() in [u.username.lower() for u in self.users.values()]:
                return None, "Username already exists."

            if email.lower() in [u.email.lower() for u in self.users.values()]:
                return None, "Email already registered."

            user = SecureUser(
                id=f"user_{secrets.token_hex(16)}",
                username=username,
                email=email.lower(),
                password_hash=PasswordService.hash_password(password),
                api_key=secrets.token_urlsafe(48)
            )

            self.users[username] = user
            self.api_keys[user.api_key] = user.id
            self._persist_user(user)

        self.audit.log(
            event_type="AUTH",
            ip_address=ip_address,
            resource="user",
            action="register",
            success=True,
            user_id=user.id
        )

        return user, "Success"

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str
    ) -> Tuple[Optional[SecureUser], str]:
        allowed, info = self.rate_limiter.is_allowed(ip_address)
        if not allowed:
            return None, f"Too many requests. Retry after {info.get('retry_after', 60)} seconds."

        locked, lockout_until = self.brute_force.is_locked(username)
        if locked:
            remaining = int(lockout_until - time.time())
            return None, f"Account temporarily locked. Try again in {remaining} seconds."

        locked, lockout_until = self.brute_force.is_locked(ip_address)
        if locked:
            remaining = int(lockout_until - time.time())
            return None, f"Too many failed attempts from this IP. Try again in {remaining} seconds."

        user = self.users.get(username)
        if user and PasswordService.verify_password(password, user.password_hash):
            self.brute_force.record_attempt(username, True, ip_address)
            user.last_login = datetime.utcnow()
            user.failed_login_count = 0
            self._persist_user(user)

            self.audit.log(
                event_type="AUTH",
                ip_address=ip_address,
                resource="session",
                action="login",
                success=True,
                user_id=user.id
            )

            return user, "Success"

        self.brute_force.record_attempt(username, False, ip_address)
        self.brute_force.record_attempt(ip_address, False, ip_address)

        self.audit.log(
            event_type="AUTH",
            ip_address=ip_address,
            resource="session",
            action="login",
            success=False,
            details={"username": username}
        )

        remaining = self.brute_force.get_remaining_attempts(username)
        return None, f"Invalid credentials. {remaining} attempts remaining."

    def create_session(self, user_id: str) -> str:
        session_id = secrets.token_urlsafe(64)
        expires_at = time.time() + (SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60)

        with self.lock:
            self.sessions[session_id] = (user_id, expires_at)

        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        with self.lock:
            if session_id not in self.sessions:
                return None

            user_id, expires_at = self.sessions[session_id]
            if time.time() > expires_at:
                del self.sessions[session_id]
                return None

            return user_id

    def validate_api_key(self, api_key: str) -> Optional[str]:
        return self.api_keys.get(api_key)

    def get_user_by_id(self, user_id: str) -> Optional[SecureUser]:
        with self.lock:
            for user in self.users.values():
                if user.id == user_id:
                    return user
        return None

    def logout(self, session_id: str):
        with self.lock:
            self.sessions.pop(session_id, None)

    def cleanup_expired_sessions(self):
        now = time.time()
        with self.lock:
            expired = [
                sid for sid, (_, exp) in self.sessions.items()
                if now > exp
            ]
            for sid in expired:
                del self.sessions[sid]



class SecurityMiddlewareHelper:

    def __init__(self):
        self.rate_limiter = HardenedRateLimiter()
        self.ip_manager = IPManager()
        self.audit = AuditLogger()
        self.csrf = CSRFProtector()

    def check_request(
        self,
        ip_address: str,
        path: str,
        method: str,
        headers: Dict[str, str],
        user_agent: str = None
    ) -> Tuple[bool, Optional[str], int]:
        if self.ip_manager.is_blocked(ip_address):
            self.audit.log("SECURITY", ip_address, path, "blocked_ip", False)
            return False, "Access denied", 403

        if self.ip_manager.is_whitelisted(ip_address):
            return True, None, 200

        if user_agent and not RequestValidator.validate_user_agent(user_agent):
            self.audit.log("SECURITY", ip_address, path, "suspicious_user_agent", False)
            return False, "Access denied", 403

        if RequestValidator.detect_request_smuggling(headers):
            self.ip_manager.block_ip(ip_address, duration_hours=24)
            self.audit.log("SECURITY", ip_address, path, "request_smuggling_attempt", False)
            return False, "Malformed request", 400

        allowed, info = self.rate_limiter.is_allowed(ip_address)
        if not allowed:
            retry_after = info.get("retry_after", 60)
            self.audit.log("SECURITY", ip_address, path, "rate_limited", False)
            return False, f"Rate limit exceeded. Retry after {retry_after} seconds", 429

        return True, None, 200

    def get_client_ip(self, request_headers: Dict[str, str], direct_ip: str) -> str:
        trusted_proxies = SecurityConfig.TRUSTED_PROXIES
        if direct_ip in trusted_proxies:
            forwarded_for = request_headers.get("x-forwarded-for", "")
            if forwarded_for:
                ip = forwarded_for.split(",")[0].strip()
                if self.ip_manager.is_valid_ip(ip):
                    return ip

            real_ip = request_headers.get("x-real-ip", "")
            if real_ip and self.ip_manager.is_valid_ip(real_ip):
                return real_ip

        return direct_ip



_rate_limiter = None
_auth_service = None
_audit_logger = None
_ip_manager = None
_security_helper = None

def get_rate_limiter() -> HardenedRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = HardenedRateLimiter()
    return _rate_limiter

def get_auth_service() -> HardenedAuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = HardenedAuthService()
    return _auth_service

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def get_ip_manager() -> IPManager:
    global _ip_manager
    if _ip_manager is None:
        _ip_manager = IPManager()
    return _ip_manager

def get_security_helper() -> SecurityMiddlewareHelper:
    global _security_helper
    if _security_helper is None:
        _security_helper = SecurityMiddlewareHelper()
    return _security_helper


class LRUCache(OrderedDict):
    def __init__(self, max_size: int = 1000):
        super().__init__()
        self.max_size = max_size
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
            return None

    def put(self, key: str, value: Any):
        with self.lock:
            if key in self:
                self.move_to_end(key)
            self[key] = value
            if len(self) > self.max_size:
                self.popitem(last=False)


class CacheService:
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: LRUCache = LRUCache(max_size)
        self.expiry: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.lock = Lock()

    def _generate_key(self, *args, **kwargs) -> str:
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.expiry:
                if time.time() > self.expiry[key]:
                    self.cache.pop(key, None)
                    del self.expiry[key]
                    self.misses += 1
                    return None
            val = self.cache.get(key)
            if val is not None:
                self.hits += 1
                return val
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int = None):
        with self.lock:
            self.cache.put(key, value)
            self.expiry[key] = time.time() + (ttl or self.ttl)

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "size": len(self.cache)
        }

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.expiry.clear()


class MonitoringService:
    def __init__(self):
        self.requests: Dict[str, int] = defaultdict(int)
        self.errors: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.start = datetime.utcnow()
        self.lock = Lock()

    def record(self, endpoint: str, response_time: float = None, error: bool = False):
        with self.lock:
            self.requests[endpoint] += 1
            if error:
                self.errors[endpoint] += 1
            if response_time:
                self.response_times[endpoint].append(response_time)
                if len(self.response_times[endpoint]) > 1000:
                    self.response_times[endpoint] = self.response_times[endpoint][-500:]

    def get_metrics(self) -> Dict:
        with self.lock:
            total = sum(self.requests.values())
            total_errors = sum(self.errors.values())
            avg_response = {}
            for ep, times in self.response_times.items():
                if times:
                    avg_response[ep] = sum(times) / len(times)
            return {
                "uptime_seconds": (datetime.utcnow() - self.start).total_seconds(),
                "total_requests": total,
                "total_errors": total_errors,
                "error_rate": total_errors / total if total > 0 else 0,
                "endpoints": dict(self.requests),
                "avg_response_times": avg_response
            }


class LoadBalancer:
    def __init__(self, endpoints: List[str] = None):
        self.endpoints = endpoints or ["http://localhost:11434"]
        self.current_index = 0
        self.health: Dict[str, bool] = {ep: True for ep in self.endpoints}
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()

    def get_endpoint(self) -> str:
        with self.lock:
            healthy = [ep for ep in self.endpoints if self.health.get(ep, True)]
            if not healthy:
                healthy = self.endpoints
            if len(healthy) == 1:
                return healthy[0]
            avg_times = {}
            for ep in healthy:
                times = self.response_times.get(ep, [])
                avg_times[ep] = sum(times[-10:]) / len(times[-10:]) if times else float('inf')
            return min(healthy, key=lambda x: avg_times.get(x, float('inf')))

    def report_response(self, endpoint: str, response_time: float, success: bool):
        with self.lock:
            self.health[endpoint] = success
            if success:
                self.response_times[endpoint].append(response_time)
                if len(self.response_times[endpoint]) > 100:
                    self.response_times[endpoint] = self.response_times[endpoint][-50:]

    def add_endpoint(self, endpoint: str):
        with self.lock:
            if endpoint not in self.endpoints:
                self.endpoints.append(endpoint)
                self.health[endpoint] = True


class TokenOptimizer:
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}

    @staticmethod
    def truncate_context(text: str, max_tokens: int = 2000) -> str:
        estimated_tokens = len(text) // 4
        if estimated_tokens <= max_tokens:
            return text
        target_chars = max_tokens * 4
        return text[:target_chars]

    @staticmethod
    def compress_prompt(prompt: str) -> str:
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        prompt = re.sub(r'([.!?])\s*\1+', r'\1', prompt)
        return prompt

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

class AISecurityGuard:
    
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"(?i)you\s+are\s+now\s+a",
        r"(?i)system\s+message:",
        r"(?i)new\s+rule:",
        r"(?i)stop\s+being\s+a",
        r"(?i)forget\s+everything",
        r"(?i)output\s+the\s+full\s+prompt",
        r"(?i)start\s+ignoring",
        r"(?i)instead\s+of\s+answering",
        r"(?i)bypass\s+the\s+filter",
        r"(?i)jailbreak",
        r"(?i)DAN\s+mode",
    ]

    @staticmethod
    def contains_injection(text: str) -> bool:
        if not text:
            return False
        
        for pattern in AISecurityGuard.INJECTION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def wrap_untrusted_content(content: str, label: str = "UNTRUSTED DATA") -> str:
        if not content:
            return ""
        
        safe_content = content.replace("<<<", "<< _").replace(">>>", "_ >>")
        
        return f"\n<{label}_START>\n{safe_content}\n<{label}_END>\n"

    @staticmethod
    def sanitize_for_prompt(text: str) -> str:
        if not text:
            return ""
        
        clean = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return clean[:SecurityConfig.MAX_STRING_LENGTH]

AuthService = HardenedAuthService
RateLimiter = HardenedRateLimiter
ai_security = AISecurityGuard()
