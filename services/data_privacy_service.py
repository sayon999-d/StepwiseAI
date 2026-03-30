from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import base64
import os
import zipfile
import io
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class DataCategory(str, Enum):
    PROFILE = "profile"
    DECISIONS = "decisions"
    JOURNAL = "journal"
    ANALYTICS = "analytics"
    BIASES = "biases"
    GOALS = "goals"
    PREFERENCES = "preferences"

class ConsentType(str, Enum):
    DATA_COLLECTION = "data_collection"
    ANALYTICS = "analytics"
    GLOBAL_INSIGHTS = "global_insights"
    MARKETING = "marketing"

@dataclass
class UserConsent:
    user_id: str
    consent_type: ConsentType
    granted: bool
    granted_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    ip_address: str = ""

@dataclass
class DataExportRequest:
    id: str
    user_id: str
    categories: List[DataCategory]
    status: str
    requested_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None

@dataclass
class DeletionRequest:
    id: str
    user_id: str
    status: str
    reason: Optional[str]
    requested_at: datetime
    scheduled_at: datetime
    completed_at: Optional[datetime] = None
    data_deleted: List[str] = field(default_factory=list)

class EncryptionService:
    def __init__(self, master_key: str = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            env_key = os.getenv("ENCRYPTION_KEY", "")
            if env_key:
                self.master_key = env_key.encode()
            else:
                is_debug = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
                if not is_debug:
                    raise RuntimeError(
                        "ENCRYPTION_KEY environment variable MUST be set in production. "
                        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                    )
                import warnings
                warnings.warn(
                    "Using insecure default encryption key. Set ENCRYPTION_KEY env var for production!",
                    stacklevel=2,
                )
                self.master_key = b"INSECURE_DEV_KEY_DO_NOT_USE_IN_PRODUCTION"

        self._fernet = None

    def _get_fernet(self, salt: bytes = None) -> Fernet:
        if salt is None:
            salt = b"default_salt_v1"

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)

    def encrypt_data(self, data: str, user_salt: str = None) -> str:
        salt = (user_salt or "default").encode()[:16].ljust(16, b'\0')
        fernet = self._get_fernet(salt)
        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str, user_salt: str = None) -> str:
        salt = (user_salt or "default").encode()[:16].ljust(16, b'\0')
        fernet = self._get_fernet(salt)
        decrypted = fernet.decrypt(base64.urlsafe_b64decode(encrypted_data))
        return decrypted.decode()

    def hash_pii(self, data: str) -> str:
        return hashlib.sha256(f"{data}{self.master_key.decode()}".encode()).hexdigest()

    def anonymize_user_id(self, user_id: str) -> str:
        return f"anon_{hashlib.sha256(f'{user_id}_anonymized'.encode()).hexdigest()[:12]}"

class DataPrivacyService:
    RETENTION_PERIODS = {
        DataCategory.PROFILE: 365 * 3,
        DataCategory.DECISIONS: 365 * 5,
        DataCategory.JOURNAL: 365 * 5,
        DataCategory.ANALYTICS: 365 * 2,
        DataCategory.BIASES: 365 * 2,
        DataCategory.GOALS: 365 * 3,
        DataCategory.PREFERENCES: 365 * 3,
    }

    def __init__(self):
        self.encryption = EncryptionService()
        self.user_consents: Dict[str, Dict[ConsentType, UserConsent]] = {}
        self.export_requests: Dict[str, DataExportRequest] = {}
        self.deletion_requests: Dict[str, DeletionRequest] = {}
        self.data_access_log: List[Dict] = []

    def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        ip_address: str = ""
    ) -> UserConsent:
        consent = UserConsent(
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            granted_at=datetime.utcnow() if granted else None,
            revoked_at=None if granted else datetime.utcnow(),
            ip_address=ip_address
        )

        if user_id not in self.user_consents:
            self.user_consents[user_id] = {}

        self.user_consents[user_id][consent_type] = consent

        self._log_access(user_id, "consent_update", f"{consent_type.value}: {granted}")

        return consent

    def check_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        if user_id not in self.user_consents:
            return False
        if consent_type not in self.user_consents[user_id]:
            return False
        return self.user_consents[user_id][consent_type].granted

    def get_user_consents(self, user_id: str) -> Dict[str, bool]:
        if user_id not in self.user_consents:
            return {ct.value: False for ct in ConsentType}

        return {
            ct.value: self.user_consents[user_id].get(ct, UserConsent(user_id, ct, False)).granted
            for ct in ConsentType
        }

    def request_data_export(
        self,
        user_id: str,
        categories: List[DataCategory] = None
    ) -> DataExportRequest:
        if categories is None:
            categories = list(DataCategory)

        request_id = f"export_{user_id}_{datetime.utcnow().timestamp()}"

        request = DataExportRequest(
            id=request_id,
            user_id=user_id,
            categories=categories,
            status="pending",
            requested_at=datetime.utcnow()
        )

        self.export_requests[request_id] = request
        self._log_access(user_id, "export_request", f"Categories: {[c.value for c in categories]}")

        return request

    async def process_data_export(
        self,
        request_id: str,
        data_sources: Dict[str, Any]
    ) -> DataExportRequest:
        if request_id not in self.export_requests:
            return None

        request = self.export_requests[request_id]
        request.status = "processing"

        export_data = {
            "export_info": {
                "user_id": request.user_id,
                "exported_at": datetime.utcnow().isoformat(),
                "categories": [c.value for c in request.categories],
                "request_id": request_id
            },
            "data": {}
        }

        for category in request.categories:
            if category.value in data_sources:
                export_data["data"][category.value] = data_sources[category.value]

        json_data = json.dumps(export_data, indent=2, default=str)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("user_data.json", json_data)
            zip_file.writestr("README.txt", self._generate_export_readme())

        request.status = "completed"
        request.completed_at = datetime.utcnow()
        request.expires_at = datetime.utcnow() + timedelta(days=7)
        request.download_url = f"/api/privacy/download/{request_id}"

        self._log_access(request.user_id, "export_completed", request_id)

        return request

    def _generate_export_readme(self) -> str:
        return """
DATA EXPORT README
==================

This archive contains your personal data exported from Career Decision Regret AI.

Files included:
- user_data.json: All your personal data in JSON format

Categories:
- profile: Your user profile and preferences
- decisions: Your decision history and analysis
- journal: Your journal entries
- analytics: Your analytics and insights data
- biases: Detected cognitive biases
- goals: Your career goals and progress

This data is provided in compliance with GDPR Article 20 (Right to data portability).

For questions, contact: privacy@example.com
"""

    def request_account_deletion(
        self,
        user_id: str,
        reason: str = None
    ) -> DeletionRequest:
        request_id = f"delete_{user_id}_{datetime.utcnow().timestamp()}"

        request = DeletionRequest(
            id=request_id,
            user_id=user_id,
            status="pending",
            reason=reason,
            requested_at=datetime.utcnow(),
            scheduled_at=datetime.utcnow() + timedelta(days=30)
        )

        self.deletion_requests[request_id] = request
        self._log_access(user_id, "deletion_request", f"Scheduled for: {request.scheduled_at.isoformat()}")

        return request

    def cancel_deletion_request(self, request_id: str, user_id: str) -> bool:
        if request_id not in self.deletion_requests:
            return False

        request = self.deletion_requests[request_id]
        if request.user_id != user_id:
            return False
        if request.status != "pending":
            return False

        request.status = "cancelled"
        self._log_access(user_id, "deletion_cancelled", request_id)

        return True

    async def execute_deletion(
        self,
        request_id: str,
        data_deleters: Dict[str, callable]
    ) -> DeletionRequest:
        if request_id not in self.deletion_requests:
            return None

        request = self.deletion_requests[request_id]

        if request.status != "pending":
            return request

        if datetime.utcnow() < request.scheduled_at:
            return request

        request.status = "executing"

        for category, deleter in data_deleters.items():
            try:
                await deleter(request.user_id)
                request.data_deleted.append(category)
            except Exception as e:
                print(f"Error deleting {category}: {e}")

        request.status = "completed"
        request.completed_at = datetime.utcnow()

        self._log_access(request.user_id, "deletion_completed", f"Deleted: {request.data_deleted}")

        return request

    def anonymize_data_for_global_db(self, user_data: Dict) -> Dict:
        anonymized = {}

        pii_fields = ["user_id", "name", "email", "phone", "address"]

        for key, value in user_data.items():
            if key in pii_fields:
                if key == "user_id":
                    anonymized[key] = self.encryption.anonymize_user_id(str(value))
                else:
                    continue
            elif isinstance(value, dict):
                anonymized[key] = self.anonymize_data_for_global_db(value)
            elif isinstance(value, list):
                anonymized[key] = [
                    self.anonymize_data_for_global_db(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                anonymized[key] = value

        return anonymized

    def encrypt_sensitive_entry(self, user_id: str, entry_data: Dict) -> Dict:
        sensitive_fields = ["journal_content", "personal_notes", "private_thoughts"]

        encrypted = entry_data.copy()

        for field in sensitive_fields:
            if field in encrypted and encrypted[field]:
                encrypted[field] = self.encryption.encrypt_data(
                    str(encrypted[field]),
                    user_id
                )
                encrypted[f"{field}_encrypted"] = True

        return encrypted

    def decrypt_sensitive_entry(self, user_id: str, entry_data: Dict) -> Dict:
        decrypted = entry_data.copy()

        for key in list(decrypted.keys()):
            if key.endswith("_encrypted") and decrypted[key]:
                original_field = key.replace("_encrypted", "")
                if original_field in decrypted:
                    try:
                        decrypted[original_field] = self.encryption.decrypt_data(
                            decrypted[original_field],
                            user_id
                        )
                    except Exception:
                        pass
                del decrypted[key]

        return decrypted

    def _log_access(self, user_id: str, action: str, details: str):
        self.data_access_log.append({
            "user_id": self.encryption.hash_pii(user_id),
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })

        if len(self.data_access_log) > 10000:
            self.data_access_log = self.data_access_log[-5000:]

    def get_access_log(self, user_id: str, limit: int = 100) -> List[Dict]:
        hashed_id = self.encryption.hash_pii(user_id)

        user_logs = [
            log for log in self.data_access_log
            if log["user_id"] == hashed_id
        ]

        return user_logs[-limit:]

    def get_privacy_dashboard(self, user_id: str) -> Dict[str, Any]:
        consents = self.get_user_consents(user_id)

        pending_exports = [
            r for r in self.export_requests.values()
            if r.user_id == user_id and r.status in ["pending", "processing"]
        ]

        pending_deletions = [
            r for r in self.deletion_requests.values()
            if r.user_id == user_id and r.status == "pending"
        ]

        recent_access = self.get_access_log(user_id, limit=10)

        return {
            "user_id": user_id,
            "consents": consents,
            "data_retention": {
                category.value: f"{days} days"
                for category, days in self.RETENTION_PERIODS.items()
            },
            "pending_export_requests": len(pending_exports),
            "pending_deletion_requests": len(pending_deletions),
            "scheduled_deletion": pending_deletions[0].scheduled_at.isoformat() if pending_deletions else None,
            "recent_data_access": recent_access,
            "rights": {
                "access": "You can request a copy of all your data",
                "rectification": "You can update your personal information",
                "erasure": "You can request deletion of your account",
                "portability": "You can export your data in a portable format",
                "objection": "You can opt out of data processing for specific purposes"
            }
        }

data_privacy_service = DataPrivacyService()
