import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from jose import jwt, JWTError
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .database_service import db_service

SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
REFRESH_TOKEN_EXPIRE_DAYS = 30

security = HTTPBearer(auto_error=False)


class AuthService:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.failed_attempts = {}
    
    def create_access_token(self, user_id: str, extra_data: Dict = None) -> str:
        expires = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        payload = {
            "sub": user_id,
            "exp": expires,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        if extra_data:
            payload.update(extra_data)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        expires = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        payload = {
            "sub": user_id,
            "exp": expires,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != token_type:
                return None
            return payload
        except JWTError:
            return None
    
    def register_user(self, email: str, username: str, password: str, full_name: str = None) -> Dict:
        import re
        if len(password) < 12:
            raise HTTPException(status_code=400, detail="Password must be at least 12 characters")
        if not re.search(r'[A-Z]', password):
            raise HTTPException(status_code=400, detail="Password must contain an uppercase letter")
        if not re.search(r'[a-z]', password):
            raise HTTPException(status_code=400, detail="Password must contain a lowercase letter")
        if not re.search(r'\d', password):
            raise HTTPException(status_code=400, detail="Password must contain a number")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise HTTPException(status_code=400, detail="Password must contain a special character")
        
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
        
        existing = db_service.get_user_by_email(email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user = db_service.create_user(email, username, password, full_name)
        if not user:
            raise HTTPException(status_code=400, detail="Username already taken")
        
        access_token = self.create_access_token(user['id'])
        refresh_token = self.create_refresh_token(user['id'])
        session = db_service.create_session(user['id'])
        
        return {
            "user": {
                "id": user['id'],
                "email": user['email'],
                "username": user['username'],
                "full_name": user.get('full_name')
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
    
    def login(self, email: str, password: str, device_info: str = None, ip_address: str = None) -> Dict:
        if self._is_blocked(email):
            raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")
        
        user = db_service.authenticate_user(email, password)
        if not user:
            self._record_failed_attempt(email)
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        self._clear_failed_attempts(email)
        
        access_token = self.create_access_token(user['id'], {"email": user['email']})
        refresh_token = self.create_refresh_token(user['id'])
        
        session = db_service.create_session(user['id'], device_info, ip_address)
        
        return {
            "user": {
                "id": user['id'],
                "email": user['email'],
                "username": user['username'],
                "full_name": user.get('full_name'),
                "theme": user.get('theme', 'dark')
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_token": session['token'],
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
    
    def logout(self, token: str) -> bool:
        return db_service.invalidate_session(token)
    
    def refresh_tokens(self, refresh_token: str) -> Dict:
        payload = self.verify_token(refresh_token, "refresh")
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
        
        user_id = payload.get("sub")
        user = db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        new_access_token = self.create_access_token(user_id)
        new_refresh_token = self.create_refresh_token(user_id)
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
    
    def get_current_user(self, token: str) -> Optional[Dict]:
        payload = self.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        return db_service.get_user_by_id(user_id)
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
        
        success = db_service.change_password(user_id, old_password, new_password)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid current password")
        
        return True
    
    def request_password_reset(self, email: str) -> str:
        user = db_service.get_user_by_email(email)
        if not user:
            return secrets.token_urlsafe(32)
        
        reset_token = jwt.encode({
            "sub": user['id'],
            "exp": datetime.utcnow() + timedelta(hours=1),
            "type": "reset"
        }, self.secret_key, algorithm=self.algorithm)
        
        return reset_token
    
    def reset_password(self, reset_token: str, new_password: str) -> bool:
        try:
            payload = jwt.decode(reset_token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "reset":
                raise HTTPException(status_code=400, detail="Invalid reset token")
            
            user_id = payload.get("sub")
            return True
        except JWTError as e:
            if "expired" in str(e).lower():
                raise HTTPException(status_code=400, detail="Reset token expired")
            raise HTTPException(status_code=400, detail="Invalid reset token")
    
    def _is_blocked(self, email: str) -> bool:
        if email not in self.failed_attempts:
            return False
        
        attempts, last_attempt = self.failed_attempts[email]
        
        if attempts >= 5:
            if datetime.utcnow() - last_attempt < timedelta(minutes=15):
                return True
            else:
                del self.failed_attempts[email]
        
        return False
    
    def _record_failed_attempt(self, email: str):
        if email in self.failed_attempts:
            attempts, _ = self.failed_attempts[email]
            self.failed_attempts[email] = (attempts + 1, datetime.utcnow())
        else:
            self.failed_attempts[email] = (1, datetime.utcnow())
    
    def _clear_failed_attempts(self, email: str):
        if email in self.failed_attempts:
            del self.failed_attempts[email]


auth_service = AuthService()


async def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[Dict]:
    if not credentials:
        return None
    
    return auth_service.get_current_user(credentials.credentials)
