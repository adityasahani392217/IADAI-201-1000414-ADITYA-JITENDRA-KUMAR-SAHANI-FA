"""
core/security.py
================
Enterprise-grade client-side credential store and authentication manager.
Utilizes salted PBKDF2-HMAC-SHA256 password derivation to ensure no plaintext
passwords are ever stored on disk.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st


class SecureCredentialVault:
    """
    Manages user registration, credential authentication, and persistent local storage.
    Passwords are encrypted using PBKDF2-HMAC-SHA256 with 250,000 iterations and
    cryptographically unique per-user 16-byte salts.
    """

    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    PBKDF2_ITERATIONS = 250_000

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Create storage directory and seed demo account if file does not exist."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.storage_path.exists():
                # Seed default demo account: demo@safefall.ai / safefall123
                demo_salt = secrets.token_bytes(16)
                demo_hash = self._derive_hash("safefall123", demo_salt)
                initial_vault = {
                    "demo@safefall.ai": {
                        "name": "Alex Morgan",
                        "salt": demo_salt.hex(),
                        "hash": demo_hash,
                        "created": datetime.now().isoformat(timespec="seconds")
                    }
                }
                self.storage_path.write_text(json.dumps(initial_vault, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_vault(self) -> Dict[str, Dict[str, Any]]:
        """Load user accounts from persistent JSON file with session fallback."""
        try:
            if self.storage_path.exists():
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return dict(st.session_state.get("_cached_user_vault", {}))

    def _save_vault(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Commit user accounts to persistent disk and session cache."""
        st.session_state["_cached_user_vault"] = data
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def _derive_hash(cls, password: str, salt_bytes: bytes) -> str:
        """Derive cryptographic hex hash using PBKDF2-HMAC-SHA256."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_bytes,
            cls.PBKDF2_ITERATIONS
        ).hex()

    def register_account(
        self,
        full_name: str,
        email_addr: str,
        password: str,
        confirm_password: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate input parameters and register a new user account."""
        clean_name = full_name.strip()
        clean_email = email_addr.strip().lower()

        if len(clean_name) < 2:
            return False, "Please provide your full name (minimum 2 characters).", None

        if not self.EMAIL_PATTERN.match(clean_email):
            return False, "Please enter a valid email address format.", None

        if len(password) < 8:
            return False, "Password must contain at least 8 characters.", None

        if password != confirm_password:
            return False, "Passwords do not match. Please re-enter.", None

        vault = self._load_vault()
        if clean_email in vault:
            return False, "An account with this email address already exists. Please sign in.", None

        user_salt = secrets.token_bytes(16)
        password_hash = self._derive_hash(password, user_salt)

        vault[clean_email] = {
            "name": clean_name,
            "salt": user_salt.hex(),
            "hash": password_hash,
            "created": datetime.now().isoformat(timespec="seconds")
        }
        self._save_vault(vault)

        user_profile = {
            "name": clean_name,
            "email": clean_email,
            "guest": False,
            "created_at": vault[clean_email]["created"]
        }
        return True, "Account registered successfully.", user_profile

    def authenticate_credentials(
        self,
        email_addr: str,
        password: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Authenticate user against stored salted hashes using constant-time comparison."""
        clean_email = email_addr.strip().lower()
        vault = self._load_vault()

        user_record = vault.get(clean_email)
        if not user_record:
            return False, "No account located with this email address.", None

        salt_bytes = bytes.fromhex(user_record["salt"])
        computed_hash = self._derive_hash(password, salt_bytes)

        if not hmac.compare_digest(computed_hash, user_record["hash"]):
            return False, "Invalid password credentials provided.", None

        user_profile = {
            "name": user_record["name"],
            "email": clean_email,
            "guest": False,
            "created_at": user_record.get("created", "")
        }
        return True, "Authentication successful. Welcome back.", user_profile
