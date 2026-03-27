"""
vault_core.py — Encryption, audit logging, backup, and recovery engine.
"""

import os
import io
import json
import stat
import hmac
import hashlib
import logging
import secrets
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

log = logging.getLogger("vault.core")
ph  = PasswordHasher()

VAULT_FILE   = Path(os.environ.get("VAULT_FILE",   "/etc/openclaw/vault.enc"))
AUDIT_FILE   = Path(os.environ.get("AUDIT_FILE",   "/etc/openclaw/audit.log"))
BACKUP_DIR   = Path(os.environ.get("BACKUP_DIR",   "/etc/openclaw/backups"))
META_FILE    = Path(os.environ.get("META_FILE",    "/etc/openclaw/vault.meta.json"))
TOKENS_FILE  = Path(os.environ.get("TOKENS_FILE",  "/etc/openclaw/tokens.enc"))
ROTATE_FILE  = Path(os.environ.get("ROTATE_FILE",  "/etc/openclaw/rotation.json"))
ENV_FILE     = Path(os.environ.get("VAULT_ENV",    "/etc/openclaw/vault.env"))
STALE_DAYS   = int(os.environ.get("STALE_DAYS", "90"))
ACL_FILE     = Path(os.environ.get("ACL_FILE",     "/etc/openclaw/vault_acl.json"))


def check_env_file_permissions() -> list:
    warnings = []
    if not ENV_FILE.exists():
        return warnings
    mode = ENV_FILE.stat().st_mode
    if mode & stat.S_IROTH:
        raise SystemExit(
            f"\n🚨 FATAL: {ENV_FILE} is world-readable (permissions: {oct(mode)}).\n"
            f"   Fix with:  chmod 600 {ENV_FILE}\n"
        )
    if mode & stat.S_IRGRP:
        warnings.append(f"⚠️  {ENV_FILE} is group-readable. Consider: chmod 600 {ENV_FILE}")
    if ENV_FILE.stat().st_uid != os.getuid():
        warnings.append(f"⚠️  {ENV_FILE} is owned by a different user.")
    if VAULT_FILE.exists():
        vmode = VAULT_FILE.stat().st_mode
        if vmode & (stat.S_IROTH | stat.S_IWOTH):
            raise SystemExit(f"\n🚨 FATAL: {VAULT_FILE} is world-readable/writable.\n   Fix with:  chmod 600 {VAULT_FILE}\n")
    return warnings


def load_vault(fernet: Fernet) -> dict:
    if not VAULT_FILE.exists():
        return {}
    return json.loads(fernet.decrypt(VAULT_FILE.read_bytes()))


def save_vault(fernet: Fernet, data: dict) -> None:
    VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = VAULT_FILE.with_suffix(".tmp")
    tmp.write_bytes(fernet.encrypt(json.dumps(data).encode()))
    tmp.chmod(0o600)
    tmp.replace(VAULT_FILE)


def load_rotation() -> dict:
    if not ROTATE_FILE.exists():
        return {}
    return json.loads(ROTATE_FILE.read_text())


def save_rotation(data: dict) -> None:
    ROTATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROTATE_FILE.write_text(json.dumps(data, indent=2))
    ROTATE_FILE.chmod(0o600)


def track_key_write(name: str, is_new: bool) -> None:
    rotation = load_rotation()
    now = datetime.now(timezone.utc).isoformat()
    if name not in rotation or is_new:
        rotation[name] = {"created_at": now, "updated_at": now}
    else:
        rotation[name]["updated_at"] = now
    save_rotation(rotation)


def track_key_delete(name: str) -> None:
    rotation = load_rotation()
    rotation.pop(name, None)
    save_rotation(rotation)


def get_stale_keys(vault_keys: list) -> list:
    if STALE_DAYS == 0:
        return []
    rotation = load_rotation()
    stale    = []
    cutoff   = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
    for name in vault_keys:
        entry = rotation.get(name)
        if not entry:
            stale.append({"name": name, "days_old": None, "reason": "no rotation record"})
            continue
        updated = datetime.fromisoformat(entry["updated_at"])
        if updated < cutoff:
            days = (datetime.now(timezone.utc) - updated).days
            stale.append({"name": name, "days_old": days, "reason": f"not rotated in {days} days"})
    return stale


def get_key_ages(vault_keys: list) -> dict:
    rotation = load_rotation()
    result   = {}
    for name in vault_keys:
        entry = rotation.get(name)
        if entry:
            updated = datetime.fromisoformat(entry["updated_at"])
            days    = (datetime.now(timezone.utc) - updated).days
            result[name] = {"updated_at": entry["updated_at"], "created_at": entry["created_at"],
                            "days_since_rotation": days, "stale": STALE_DAYS > 0 and days >= STALE_DAYS}
        else:
            result[name] = {"updated_at": None, "created_at": None, "days_since_rotation": None, "stale": False}
    return result


def load_tokens(fernet: Fernet) -> dict:
    if not TOKENS_FILE.exists():
        return {}
    return json.loads(fernet.decrypt(TOKENS_FILE.read_bytes()))


def save_tokens(fernet: Fernet, data: dict) -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKENS_FILE.with_suffix(".tmp")
    tmp.write_bytes(fernet.encrypt(json.dumps(data).encode()))
    tmp.chmod(0o600)
    tmp.replace(TOKENS_FILE)


def mint_token(fernet: Fernet, label: str, scope: list, allowed_ips: list) -> str:
    raw   = secrets.token_urlsafe(48)
    thash = hashlib.sha256(raw.encode()).hexdigest()
    tokens = load_tokens(fernet)
    tokens[thash] = {"label": label, "scope": scope, "allowed_ips": allowed_ips,
                     "created_at": datetime.now(timezone.utc).isoformat(), "last_used": None, "use_count": 0}
    save_tokens(fernet, tokens)
    log.info("Machine token minted: %s", label)
    return raw


def revoke_token(fernet: Fernet, token_hash: str) -> bool:
    tokens = load_tokens(fernet)
    match = next((k for k in tokens if k == token_hash or k.startswith(token_hash.rstrip("."))), None)
    if not match:
        return False
    tokens.pop(match)
    save_tokens(fernet, tokens)
    return True


def verify_machine_token(fernet: Fernet, raw_token: str, client_ip: str, key_name: str):
    thash  = hashlib.sha256(raw_token.encode()).hexdigest()
    tokens = load_tokens(fernet)
    entry  = tokens.get(thash)
    if not entry:
        return False, "invalid token"
    allowed_ips = entry.get("allowed_ips", [])
    if allowed_ips and client_ip not in allowed_ips:
        return False, f"ip {client_ip} not in allowlist"
    scope = entry.get("scope", ["*"])
    if scope != ["*"] and key_name not in scope:
        return False, f"key '{key_name}' not in token scope"
    entry["last_used"] = datetime.now(timezone.utc).isoformat()
    entry["use_count"] = entry.get("use_count", 0) + 1
    tokens[thash] = entry
    save_tokens(fernet, tokens)
    return True, "ok"


def load_meta() -> dict:
    if not META_FILE.exists():
        return {}
    return json.loads(META_FILE.read_text())


def save_meta(data: dict) -> None:
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(data, indent=2))
    META_FILE.chmod(0o600)


def set_breakglass(passphrase: str, fernet: Fernet) -> str:
    meta = load_meta()
    meta["breakglass_hash"] = ph.hash(passphrase)
    save_meta(meta)
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.backends import default_backend
    import base64
    salt = secrets.token_bytes(16)
    kdf  = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1, backend=default_backend())
    key  = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    recovery_fernet = Fernet(key)
    vault = load_vault(fernet)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    recovery_path = BACKUP_DIR / "recovery.enc"
    recovery_path.write_bytes(salt + recovery_fernet.encrypt(json.dumps(vault).encode()))
    recovery_path.chmod(0o600)
    return str(recovery_path)


def verify_breakglass(passphrase: str) -> bool:
    meta = load_meta()
    stored = meta.get("breakglass_hash")
    if not stored:
        return False
    try:
        ph.verify(stored, passphrase)
        return True
    except VerifyMismatchError:
        return False


def recover_vault(passphrase: str) -> Optional[dict]:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.backends import default_backend
    import base64
    recovery_path = BACKUP_DIR / "recovery.enc"
    if not recovery_path.exists():
        return None
    raw  = recovery_path.read_bytes()
    salt = raw[:16]
    data = raw[16:]
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1, backend=default_backend())
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    try:
        return json.loads(Fernet(key).decrypt(data))
    except Exception:
        return None


def setup_totp(issuer: str = "OpenClaw Vault"):
    import pyotp
    secret = pyotp.random_base32()
    totp   = pyotp.TOTP(secret)
    url    = totp.provisioning_uri(name="admin", issuer_name=issuer)
    meta   = load_meta()
    meta["totp_secret"]  = secret
    meta["totp_enabled"] = True
    save_meta(meta)
    return secret, url


def verify_totp(code: str) -> bool:
    meta   = load_meta()
    secret = meta.get("totp_secret")
    if not secret or not meta.get("totp_enabled"):
        return True
    import pyotp
    return pyotp.TOTP(secret).verify(code, valid_window=1)


def totp_enabled() -> bool:
    return load_meta().get("totp_enabled", False)


def write_backup(fernet: Fernet) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = BACKUP_DIR / f"vault_{ts}.enc"
    shutil.copy2(VAULT_FILE, dest)
    dest.chmod(0o600)
    backups = sorted(BACKUP_DIR.glob("vault_*.enc"))
    for old in backups[:-10]:
        old.unlink()
    return dest


def list_backups() -> list:
    if not BACKUP_DIR.exists():
        return []
    backups = sorted(BACKUP_DIR.glob("vault_*.enc"), reverse=True)
    return [{"filename": b.name, "size": b.stat().st_size, "created": b.stat().st_mtime} for b in backups]


def restore_backup(fernet: Fernet, filename: str) -> bool:
    filename = Path(filename).name
    src = BACKUP_DIR / filename
    if not src.exists() or not filename.startswith("vault_"):
        return False
    try:
        json.loads(fernet.decrypt(src.read_bytes()))
    except Exception:
        return False
    write_backup(fernet)
    shutil.copy2(src, VAULT_FILE)
    VAULT_FILE.chmod(0o600)
    return True


def audit(event: str, actor: str, ip: str, detail: str = "", success: bool = True) -> None:
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
             "actor": actor, "ip": ip, "detail": detail, "success": success}
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit(limit: int = 200) -> list:
    if not AUDIT_FILE.exists():
        return []
    lines = AUDIT_FILE.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-limit:]][::-1]


# ── ACL — Per-user key access control ─────────────────────────────────────

# ACL format:
# {
#   "users": {
#     "derek": {
#       "permissions": { "OPENROUTER_*": "rw", "HA_TOKEN": "r" },
#       "created_at": "...", "created_by": "root"
#     }
#   }
# }
#
# Permission values: "r" = read (list + view), "w" = write (add/rotate), "rw" = both
# Key patterns: exact match or trailing wildcard ("OPENROUTER_*")
# Root always has full access — ACL is only enforced for non-root callers.

import fnmatch
import pwd


def load_acl() -> dict:
    """Load the ACL file. Returns {"users": {...}} or empty structure."""
    if not ACL_FILE.exists():
        return {"users": {}}
    try:
        data = json.loads(ACL_FILE.read_text())
        if "users" not in data:
            data["users"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return {"users": {}}


def save_acl(data: dict) -> None:
    """Save ACL file with restricted permissions."""
    ACL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ACL_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.chmod(0o600)
    tmp.replace(ACL_FILE)


def get_calling_user() -> str:
    """Get the actual calling user (handles sudo)."""
    sudo_user = os.environ.get("SUDO_USER", "")
    if sudo_user:
        return sudo_user
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return str(os.getuid())


def is_root_caller() -> bool:
    """True if the effective user is root AND not sudo-ing from another user."""
    return os.getuid() == 0 and not os.environ.get("SUDO_USER", "")


def acl_grant(username: str, key_pattern: str, permission: str, granted_by: str = "root") -> None:
    """Grant a user access to keys matching pattern.

    Args:
        username:    Linux username to grant access to
        key_pattern: Key name or glob pattern (e.g. "OPENROUTER_*", "*")
        permission:  "r" (read/view), "w" (write/rotate), "rw" (both)
        granted_by:  Who issued the grant (for audit trail)
    """
    if permission not in ("r", "w", "rw"):
        raise ValueError(f"Invalid permission '{permission}' — use 'r', 'w', or 'rw'")

    acl = load_acl()
    if username not in acl["users"]:
        acl["users"][username] = {
            "permissions": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": granted_by,
        }
    acl["users"][username]["permissions"][key_pattern] = permission
    save_acl(acl)
    audit("acl_grant", granted_by, "localhost",
          f"user={username} pattern={key_pattern} perm={permission}")


def acl_revoke(username: str, key_pattern: str = None) -> bool:
    """Revoke a user's access. If key_pattern is None, remove the user entirely."""
    acl = load_acl()
    if username not in acl["users"]:
        return False
    if key_pattern is None:
        del acl["users"][username]
    else:
        acl["users"][username]["permissions"].pop(key_pattern, None)
        if not acl["users"][username]["permissions"]:
            del acl["users"][username]
    save_acl(acl)
    detail = f"user={username}" + (f" pattern={key_pattern}" if key_pattern else " (all)")
    audit("acl_revoke", get_calling_user(), "localhost", detail)
    return True


def acl_check(username: str, key_name: str, need: str = "r") -> bool:
    """Check if a user has the required permission for a specific key.

    Args:
        username:  Linux username
        key_name:  The actual key name (e.g. "OPENROUTER_API_KEY")
        need:      "r" for read/view, "w" for write/rotate
    Returns:
        True if access is allowed
    """
    acl = load_acl()
    user_entry = acl["users"].get(username)
    if not user_entry:
        return False

    for pattern, perm in user_entry["permissions"].items():
        if fnmatch.fnmatch(key_name, pattern) or fnmatch.fnmatch(key_name.upper(), pattern.upper()):
            if need == "r" and perm in ("r", "rw"):
                return True
            if need == "w" and perm in ("w", "rw"):
                return True
    return False


def acl_filter_keys(username: str, vault_keys: list, need: str = "r") -> list:
    """Filter a list of key names to only those the user can access."""
    return [k for k in vault_keys if acl_check(username, k, need)]


def acl_list() -> dict:
    """Return the full ACL for display."""
    return load_acl()["users"]
