from flask import request, abort
import base64
from .config import Config

def require_basic_auth():
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Basic "):
        abort(401, description="Missing Basic auth")
    try:
        import base64 as b64
        decoded = b64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        abort(401, description="Malformed auth header")
    if username != Config.AUTH_USERNAME or password != Config.AUTH_PASSWORD:
        abort(401, description="Invalid credentials")
