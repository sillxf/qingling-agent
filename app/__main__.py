"""Local-first command line entry point."""
from __future__ import annotations

import argparse
import ipaddress


def main() -> None:
    parser = argparse.ArgumentParser(description="Qingling security investigation workbench")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    from .config import settings
    try:
        local = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        local = args.host == "localhost"
    if not local and not settings.auth_enabled:
        parser.error("network binding requires QINGLING_AUTH_ENABLED=true and configured tokens")
    import uvicorn
    uvicorn.run("app.main:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
