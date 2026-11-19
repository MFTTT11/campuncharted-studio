#!/usr/bin/env python3
"""Local helper script to inspect/refresh YouTube OAuth token.json and upload GitHub Secrets safely.

Never prints secret values to the terminal. Run on your PC only once during setup:
    python scripts/export_github_secrets.py --token path/to/token.json --repo MFTTT11/campuncharted-studio
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def check_token_file(token_path: Path, client_secret_path: Path | None = None) -> dict:
    data = json.loads(token_path.read_text(encoding="utf-8"))
    if client_secret_path and client_secret_path.is_file():
        sec = json.loads(client_secret_path.read_text(encoding="utf-8"))
        inner = sec.get("installed") or sec.get("web") or sec
        data.setdefault("client_id", inner.get("client_id"))
        data.setdefault("client_secret", inner.get("client_secret"))
        data.setdefault("token_uri", inner.get("token_uri", "https://oauth2.googleapis.com/token"))

    missing = [k for k in ("refresh_token", "client_id", "client_secret") if not data.get(k)]
    scopes = set(data.get("scopes") or [])
    print(f"[CHECK] Token file: {token_path}")
    print(f"[CHECK] Has refresh_token: {'YES' if data.get('refresh_token') else 'NO (re-auth needed!)'}")
    print(f"[CHECK] Has client_id & client_secret: {'YES' if data.get('client_id') and data.get('client_secret') else 'NO'}")
    print(f"[CHECK] Scopes present: {sorted(scopes) if scopes else '(unknown / legacy format)'}")
    if missing:
        print(f"[WARN] Missing required fields for unattended 24/7 refresh: {missing}")
    return data


def reauthorize_youtube(client_secret_path: Path, out_token_path: Path) -> dict:
    """Run one-time local browser OAuth flow with access_type=offline to create a fresh token.json."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), REQUIRED_SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    out_token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"[OK] Saved fresh OAuth token to {out_token_path} (keep this file private, never commit it!).")
    return json.loads(creds.to_json())


def push_secret_via_gh(repo: str, secret_name: str, value: str) -> None:
    proc = subprocess.run(
        ["gh", "secret", "set", secret_name, "--repo", repo],
        input=value,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh secret set {secret_name} failed: {proc.stderr.strip()}")
    print(f"[OK] Secret '{secret_name}' uploaded securely to {repo}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or refresh YouTube OAuth token and push GitHub Secrets.")
    parser.add_argument("--token", type=Path, help="Path to existing token.json")
    parser.add_argument("--client-secret", type=Path, help="Optional path to client_secret.json")
    parser.add_argument(
        "--reauth",
        action="store_true",
        help="Run local browser login using --client-secret to generate a fresh offline refresh_token",
    )
    parser.add_argument("--repo", default="MFTTT11/campuncharted-studio", help="Target GitHub repo (owner/name)")
    parser.add_argument("--upload-gh", action="store_true", help="Upload YOUTUBE_TOKEN_JSON directly using gh CLI")
    args = parser.parse_args()

    if args.reauth:
        if not args.client_secret or not args.client_secret.is_file():
            print("ERROR: --reauth requires --client-secret path/to/client_secret.json", file=sys.stderr)
            return 1
        out_path = args.token or Path("token.json")
        token_data = reauthorize_youtube(args.client_secret, out_path)
    else:
        if not args.token or not args.token.is_file():
            print("ERROR: Provide --token path/to/token.json (or use --reauth --client-secret ...)", file=sys.stderr)
            return 1
        token_data = check_token_file(args.token, args.client_secret)

    if args.upload_gh:
        if not token_data.get("refresh_token"):
            print("ERROR: Refusing to upload token without refresh_token.", file=sys.stderr)
            return 1
        push_secret_via_gh(args.repo, "YOUTUBE_TOKEN_JSON", json.dumps(token_data, separators=(",", ":")))
    else:
        print(
            "\nNext step: Copy the contents of your token.json into GitHub:\n"
            f"  https://github.com/{args.repo}/settings/secrets/actions -> New repository secret -> YOUTUBE_TOKEN_JSON\n"
            "Or re-run this script with --upload-gh if GitHub CLI (gh) is logged in."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
