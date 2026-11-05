"""
Verification script for JEE-RAG setup and environment sanity check.
Run this script to verify your .env configuration, dependencies, and external services.
"""
from __future__ import annotations

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_env_file() -> bool:
    print("[1/5] Checking environment configuration (.env)...")
    if not os.path.exists(".env"):
        print("  [-] .env file not found! Copy .env.example to .env and fill in your API keys.")
        return False

    from config import get_settings

    try:
        settings = get_settings()
        print("  [+] Required environment variables loaded successfully:")
        print(f"      - Groq Model:        {settings.groq_model}")
        print(f"      - Ollama Base URL:   {settings.ollama_base_url}")
        print(f"      - Ollama Model:      {settings.ollama_embed_model}")
        print(f"      - Qdrant URL:        {settings.qdrant_url}")
        print(f"      - Qdrant Collection: {settings.qdrant_collection}")
        return True
    except Exception as e:
        print(f"  [-] Error loading settings: {e}")
        return False


def check_dependencies() -> bool:
    print("\n[2/5] Checking Python package dependencies...")
    required = [
        "fastapi",
        "uvicorn",
        "groq",
        "qdrant_client",
        "httpx",
        "dotenv",
        "tavily",
        "pydantic",
    ]
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [+] {pkg}")
        except ImportError:
            print(f"  [-] Missing package: {pkg}. Run: pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_ollama() -> bool:
    print("\n[3/5] Checking Ollama embedding service...")
    try:
        import httpx
        from config import get_settings

        settings = get_settings()
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            print(f"  [+] Ollama is running at {settings.ollama_base_url}")
            target_model = settings.ollama_embed_model
            has_model = any(target_model in m for m in models)
            if has_model:
                print(f"  [+] Model '{target_model}' is available.")
            else:
                print(f"  [!] Model '{target_model}' not found in Ollama.")
                print(f"      Please run: ollama pull {target_model}")
            return True
        else:
            print(f"  [!] Ollama responded with status code: {r.status_code}")
            return False
    except Exception as e:
        print(f"  [!] Ollama connection skipped / not reachable: {e}")
        print("      Note: Ensure Ollama is running if you plan to run local embeddings.")
        return False


def check_qdrant() -> bool:
    print("\n[4/5] Checking Qdrant connection...")
    try:
        from config import get_settings
        from qdrant_client import QdrantClient

        settings = get_settings()
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5.0)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"  [+] Connected to Qdrant. Found {len(col_names)} collection(s): {col_names}")
        return True
    except Exception as e:
        print(f"  [!] Qdrant connection check failed or skipped: {e}")
        return False


def check_groq() -> bool:
    print("\n[5/5] Checking Groq API connection...")
    try:
        from config import get_settings
        from groq import Groq

        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)
        models = client.models.list()
        print(f"  [+] Groq API authenticated successfully ({len(models.data)} models available).")
        return True
    except Exception as e:
        print(f"  [!] Groq API check failed or skipped: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("JEE-RAG Project Configuration & Health Check")
    print("=" * 60)

    env_ok = check_env_file()
    deps_ok = check_dependencies()
    if not (env_ok and deps_ok):
        print("\n[-] Setup incomplete. Please resolve the issues above.")
        sys.exit(1)

    check_ollama()
    check_qdrant()
    check_groq()

    print("\n" + "=" * 60)
    print("Diagnostic check completed.")
    print("To start the server, run:")
    print("  python main.py")
    print("Or:")
    print("  uvicorn main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
