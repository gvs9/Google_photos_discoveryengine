"""
smoke_test.py
-------------
Phase 0 exit-criteria verification script.

Checks that:
  1. All required third-party packages can be imported
  2. config/settings.py loads without errors
  3. All data directories exist (or are created)
  4. Required API keys are present in .env

Usage:
    python smoke_test.py
"""

import sys


def check_imports() -> list:
    """Try importing every major dependency. Return list of failures."""
    required = [
        "requests",
        "bs4",                    # beautifulsoup4
        "google_play_scraper",
        "app_store_scraper",
        "apify_client",           # Apify — Reddit scraping
        "tweepy",
        "langdetect",
        "spacy",
        "sentence_transformers",
        "groq",
        "chromadb",
        "hdbscan",
        "sklearn",                # scikit-learn
        "numpy",
        "pandas",
        "pyarrow",
        "datasketch",
        "tenacity",
        "tqdm",
        "pydantic_settings",
        "dotenv",                 # python-dotenv
        "streamlit",
        "jinja2",
    ]

    failures = []
    for module in required:
        try:
            __import__(module)
            print(f"  [OK]    {module}")
        except ImportError as exc:
            print(f"  [FAIL]  {module}  ->  {exc}")
            failures.append(module)
    return failures


def check_settings() -> list:
    """Load settings and validate required API keys."""
    try:
        from config.settings import settings
    except Exception as exc:
        print(f"  [FAIL]  config/settings.py failed to load: {exc}")
        return ["settings_load_error"]

    print("  [OK]    config/settings.py loaded")

    missing = settings.validate_required_keys()
    if missing:
        for key in missing:
            print(f"  [WARN]  Missing API key: {key}")
    else:
        print("  [OK]    All required API keys present")

    settings.ensure_data_dirs()
    print("  [OK]    Data directories verified / created")

    return missing


def main() -> None:
    print("\n=== Phase 0 Smoke Test ==========================================\n")

    print("[*] Checking package imports ...")
    import_failures = check_imports()

    print("\n[*] Checking configuration ...")
    key_failures = check_settings()

    print("\n=== Results =====================================================\n")

    if import_failures:
        print(f"[FAIL]    {len(import_failures)} package(s) missing: {import_failures}")
        print("          Run:  pip install -r requirements.txt")
    else:
        print("[OK]      All packages imported successfully")

    if key_failures and key_failures != ["settings_load_error"]:
        print(f"[WARN]    {len(key_failures)} API key(s) not set -- fill in your .env file")
        print("          Template: .env.example")
    elif not key_failures:
        print("[OK]      All API keys present")

    print()

    if import_failures:
        print("[FAIL]    Phase 0 NOT complete -- fix missing packages first.")
        sys.exit(1)
    elif key_failures:
        print("[PARTIAL] Phase 0 PARTIAL -- packages OK, but API keys need to be set.")
        sys.exit(0)
    else:
        print("[PASS]    Phase 0 COMPLETE -- ready to begin Phase 1.")
        sys.exit(0)


if __name__ == "__main__":
    main()
