"""
Shared API key resolver for NV_Comfy_Utils nodes.

Resolves API keys with this precedence:
  1. Explicit value from node input (override)
  2. Environment variable (GEMINI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY)
  3. .env file in the NV_Comfy_Utils root directory

Usage:
    from .api_keys import resolve_api_key

    key = resolve_api_key(api_key_input, provider="gemini")
"""

import os

# ---------------------------------------------------------------------------
# .env loader (minimal, no python-dotenv dependency)
# ---------------------------------------------------------------------------

_ENV_LOADED = False


def _load_dotenv_once():
    """Load .env file from NV_Comfy_Utils root, once per process."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    # NV_Comfy_Utils root is three levels up from this file:
    #   src/KNF_Utils/api_keys.py -> src/KNF_Utils/ -> src/ -> NV_Comfy_Utils/
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(package_root, ".env")

    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                # Only set if not already in environment (real env vars take priority)
                if key and key not in os.environ:
                    os.environ[key] = value
        print(f"[NV_Comfy_Utils] Loaded .env from {env_path}")
    except Exception as e:
        print(f"[NV_Comfy_Utils] Warning: failed to read .env: {e}")


# ---------------------------------------------------------------------------
# Provider -> env var mapping
# ---------------------------------------------------------------------------

_PROVIDER_ENV_VARS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_api_key(api_key: str, provider: str = "gemini") -> str:
    """Resolve API key from explicit input, environment, or .env file.

    Args:
        api_key: Explicit key from node input (takes priority). Empty string = skip.
        provider: "gemini" or "openrouter" — determines which env vars to check.

    Returns:
        Resolved API key string.

    Raises:
        RuntimeError: If no key is found from any source.
    """
    # 1. Explicit input wins
    if api_key and api_key.strip():
        return api_key.strip()

    # 2. Load .env (if present and not already loaded)
    _load_dotenv_once()

    # 3. Check environment variables
    env_vars = _PROVIDER_ENV_VARS.get(provider, ())
    for var in env_vars:
        val = os.environ.get(var)
        if val and val.strip():
            print(f"[NV_Comfy_Utils] API key loaded from {var}")
            return val.strip()

    # 4. No key found
    var_names = " / ".join(env_vars) if env_vars else provider.upper() + "_API_KEY"
    raise RuntimeError(
        f"No API key for {provider}. Either:\n"
        f"  - Set {var_names} environment variable, or\n"
        f"  - Add it to NV_Comfy_Utils/.env file, or\n"
        f"  - Paste it into the api_key node input."
    )
