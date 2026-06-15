import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Voice Agent Control Panel", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.getenv(
    "CONFIG_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    ),
)

DEFAULT_CONFIG: dict = {
    "session_mode": "premium",
    "custom_prompt": "",
    "premium": {
        "model": "gemini-3.1-flash-live-preview",
        "voice": "Zephyr",
    },
    "standard": {
        "stt_language": "en-IN",
        "llm_model": "gemini-2.0-flash",
        "tts_voice": "",
    },
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PremiumConfig(BaseModel):
    model: str = "gemini-3.1-flash-live-preview"
    voice: str = "Zephyr"


class StandardConfig(BaseModel):
    stt_language: str = "en-IN"
    llm_model: str = "gemini-2.0-flash"
    tts_voice: str = ""


class AgentConfig(BaseModel):
    session_mode: str = "premium"
    custom_prompt: str = ""
    premium: PremiumConfig = PremiumConfig()
    standard: StandardConfig = StandardConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        # Write defaults on first run so the agent always has a config
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read config: {e!s}"
        ) from e


def save_config(data: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write config: {e!s}"
        ) from e


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    """Return the current agent configuration."""
    return load_config()


@app.post("/api/config")
async def update_config(config: AgentConfig):
    """Save updated agent configuration."""
    data = config.model_dump()
    save_config(data)
    return {"status": "success", "config": data}


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
