import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="LiveKit Agent Configurer", version="1.0.0")

# Enable CORS for development ease
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


class StandardVoiceConfig(BaseModel):
    model: str
    voice: str


class PremiumVoiceConfig(BaseModel):
    stt_provider: str
    stt_language: str
    llm_model: str
    tts_model: str
    tts_voice_id: str


class AgentConfig(BaseModel):
    voice_mode: str
    active_prompt_id: str = "custom"
    custom_prompt: str
    standard_voice: StandardVoiceConfig
    premium_voice: PremiumVoiceConfig


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=404, detail="Config file not found")
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read config: {e!s}"
        ) from e


def save_config(config_data: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write config: {e!s}"
        ) from e


@app.get("/api/config")
async def get_config():
    """Retrieve the current active configuration."""
    return load_config()


@app.post("/api/config")
async def update_config(config: AgentConfig):
    """Update the agent's configuration."""
    config_dict = config.model_dump()
    save_config(config_dict)
    return {"status": "success", "config": config_dict}


# Serve static frontend files from src/static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    # Bind to port 8000 by default, allowing external access if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
