import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent import PROMPTS, DefaultAgent

# Import the web server and agent modules
from src.web_server import app

# Setup FastAPI TestClient
client = TestClient(app)


def test_default_agent_instructions():
    """Test that DefaultAgent initializes with correct instructions."""
    test_instructions = "You are a test robot agent."
    agent = DefaultAgent(instructions=test_instructions)
    assert agent.instructions == test_instructions


def test_prompts_dict_structure():
    """Test that static prompts are correctly defined and populated."""
    assert "outbound_ghee" in PROMPTS
    assert "customer_service" in PROMPTS
    assert "front_desk" in PROMPTS
    assert "Buffalo Ghee" in PROMPTS["outbound_ghee"]
    assert "receptionist" in PROMPTS["front_desk"]
    assert "customer support" in PROMPTS["customer_service"]


def test_api_get_config(tmp_path):
    """Test API endpoint to retrieve configuration."""
    temp_config = tmp_path / "config.json"
    dummy_config = {
        "voice_mode": "standard",
        "active_prompt_id": "outbound_ghee",
        "custom_prompt": "Hello custom",
        "standard_voice": {"model": "gemini-3.1-flash-live-preview", "voice": "Zephyr"},
        "premium_voice": {
            "stt_provider": "google",
            "stt_language": "en-IN",
            "llm_model": "gemini-3.1-flash-lite",
            "tts_model": "speech-02-turbo",
            "tts_voice_id": "socialmedia_female_2_v1",
        },
    }

    # Save the dummy config to temp path
    with open(temp_config, "w", encoding="utf-8") as f:
        json.dump(dummy_config, f)

    with patch("src.web_server.CONFIG_PATH", str(temp_config)):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["voice_mode"] == "standard"
        assert data["active_prompt_id"] == "outbound_ghee"


def test_api_post_config(tmp_path):
    """Test API endpoint to update configuration."""
    temp_config = tmp_path / "config.json"

    new_config = {
        "voice_mode": "premium",
        "active_prompt_id": "custom",
        "custom_prompt": "You are a specialized custom helper.",
        "standard_voice": {"model": "gemini-3.1-flash-live-preview", "voice": "Zephyr"},
        "premium_voice": {
            "stt_provider": "deepgram",
            "stt_language": "en-US",
            "llm_model": "gemini-3.1-flash-lite",
            "tts_model": "speech-02-turbo",
            "tts_voice_id": "socialmedia_female_2_v1",
        },
    }

    # Mocking the CONFIG_PATH to save to our temp file
    with patch("src.web_server.CONFIG_PATH", str(temp_config)):
        # Initially, create an empty file so load_config doesn't fail
        with open(temp_config, "w", encoding="utf-8") as f:
            json.dump(new_config, f)

        response = client.post("/api/config", json=new_config)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Verify it wrote to file correctly
        with open(temp_config, encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["voice_mode"] == "premium"
            assert saved_data["active_prompt_id"] == "custom"
            assert saved_data["custom_prompt"] == "You are a specialized custom helper."
            assert saved_data["premium_voice"]["stt_provider"] == "deepgram"


def test_api_post_config_default_prompt_id(tmp_path):
    """Test API endpoint to update configuration when active_prompt_id is omitted."""
    temp_config = tmp_path / "config.json"

    new_config = {
        "voice_mode": "premium",
        "custom_prompt": "You are a specialized custom helper.",
        "standard_voice": {"model": "gemini-3.1-flash-live-preview", "voice": "Zephyr"},
        "premium_voice": {
            "stt_provider": "deepgram",
            "stt_language": "en-US",
            "llm_model": "gemini-3.1-flash-lite",
            "tts_model": "speech-02-turbo",
            "tts_voice_id": "socialmedia_female_2_v1",
        },
    }

    with patch("src.web_server.CONFIG_PATH", str(temp_config)):
        with open(temp_config, "w", encoding="utf-8") as f:
            json.dump(new_config, f)

        response = client.post("/api/config", json=new_config)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Verify it wrote to file correctly and set active_prompt_id to default "custom"
        with open(temp_config, encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["voice_mode"] == "premium"
            assert saved_data["active_prompt_id"] == "custom"
            assert saved_data["custom_prompt"] == "You are a specialized custom helper."
