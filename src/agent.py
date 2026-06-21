import logging
import json
import asyncio
import aiohttp
from datetime import datetime,UTC
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    ToolError,
    cli,
    inference,
    utils,
    room_io,
)
from livekit.plugins import (
    ai_coustics,
    silero,
    sarvam,
    cartesia,
    google
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import the agents we defined in other files
from agent_support import CustomerSupportAgent
from agent_add_to_cart import AddtoCartAgent
from agent_review import ReviewAgent
from agent_order_confirmation import OrderConfirmationAgent

# Import shared functions
from shared_utils import _summarize_session

logger = logging.getLogger("ecommerce-agent")
load_dotenv(".env.local")

server = AgentServer(shutdown_process_timeout=60.0)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.7,  
        min_speech_duration=0.25, 
        min_silence_duration=1.0,
    )

server.setup_fnc = prewarm

async def _on_session_end_func(ctx: JobContext) -> None:
    ended_at = datetime.now(UTC)
    session = ctx._primary_agent_session
    if not session:
        logger.error("no primary agent session found for end_of_call processing")
        return

    report = ctx.make_session_report()
    summarizer = google.LLM(model="gemini-2.5-flash",api_key=os.getenv("GEMINI_API_KEY"))
    summary = await _summarize_session(summarizer, report.chat_history)
    
    headers_dict = {}
    body = {
        "job_id": report.job_id,
        "room_id": report.room_id,
        "room": report.room,
        "started_at": datetime.fromtimestamp(report.started_at, UTC).isoformat().replace("+00:00", "Z")
            if report.started_at
            else None,
        "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
        "summary": summary,
    }
    
    dc_results = ctx.proc.userdata.get("dc_results")
    if dc_results is not None:
        body["results"] = dc_results

    try:
        http_session = utils.http_context.http_session()
        timeout = aiohttp.ClientTimeout(total=10)
        resp = await asyncio.shield(http_session.post(
            "https://n8n.larynxai.in/webhook/28456216-8a3f-4153-a43b-73320dd5a536", timeout=timeout, json=body, headers=headers_dict
        ))
        if resp.status >= 400:
            logger.error(f"Webhook failed: HTTP {resp.status} - {resp.reason}")
        await resp.release()
    except ToolError:
        raise
    except (TimeoutError, aiohttp.ClientError) as e:
        raise ToolError(f"error: {e!s}") from e


@server.rtc_session(agent_name="ecommerce-agent", on_session_end=_on_session_end_func)
async def entrypoint(ctx: JobContext):
    
    # 1. Parse metadata safely
    metadata_str = ctx.job.metadata or "{}"
    try:
        metadata_dict = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata_dict = {}

    # 2. Extract agent_type (default to customersupport if not found)
    agent_type = metadata_dict.get("agent_type", "customersupport")
    
    # 3. Select Agent and Model conditionally
    if agent_type == "Cart":
        target_llm_model = "google/gemini-2.5-flash-lite"
        active_agent = AddtoCartAgent(metadata=metadata_str)

    elif agent_type == "Review":
        target_llm_model = "google/gemini-2.5-flash-lite"
        active_agent = ReviewAgent(metadata=metadata_str)

    elif agent_type == "orderconfirmation":
        target_llm_model = "google/gemini-2.5-flash-lite"
        active_agent = OrderConfirmationAgent(metadata=metadata_str)

    else:  # "customersupport" or fallback
        target_llm_model = "google/gemini-2.5-flash-lite"
        active_agent = CustomerSupportAgent()

    # Initialize the data collection user data context state
    ctx.proc.userdata["dc_results"] = None

    # 4. Initialize the session using the selected agent setup
    session = AgentSession(
        stt=sarvam.STT(model="saaras:v3",sample_rate=16000),
        llm=google.LLM(
            model=target_llm_model,
        ),
        tts=cartesia.TTS(
            language="hi",
            model="sonic-3.5",
            voice="605f8e6f-da68-4cb2-9931-1fc798664cc7",
            ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    await session.start(
        agent=active_agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L,
                ),
            ),
        ),
    )

if __name__ == "__main__":
    cli.run_app(server)
