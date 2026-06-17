import logging
import json
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import (
    ai_coustics,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import sarvam
from livekit.plugins import cartesia


# Import the agents we defined in other files
from agent_support import CustomerSupportAgent
from agent_add_to_cart import AddtoCartAgent
from agent_review import ReviewAgent
from agent_order_confirmation import OrderConfirmationAgent

logger = logging.getLogger("ecommerce-agent")
load_dotenv(".env.local")

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="ecommerce-agent")
async def entrypoint(ctx: JobContext):
    
    # 1. Parse metadata safely
    metadata_str = ctx.job.metadata or "{}"
    try:
        metadata_dict = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata_dict = {}

    # 2. Extract agent_type (default to customersupport if not found)
    agent_type = metadata_dict.get("agent_type", "customersupport")
    agent_type="orderconfirmation"
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

    # 4. Initialize the session using the selected agent setup
    session = AgentSession(
        stt=sarvam.STT(model="saaras:v3",sample_rate=16000),
        llm=inference.LLM(
            model=target_llm_model,
        ),
        tts=cartesia.TTS(
            language="hi",
            model="sonic-3.5",
            voice="605f8e6f-da68-4cb2-9931-1fc798664cc7",
            ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
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