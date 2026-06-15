from livekit import api
import asyncio

from livekit.protocol.agent_dispatch import RoomAgentDispatch
from dotenv import load_dotenv

load_dotenv()


async def connectTwillio():
    lkapi = api.LiveKitAPI("wss://agentic-ai-lnzb60jk.livekit.cloud","APIjeHVDwznPYYu","MFfeq6FyAAWEeAOsFdFqSaycCZe2YKgNKAMw6p9Y4KPB")
    res = await lkapi.connector.connect_twilio_call(
    api.ConnectTwilioCallRequest(
        twilio_call_direction=api.ConnectTwilioCallRequest.TWILIO_CALL_DIRECTION_OUTBOUND,
        room_name="twilio-connector-test",
        agents=[RoomAgentDispatch(agent_name="voice-assistant")],
        )
    )

asyncio.run(connectTwillio())

