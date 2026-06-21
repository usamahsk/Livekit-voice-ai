import asyncio
import json
from livekit import api

async def make_outbound_call_with_metadata():
    livekit_api = api.LiveKitAPI("wss://livekit.larynxai.in","devkey","secret")
    
    room_name = "larynxai"
    target_number = "+919967835090"
    
    # 1. Dispatch the agent with custom metadata
    dispatch_req = api.CreateAgentDispatchRequest(
        agent_name="ecommerce-agent",  # Must match the agent name in your worker code
        room=room_name,
        metadata=json.dumps({
            "target_number": target_number,
            "customer_name": "Alice",
            "intent": "appointment_reminder","agent_type":"Cart"
        })
    )
    await livekit_api.agent_dispatch.create_dispatch(dispatch_req)

    # 2. Initiate the SIP call to the same room
    sip_req = api.CreateSIPParticipantRequest(
        sip_trunk_id="ST_mbc2uA3YBXQH",
        sip_call_to=target_number,
        room_name=room_name,
        participant_identity="sip-outbound",
        participant_name="Outbound Caller"
    )
    
    try:
        await livekit_api.sip.create_sip_participant(sip_req)
        print("Agent dispatched and SIP call initiated successfully.")
    except Exception as e:
        print(f"Failed to initiate call: {e}")
    finally:
        await livekit_api.aclose()

if __name__ == "__main__":
    asyncio.run(make_outbound_call_with_metadata())
