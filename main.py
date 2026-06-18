import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from livekit import api

app = FastAPI()

# Input format for your API
class CallPayload(BaseModel):
    target_number: str
    customer_name: str
    intent: str
    agent_type: str

@app.post("/make-call")
async def make_call(payload: CallPayload):
    # Hardcoded your exact details from the working file
    livekit_api = api.LiveKitAPI("ws://localhost:7880","devkey","secret")
    room_name = "larynxai"
    sip_trunk_id = "ST_mbc2uA3YBXQH"
    
    try:
        # 1. Dispatch the agent with the custom metadata passed to the API
        dispatch_req = api.CreateAgentDispatchRequest(
            agent_name="ecommerce-agent",  
            room=room_name,
            metadata=json.dumps({
                "target_number": payload.target_number,
                "customer_name": payload.customer_name,
                "intent": payload.intent,
                "agent_type": payload.agent_type
            })
        )
        await livekit_api.agent_dispatch.create_dispatch(dispatch_req)

        # 2. Initiate the SIP call to the same room
        sip_req = api.CreateSIPParticipantRequest(
            sip_trunk_id=sip_trunk_id,
            sip_call_to=payload.target_number,
            room_name=room_name,
            participant_identity="sip-outbound",
            participant_name="Outbound Caller"
        )
        await livekit_api.sip.create_sip_participant(sip_req)
        
        return {"status": "success", "message": "Call initiated successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanly close the connection
        await livekit_api.aclose()