import asyncio

from livekit import api
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPOutboundConfig


async def main():
    livekit_api = api.LiveKitAPI(
        "wss://voice-t4cil9uy.livekit.cloud",
        "APICP5xbCz8z7wo",
        "WEUVwMirOJd1gvU1AwXXP6u48BrVgwdrGge9T18LTuc",
    )

    trunk_config = SIPOutboundConfig(
        hostname="larynxai-demo-trunk.pstn.twilio.com",  # For example, <my-trunk>.pstn.twilio.com or sip.telnyx.com
        auth_username="castlecraft",
        auth_password="Castlecraft@2019",
    )

    request = CreateSIPParticipantRequest(
        trunk=trunk_config,
        sip_number="+19452225059",  # Required when using inline trunk config
        sip_call_to="+919870209779",
        # sip_call_to="+919967835090",
        room_name="open-room",
        participant_identity="sip-test",
        participant_name="Test Caller",
        krisp_enabled=True,
        wait_until_answered=True,
    )

    try:
        participant = await livekit_api.sip.create_sip_participant(request)
        print(f"Successfully created {participant}")
    except Exception as e:
        print(f"Error creating SIP participant: {e}")
    finally:
        await livekit_api.aclose()


asyncio.run(main())
