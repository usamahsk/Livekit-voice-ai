import asyncio

from livekit import api

async def main():
  livekit_api = api.LiveKitAPI("wss://voice-t4cil9uy.livekit.cloud","APICP5xbCz8z7wo","WEUVwMirOJd1gvU1AwXXP6u48BrVgwdrGge9T18LTuc")

  # Create a dispatch rule to place all callers in the same room
  rule = api.SIPDispatchRule(
    dispatch_rule_direct = api.SIPDispatchRuleDirect(
      room_name = 'open-room',
    )
  )

  request = api.CreateSIPDispatchRuleRequest(
    dispatch_rule = api.SIPDispatchRuleInfo(
      rule = rule,
      name = 'My dispatch rule',
    )
  )

  try:
    dispatchRule = await livekit_api.sip.create_sip_dispatch_rule(request)
    print(f"Successfully created {dispatchRule}")
  except api.twirp_client.TwirpError as e:
    print(f"{e.code} error: {e.message}")

  await livekit_api.aclose()

asyncio.run(main())