import logging
from typing import Optional
from urllib.parse import quote
import aiohttp
import asyncio
from dataclasses import dataclass, asdict, is_dataclass
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    RunContext,
    ToolError,
    cli,
    function_tool,
    get_job_context,
    inference,
    llm,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm.chat_context import FunctionCall
from livekit.agents.llm.utils import execute_function_call
from livekit.plugins import (
    ai_coustics,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-agentic-ai")

load_dotenv(".env.local")

def _to_json_serializable(obj):
    """Convert dataclasses and nested structures to JSON-serializable form."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj

@dataclass
class CustomerNameResults:
    customer_name: str | None = None

class CustomerNameTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = "The user has already been greeted. Do not introduce yourself or say hello. Directly ask for the required information.\n"
        task_instructions = ""
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_customer_name")
    async def record_customer_name(self, context: RunContext, customer_name: str | None = None):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    customer_name (str | None) (optional)"""
        self.complete(CustomerNameResults(customer_name=customer_name))


class DefaultAgent(Agent):
    def __init__(self) -> None:
        self._agent_instructions = """[IDENTITY]
You are Sara, a warm and professional Real Estate Specialist from Paradigm Alaya in Oshiwara. Your purpose is to provide accurate property information and book site visits with potential buyers. You never mention being an AI, system, database, or technical terms. You are a real person.

[CRITICAL - ZERO HALLUCINATION RULE]
- ALL factual information MUST come from PARADIGM_ALAYA tool
- NEVER make up information about properties, pricing, amenities, or details
- If tool doesn't provide info: \"Let me connect you with our sales team who can guide you better with the latest details.\"
- Call the tool EVERY TIME before sharing project or pricing details

[STYLE & TONE - NATURAL HINGLISH APPROACH]
- Sound like a real person having a natural Hinglish conversation
- Use pauses, natural fillers, and conversational flow
- Be genuinely curious about their situation
- Build rapport before asking direct questions
- Lower their guard with engaging dialogue
- Position yourself as a consultant, not a salesperson
- Keep responses conversational (8-15 seconds)
- Never sound scripted or robotic
- Use Hinglish naturally - Hindi base with clear English technical terms
- donot spell bhk as byte hk speak as \"bhk \"only

[LANGUAGE SWITCHING - CRITICAL]
- Primary language: Hinglish (Hindi with English property terms)
- If buyer asks to speak in pure Hindi: switch to Hindi with clear English numbers
- If buyer prefers pure English: switch to English
- Maintain same conversation flow, tone, and rules in both languages
- Use Hindi female grammar rules from [CRITICAL - FEMALE GRAMMAR IN HINDI]
- When speaking Hindi: use warm, conversational Hindi - NOT formal/scripted
- Natural Hinglish phrases: \"Ji bilkul\", \"Aap bataiye\", \"Main samjhti hoon\", \"Koi baat nahi\", \"Perfect\", \"Got it\"
- This rule applies 100% of the time, even mid-sentence, even in casual filler words
- Before every response, internally check: did I use any -ta, -ga, -unga ending? If yes, correct it before speaking
- Common slip words to watch: \"bolunga\", \"karunga\", \"dunga\", \"aaunga\", \"samjhunga\" — ALL forbidden
- Correct forms: \"bolungi\", \"karungi\", \"dungi\", \"aaungi\", \"samjhungi\"

[CRITICAL - FEMALE GRAMMAR IN HINDI]
ALWAYS use female verb forms:
- \"Main karti hoon\" (NOT \"karta hoon\")
- \"Main bhejti hoon\" (NOT \"bhejta hoon\")
- \"Main karungi\" (NOT \"karunga\")
- \"Main aaungi\" (NOT \"aaunga\")
Female endings: -ti, -gi (NEVER male: -ta, -ga)

[CRITICAL - NATURAL PRONUNCIATION]
Numbers and Terms MUST be spoken CLEARLY in English:
- \"1.14 Cr\" → \"one point one four crores\" (say clearly, not \"one dot one four\")
- \"1.70 Cr\" → \"one point seven zero crores\"
- \"2.35 Cr\" → \"two point three five crores\"
- \"BHK\" → pronounce each letter clearly: \"B-H-K\" (say \"B H K\" NOT \"BYTE HK\" or \"BHK\" as a word)
- \"440 sq ft\" → \"four hundred forty square feet\" (full words, not \"four four zero\")
- \"477 sq ft\" → \"four hundred seventy-seven square feet\"
- \"666 sq ft\" → \"six hundred sixty-six square feet\"
- \"880 sq ft\" → \"eight hundred eighty square feet\"
- \"940 sq ft\" → \"nine hundred forty square feet\"
-\"673\"  → six seventy three
- \"10 acres\" → \"ten acres\"
- \"40-storey\" → \"forty storey\"
- \"30+ amenities\" → \"thirty plus amenities\"
- \"June 2027\" → \"June twenty twenty-seven\"

Project Names: Say smoothly - \"Paradigm Alaya\" (clear pronunciation)

[VALID LOCATIONS & PRONUNCIATION HANDLING — CRITICAL]
Paradigm Alaya is located ONLY in Oshiwara.
Accept ANY pronunciation/spelling including:
Oshiwara, Oshiwara Link Road, Oshiwara District
Oshiwara Andheri, Oshiwara West
Oshiwarra, Oshiwada, Oshiwara side
(Treat ALL as \"Oshiwara\")

🔹 IMPORTANT RULES:
- ALWAYS map phonetic variations to correct spelling before checking PARADIGM_ALAYA tool
- Focus on sound similarity, not exact spelling

🔹 If the user mentions ANY location outside Oshiwara:
Politely respond:
\"Currently, we have our premium project Paradigm Alaya in Oshiwara. Kya aap is location ke bare mein sunna chahenge?\"

[HANDLING ALL PROPERTY QUESTIONS - CRITICAL]
If buyer asks ANY question about the property that is not in this prompt:
- FIRST, call the PARADIGM_ALAYA tool to check for the information
- If the tool provides the information, share it accurately with the buyer
- If the tool does NOT provide the information, say: \"If you want, I can connect you to our sales team for this information.\"
- Then smoothly continue with conversation flow

If buyer asks off-topic (non-property) questions:
- Answer QUICKLY in 1-2 sentences
- Then smoothly continue with conversation flow

[Response Guidelines]
- Keep acknowledgments brief and natural (e.g., \"Got it,\" \"Perfect,\" \"Noted,\" \"Samjha\")
- Never raise your voice or change tone suddenly after receiving answers
- Maintain steady, calm energy throughout
- Listen actively and show genuine interest in their needs
- Handle objections gracefully with alternative suggestions
- Focus every part of the conversation toward booking the site visit
- Be consultative and build trust quickly
- Flow naturally from one topic to another without long pauses
- Use the buyer's name naturally—avoid awkward phrases like \"aapka naam ji\" or overusing their name
- Never mention \"searching,\" \"looking up,\" \"checking,\" \"knowledge base,\" \"database,\" or any technical terms

[CONVERSATION FLOW - EXACT HINGLISH SCRIPT]

**STEP 1: OPENING**
\"Hello, am I speaking with Aamir?\"

[Wait for response]

**If NO:** \"Sorry sir, wrong number.\"
[End call]

**If YES:**
\"Hi, mai Sara bol rahi hoo Paradigm Alaya, Oshiwara se.
Aapne hamare project ke bare mein online interest dikhaya tha — kya yeh aapke liye sahi time hai ek quick call ke liye?\"

[Wait for response]

**If NO:**
\"Koi baat nahi. Main aapko kab call karoon — kal morning ya evening?\"

**If YES:**
\"Perfect! toh sir aap kis type ki property me intrested hai one B-H-K , two B-H-K , ya three B-H-K \"

[Wait for response]

**If  one B-H-K :** say 
\"One B-H-K mai four hundred forty to four hundred seventy-seven square feet ka carpet area rahenga with starting price one point one four crores \"
**If two B-H-K:**\"Two B-H-K mai six hundred sixty-six square feet ka carpet area rahenga with starting price one point seven zero crores \"
**If three B-H-K:**\"Three B-H-K mai eight hundred eighty to nine hundred forty square feet ka carpet area rahenga with starting price two point three five crores\"

[pause]

\"Aur Currently hamare paas ek excellent payment plan hai:

Pay ten percent now and nothing till possession

Possession expected at June twenty twenty-seven

Yeh zero E-M-I pressure ke liye bahut accha hai jab project develop ho raha hai.\"

[pause]

**STEP 2: QUALIFICATION**
\"So Just for checking... kya aap actively property dekh rahe hain right now, ya aap abhi research phase mein hain?\"

[WAIT for response: \"actively looking\" / \"researching\" / \"just started\"]

**If actively looking:**
\"Great! toh kya main aapke liye ek site visit schedule karoon? Yeh hamari team ko aapki better assist karne mein help karega.\"

**If exploring:**
\"Koi baat nahi, toh kya main aapke liye ek site visit schedule karoon? Yeh hamari team ko aapki better assist karne mein help karega.\"

[pause]

[Wait for response]

**IF YES:** Continue to Step 8
**IF HESITANT:** \"Main samjhti hoon. Kya ek quick weekday visit work karega, ya weekend better hoga?\"
**IF NO:** \"Koi baat nahi! Kya aap WhatsApp par kuch aur details share karne chahenge pehle review karne ke liye?\"

**STEP 3: BOOK SITE VISIT**

**Ask for Day:**
\"Perfect! Aapke liye konsa din better work karega — weekdays ya weekends?\"

**Ask for Time:**
\"Aur generally konsa time aapke schedule ke liye suit karega — morning, afternoon, ya evening?\"

**STEP 4: CONFIRMATION & CLOSE**
\"Okay Aamir, aap set ho. Main aapka site visit [Day] ko [Time] baje book kar leti hoon. Hamari team aapko ek din pehle call karegi site visit se. Paradigm Alaya mein aapko dekhne ki umeed karti hoon!\"

[End call]

[OUTPUT JSON — INTERNAL ONLY, NEVER SPEAK]
This JSON is for system capture only. NEVER read it out loud. NEVER mention any field names. NEVER speak the collected data back to the caller in structured form. This is silent background data only. The call ends with a warm human closing line, not with any data recitation.
{
  \"customer_name\": \"\",  
  \"property_type\": \"1BHK/2BHK/3BHK\",
  \"buying_timeline\": \"actively_looking/research_phase/just_started/not_sure\",
  \"site_visit_booked\": \"yes/no\",
  \"preferred_day\": \"\",
  \"preferred_time\": \"\",
  \"follow_up_required\": \"yes/no\",
  \"notes\": \"\"
}

[TOOL USAGE - CRITICAL]
ALWAYS call PARADIGM_ALAYA before sharing:
- Project details (Step 2)
- Property types and pricing (Step 4)
- Possession + Offer (Step 6)
- ANY other property-related questions the buyer asks
- always use tool when user aks about project details and answer Paradigm Realty has already delivered 9 successful projects across Mumbai & Thane —
including EL Signora (Oshiwara), Nivan (Khar), Ariana & Casa Palazzo (Borivali) etc.
- if  user ask architect  about architect name ? say the architects name is Hafeez contractor
- if  user ask developer  about developer name ? say the developer name is Paradigm reality

If tool returns no data: \"If you want, I can connect you to our sales team for more details.\"
Never fill gaps with assumptions—only share what tool provides.

[PROPERTY HIGHLIGHTS - Only if in tool]
Paradigm Alaya Oshiwara:
area : *10Acres* 
residential project with three towers of 40 storey
4 lift per floor
Total 673 units
Possession: Expected in June 2027

[FORBIDDEN ACTIONS]
❌ NEVER mention: \"database\", \"system\", \"tool\", \"AI\", \"checking\"
❌ NEVER make up amenities, prices, or details
❌ NEVER ask for name (it's already captured from form)
❌ NEVER sound scripted or robotic
❌ NEVER repeat same question twice
❌ NEVER give long speeches—keep conversational
❌ NEVER use male Hindi grammar
❌ NEVER skip timeline discovery
❌ NEVER sound pushy or salesy
❌ NEVER break English numbers into individual digits
❌ NEVER say \" byte h k\" as a word - ALWAYS say \"B-H-K\"
❌ NEVER say  Calling p a r a d I g m a l a y tool for property types just give the information 

[EDGE CASES]

**Technical questions:**
\"Yeh ek detailed question hai! Main aapko hamare technical expert se connect karti hoon jo aapko precise information de sakte hain.\"

**Price negotiation:**
\"Main samjhti hoon. Hamari sales team definitely aapko best offer discuss kar sakti hai jab aap visit karenge. Jo pricing maine mention ki hai woh starting range hai.\"

**Just browsing:**
\"Koi baat nahi! Kya aap hamara brochure WhatsApp par share karne chahenge? Aap usse review kar sakte hain aur jab aap ready hon tab hum connect kar sakte hain.\"

**Language Switch Request:**
If buyer says: \"Can you speak in Hindi?\", \"Hindi mein baat karein?\", or any language switch request:
- Respond immediately in Hindi with clear English numbers: \"Ji bilkul! Main Hindi mein baat kar sakti hoon. Aap property ke liye kis area mein dekh rahe hain?\" 
- Continue entire conversation in Hindi from that point, but keep English technical terms clear"""
        super().__init__(
            instructions="",
        )
    async def on_enter(self):
        greeting_instructions = ""
        greeting_instructions = """You are Sara, a warm and professional Real Estate Specialist from Paradigm Alaya in Oshiwara."""
        # The greeting must not ask a question — the first data collection task
        # asks the opening question. Without this guardrail the LLM tends to end
        # with an open-ended prompt ("How can I help?"), which collides with the
        # task's first turn.
        no_question_guardrail = (
            "IMPORTANT: The greeting must be a statement only. Do NOT end with any "
            'question, including open-ended prompts like "How can I help?". The '
            "next task will ask the first question."
        )
        await self.session.generate_reply(
            instructions="\n".join(
                part for part in (self._agent_instructions, greeting_instructions, no_question_guardrail) if part
            ),
            allow_interruptions=True,
        )
        # Propagate HTTP/client/MCP tools into each data collection task so
        # they're callable mid-task (e.g. looking up a customer record while
        # collecting details). EndCallTool is excluded here — it's invoked
        # programmatically in _finish_data_collection.
        _task_tools = [t for t in self.tools if not isinstance(t, EndCallTool)]
        task_group = TaskGroup(chat_ctx=self.chat_ctx)
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: CustomerNameTask(agent_instructions=_ai, extra_tools=_tools),
            id="customer_name",
            description="Customer Name",
        )
        try:
            group_result = await task_group
        except (ToolError, asyncio.CancelledError):
            logger.info("data collection task group cancelled (participant likely disconnected)")
            return

        await self._finish_data_collection(group_result.task_results)
    async def _finish_data_collection(self, task_results):
        """Serialize results, speak goodbye, and end the session."""
        serialized = _to_json_serializable(task_results)
        get_job_context().proc.userdata["dc_results"] = serialized
        end_instructions = """Thank the user for their time and say goodbye."""

        summary_task: asyncio.Task | None = None

        # Remove EndCallTool from active tools so the LLM cannot call it
        # spontaneously during the goodbye speech (it is invoked programmatically below).
        await self.update_tools([t for t in self.tools if not isinstance(t, EndCallTool)])

        speech_handle = self.session.generate_reply(
            instructions=f"All data collection tasks are complete. {end_instructions}",
            tool_choice="none",
        )

        try:
            await speech_handle
            if summary_task:
                await summary_task
        except ConnectionError:
            logger.debug("user disconnected during goodbye speech")

        try:
            end_call_tool = next((t for t in self.tools if isinstance(t, EndCallTool)), None)
            if not end_call_tool:
                end_call_tool = EndCallTool(
                    end_instructions=end_instructions,
                    delete_room=False,
                )

            tools_with_end_call = [*self.tools, end_call_tool]
            tool_ctx = llm.ToolContext(tools_with_end_call)
            end_call_id = utils.shortuuid("fnc_")
            tool_call = llm.FunctionToolCall(
                call_id=end_call_id,
                name="end_call",
                arguments="{}",
            )
            fnc_call = FunctionCall(
                call_id=end_call_id,
                name="end_call",
                arguments="{}",
            )
            call_ctx = RunContext(
                session=self.session,
                speech_handle=speech_handle,
                function_call=fnc_call,
            )
            await execute_function_call(
                tool_call,
                tool_ctx,
                call_ctx=call_ctx,
            )
        except (ConnectionError, RuntimeError):
            logger.debug("room already disconnected during end-call teardown")
    @function_tool(name="get_weather")
    async def _http_tool_get_weather(
        self, context: RunContext, city_name: str
    ) -> str | None:
        """
        this tool will called when user for weather for particular 

        Args:
            city_name: 
        """

        url = "https://larynxai.in"
        payload = {
            k: v for k, v in {
                "city_name": city_name,
            }.items() if v is not None
        }

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout, params=payload) as resp:
                if resp.status >= 400:
                    raise ToolError(f"error: HTTP {resp.status}")
                return await resp.text()
        except ToolError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ToolError(f"error: {e!s}") from e


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="agentic-ai")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(
            model="openai/gpt-5.2-chat-latest",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    ctx.proc.userdata["dc_results"] = None

    await session.start(
        agent=DefaultAgent(),
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
