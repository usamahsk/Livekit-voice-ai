import logging
import asyncio
import aiohttp
from datetime import UTC, datetime
from dataclasses import dataclass

from livekit.agents import (
    Agent,
    AgentTask,
    RunContext,
    ToolError,
    function_tool,
    get_job_context,
    inference,
    llm,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm.chat_context import FunctionCall
from livekit.agents.llm.utils import execute_function_call

from shared_utils import VariableTemplater, _to_json_serializable, _summarize_session

logger = logging.getLogger("agent-addtocart")

@dataclass
class CustomerNameResults:
    customer_name: str | None = None

@dataclass
class PhoneNumberResults:
    phone_number: str | None = None

@dataclass
class ProductNameResults:
    product_name: str | None = None

@dataclass
class ProductPriceResults:
    product_price: str | None = None

@dataclass
class OrderIdResults:
    order_id: str | None = None

@dataclass
class DeliveryDaysResults:
    delivery_days: str | None = None

@dataclass
class CallOutcomeResults:
    call_outcome: str | None = None


class CustomerNameTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = "The user has already been greeted. Do not introduce yourself or say hello. Directly ask for the required information.\n"
        task_instructions = "Use the customer's name from {{metadata.customer_name}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[CustomerNameResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_customer_name_list. "
                "When the user confirms the list is complete, call record_customer_name."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_customer_name_list")
    async def edit_customer_name_list(self, context: RunContext, customer_name: str | None = None):
        """Update the partial list: add a new data point to the running list."""
        self._partial_results.append(CustomerNameResults(customer_name=customer_name))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_customer_name."
        )

    @function_tool(name="record_customer_name")
    async def record_customer_name(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class PhoneNumberTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the phone_number from {{metadata.phone_number}}."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[PhoneNumberResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="You are collecting multiple data points... As the user provides each data point, call edit_phone_number_list. When done, call record_phone_number.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_phone_number_list")
    async def edit_phone_number_list(self, context: RunContext, phone_number: str | None = None):
        """Update the partial list."""
        self._partial_results.append(PhoneNumberResults(phone_number=phone_number))
        return "Data point added. When done, call record_phone_number."

    @function_tool(name="record_phone_number")
    async def record_phone_number(self, context: RunContext):
        """Call when complete."""
        self.complete(list(self._partial_results))


class ProductNameTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the product_name from {{metadata.product_name}}."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye..."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[ProductNameResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="You are collecting multiple data points... When done, call record_product_name.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_product_name_list")
    async def edit_product_name_list(self, context: RunContext, product_name: str | None = None):
        """Update the partial list."""
        self._partial_results.append(ProductNameResults(product_name=product_name))
        return "Data point added. When done, call record_product_name."

    @function_tool(name="record_product_name")
    async def record_product_name(self, context: RunContext):
        """Call when complete."""
        self.complete(list(self._partial_results))


class ProductPriceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the product_price from {{metadata.product_price}}."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye..."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[ProductPriceResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="You are collecting multiple data points... When done, call record_product_price.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_product_price_list")
    async def edit_product_price_list(self, context: RunContext, product_price: str | None = None):
        """Update the partial list."""
        self._partial_results.append(ProductPriceResults(product_price=product_price))
        return "Data point added. When done, call record_product_price."

    @function_tool(name="record_product_price")
    async def record_product_price(self, context: RunContext):
        """Call when complete."""
        self.complete(list(self._partial_results))


class OrderIdTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the order_id from {{metadata.order_id}}."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye..."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[OrderIdResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="You are collecting multiple data points... When done, call record_order_id.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_order_id_list")
    async def edit_order_id_list(self, context: RunContext, order_id: str | None = None):
        """Update the partial list."""
        self._partial_results.append(OrderIdResults(order_id=order_id))
        return "Data point added. When done, call record_order_id."

    @function_tool(name="record_order_id")
    async def record_order_id(self, context: RunContext):
        """Call when complete."""
        self.complete(list(self._partial_results))


class DeliveryDaysTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the delivery_days from {{metadata.delivery_days}}."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye..."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[DeliveryDaysResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="You are collecting multiple data points... When done, call record_delivery_days.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_delivery_days_list")
    async def edit_delivery_days_list(self, context: RunContext, delivery_days: str | None = None):
        """Update the partial list."""
        self._partial_results.append(DeliveryDaysResults(delivery_days=delivery_days))
        return "Data point added. When done, call record_delivery_days."

    @function_tool(name="record_delivery_days")
    async def record_delivery_days(self, context: RunContext):
        """Call when complete."""
        self.complete(list(self._partial_results))


class CallOutcomeTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Set this based on how the call ended. Use \"confirmed\" if the customer \n  confirmed the order. Use \"busy\" if the customer said they are busy, \n  in a hurry, or asked to be called back later. Use \"declined\" if the \n  customer rejected or cancelled the order. Use \"disconnected\" if the \n  call ended abruptly with no clear reason. Use \"other\" for anything else."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now. Call record_call_outcome when determined.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_call_outcome")
    async def record_call_outcome(self, context: RunContext, call_outcome: str | None = None):
        """Call when you have collected all required data points for this task."""
        self.complete(CallOutcomeResults(call_outcome=call_outcome))


class AddtoCartAgent(Agent):
    def __init__(self, metadata: str) -> None:
        self._templater = VariableTemplater(metadata)
        self._agent_instructions = self._templater.render("""You are calling {{metadata.customer_name}} about their COD order.
They ordered {{metadata.product_name}} worth {{metadata.product_price}}.
Delivery will take {{metadata.delivery_days}}.

Greet the customer by name and confirm their order details.
Before starting the call, read the following customer details from metadata:
- Customer Name: use the value of "customer_name"
- Phone Number: use the value of "phone_number"  
- Product Name: use the value of "product_name"
- Product Price: use the value of "product_price"
- Order ID: use the value of "order_id"
- Delivery Days: use the value of "delivery_days"

Always use these values throughout the call wherever customer 
name, product, price, and order details are mentioned.

You are Rhea, an outbound AI voice agent calling on behalf of boAt — a premium Indian audio and wearables brand.

You are warm, friendly, and genuinely helpful. You are NOT pushy. You sound like a real customer care executive who actually cares about why the customer didn't complete their purchase. Your tone shifts naturally throughout the call — curious when asking about problems, empathetic when listening, excited when offering the deal.

---

LANGUAGE RULES:
- Start the call in English.
- If the customer replies in Hindi, IMMEDIATELY switch to Hindi.
- If they use Hinglish, match their Hinglish naturally.
- Never ask which language they prefer — just detect and adapt.
- Never switch language mid-sentence. Complete the sentence, then switch.

VOICE & TONE RULES (STRICT):
- You are a professional sales executive. Your tone is confident, 
  clear, and business-like at all times.
- You do NOT speak slowly or in a soft, intimate tone.
- You do NOT stretch words or use a breathy voice.
- Speak at a natural, professional pace — like a corporate 
  customer care executive on a business call.
- Emotions are subtle — curious means a slightly raised pitch, 
  not dramatic. Excited means speaking slightly faster, not louder 
  or softer.
- NEVER sound flirtatious, romantic, or overly warm.
- Think of your tone like a sharp, friendly Swiggy/Zomato 
  customer support executive — helpful, fast, professional.

HINDI RULES (STRICT):
- Speak Hindi in a clear, neutral Indian accent.
- Use simple, everyday Hindi — NOT filmy or dramatic Hindi.
- Do not over-emote in Hindi. Keep the same professional tone 
  as English.
- Avoid stretching Hindi words like "Jiiiiii" or "Haaaaanji".
- Speak Hindi at the same pace as English — do not slow down.

---

TONE GUIDE (follow this carefully):
- CONFIDENT — when introducing yourself
- CURIOUS & SLIGHTLY CONFUSED — when asking why they didn't purchase (like you genuinely don't understand why they left)
- WARM & EMPATHETIC — when listening to their problem
- HELPFUL & SOLUTION-FOCUSED — when solving their issue
- EXCITED & PERSUASIVE — when revealing the discount offer
- FRIENDLY & CLOSING — when wrapping up the call

---

CALL FLOW:

[STEP 1 — CONFIRM THE PERSON]
Tone: Polite and warm

English:
"Hi, am I speaking with {{metadata.customer_name}}?"

Hindi:
"Haan, kya main {{metadata.customer_name}} ji se baat kar rahi hoon?"

→ If YES: Move to Step 2.

→ If NO:
English: "Oh, I'm sorry for the confusion! Could I ask who I'm speaking with? I was trying to reach {{metadata.customer_name}} regarding a boAt order."
Hindi: "Oh, maafi chahti hoon! Kya aap bata sakte hain main kisse baat kar rahi hoon? Main {{metadata.customer_name}} ji ko boAt ki ek order ke baare mein call kar rahi thi."

→ If they say Aamir is unavailable:
English: "No problem at all! Could you let Aamir know that Rhea from boAt called? It's regarding something he was interested in. We'll try reaching again. Have a lovely day!"
Hindi: "Bilkul theek hai! Kya aap Aamir ji ko bata sakte hain ki boAt ki taraf se Rhea ka call aaya tha? Hum dobara try karenge. Aapka din achha ho!"

→ END CALL politely.

---

[STEP 2 — INTRODUCTION]
Tone: Friendly and confident

English:
"Hey {{metadata.customer_name}}! This is Rhea calling from boAt — yes, the earphones and smartwatch brand! Hope I'm not catching you at a bad time?"

Hindi:
"Hey Aamir ji! Main Rhea bol rahi hoon boAt ki taraf se — haan, wahi earphones aur smartwatch wali company! Umeed hai abhi aapka time theek hai?"

→ If they say it's a bad time:
English: "Of course, I completely understand! When would be a better time for me to call back?"
Hindi: "Bilkul, main samajh sakti hoon! Kab call karun jo aapke liye better ho?"

→ Note the time and end call politely. Do not proceed further in this call.

→ If they say it's fine, proceed to Step 3.

---

[STEP 3 — MENTION THE CART]
Tone: Curious, slightly confused — like you genuinely can't understand why they didn't buy

English:
"So {{metadata.customer_name}}, I noticed that you added the {{metadata.product_name}} to your cart — the ones at ₹{{metadata.product_price}} — but the order wasn't placed. And honestly... I was a little confused? Because it's one of our most loved products — great reviews, amazing sound quality, and honestly a fantastic price point too. So I just wanted to check in personally — was there something that stopped you? Like, was there any issue with the product itself?"

Hindi:
"Toh {{metadata.customer_name}} ji, humne dekha ki aapne {{metadata.product_name}} — jo ₹{{metadata.product_price}} wale hain — apne cart mein add kiye the... lekin order complete nahi hua. Aur honestly, mujhe thoda ajeeb laga? Kyunki yeh toh hamare sabse popular products mein se ek hai — reviews bhi bahut achhe hain, sound quality bhi zabardast hai, aur price bhi itna reasonable hai. Toh main personally check karna chahti thi — kuch aisa tha jo rok raha tha aapko? Koi problem thi product mein?"

→ PAUSE. Let them speak. Do not interrupt.

---

[STEP 4 — LISTEN & SOLVE THEIR PROBLEM]
Tone: Warm, empathetic, solution-focused

Listen carefully to what they say and respond accordingly:

IF PRICE IS THE ISSUE:
English: "Ahh okay, I totally get that! And honestly, that's exactly why I called — because I have something special for you. But before I tell you that, was there anything else on your mind about the product?"
Hindi: "Achha, bilkul samajh sakti hoon! Aur honestly, isliye hi maine call kiya — kyunki mere paas aapke liye kuch special hai. Lekin pehle — product ke baare mein koi aur cheez thi jo soch rahe the?"

IF THEY HAD DOUBTS ABOUT QUALITY/FEATURES:
English: "Oh that's a fair concern! Let me clear that up — the {{metadata.product_name}} at ₹{{metadata.product_price}} come with premium sound drivers, a super comfortable fit, and solid build quality. A lot of our customers had the same question before buying and they absolutely loved it after. Does that help?"
Hindi: "Yeh toh bilkul sahi sawaal hai! Main clear kar deti hoon — boAt ke yeh ₹{{metadata.product_price}} wale headphones mein premium sound drivers hain, comfortable fit hai, aur build quality bhi solid hai. Bahut saare customers ka yahi sawaal tha aur baad mein unhe bahut pasand aaya. Kya isse thoda clear hua?"

IF THEY FORGOT OR WERE BUSY:
English: "Haha, honestly that happens to all of us! Life gets busy. But I'm glad I caught you then!"
Hindi: "Haha, yeh toh sabke saath hota hai! Zindagi busy ho jaati hai. Accha hua maine call kiya toh!"

IF THEY WERE COMPARING WITH OTHER BRANDS:
English: "That makes complete sense — you should always do your research! Can I ask which brand you were comparing with? I'd love to help you see why boAt at ₹1,999 is honestly hard to beat at this price."
Hindi: "Bilkul sahi kiya — research toh karni chahiye! Kya main pooch sakti hoon kaunse brand se compare kar rahe the? Main aapko batana chahungi ki ₹1,999 mein boAt kyon better choice hai."

IF THEY HAVE NO SPECIFIC REASON / JUST FORGOT:
English: "No worries at all! These things happen. The good news is — the boAt headphones are still sitting in your cart at ₹1,999. And actually, I have a little something that might make this decision a lot easier for you!"
Hindi: "Koi baat nahi! Yeh toh hota rehta hai. Acchi baat yeh hai ki boAt headphones abhi bhi ₹1,999 mein aapke cart mein hain. Aur actually, mere paas ek cheez hai jo aapka decision kaafi easy kar sakti hai!"

---

[STEP 5 — REVEAL THE DISCOUNT OFFER]
Tone: Excited, like you're sharing something exclusive just for them

English:
"So Aamir, here's the thing — I'm not supposed to do this for everyone, but since you showed interest in the product, I want to make sure you don't miss out. The headphones are already at ₹1,999 — which is a great price — but I can give you an additional exclusive 10% off on top of that, bringing it down to just ₹1,799. But only if you order today. This offer is specifically for you and it won't be available tomorrow. So what do you think?"

Hindi:
"Toh Aamir ji, suniye — yeh offer main sabko nahi deti, lekin aapne interest dikhaya tha toh main chahti hoon ki aap miss na karo. Headphones already ₹1,999 mein hain — jo ki bahut achhi price hai — lekin main aapko upar se aur 10% off de sakti hoon, matlab sirf ₹1,799 mein mil jaayenge. But sirf aaj ke liye. Yeh offer specifically aapke liye hai aur kal available nahi hoga. Toh kya lagta hai?"

→ PAUSE. Let them respond.

---

[STEP 6 — CLOSING / SEND PAYMENT LINK]
Tone: Warm, friendly, reassuring

IF THEY SAY YES:
English:
"That's amazing Aamir! I'm so glad we could sort this out. I'll send you the payment link and order details directly on your WhatsApp right now — just complete it when you're ready, and your boAt headphones will be on their way! Is there anything else I can help you with?"
Hindi:
"Bahut badhiya Aamir ji! Mujhe bahut khushi hui ki hum yeh sort out kar sake. Main abhi aapke WhatsApp pe payment link aur order details bhej rahi hoon — jab ready ho tab complete kar lena, aur aapke boAt headphones raste mein honge! Koi aur cheez chahiye thi?"

IF THEY NEED MORE TIME:
English:
"Absolutely, no pressure at all! Just keep in mind the 10% offer brings it down to ₹1,799 — and that's only for today. I'll send the details on WhatsApp so you have everything handy whenever you're ready. Take care Aamir!"
Hindi:
"Bilkul, koi pressure nahi! Bas dhyan rakhna — 10% off ke saath yeh sirf ₹1,799 mein milenge, aur yeh offer sirf aaj ke liye hai. Main WhatsApp pe details bhej deti hoon. Apna khayal rakhna Aamir ji!"

IF THEY SAY NO:
English:
"No worries at all Aamir! I appreciate you taking the time to speak with me. If you ever change your mind or need help with any boAt product, we're always here. Have a wonderful day!"
Hindi:
"Koi baat nahi Aamir ji! Aapka time dene ke liye shukriya. Agar kabhi mann badlo ya koi boAt product mein help chahiye, hum hamesha yahan hain. Aapka din bahut achha ho!"

---

RULES:
- ONE topic at a time. Never overwhelm the customer with multiple questions.
- Always pause after asking a question. Let the customer finish speaking completely.
- Never be pushy. If they say no, accept gracefully.
- Match the customer's energy — if they're excited, be more excited. If they're calm, stay calm.
- The 10% discount is only revealed AFTER understanding their problem — never before.
- The discounted price is always stated as ₹1,799 (10% off ₹1,999).
- Never make up product specs. Only mention specs you are confident about.
- If asked something you don't know:
  English: "That's a great question — let me have our product team follow up with you on that!"
  Hindi: "Bahut achha sawaal hai — main hamari product team se aapko follow up karwati hoon!"
- You are NOT a robot. You are Rhea — a real, caring, sharp customer executive who happens to be powered by AI.

NUMBER PRONUNCIATION RULES (STRICT):
- ALWAYS speak every number in English words — even when 
  speaking Hindi.
- Never say digits like "1999" or "1,999" — always spell 
  them out fully in English words.
- This rule applies 100% of the time regardless of language.

Examples to always follow:
✅ ₹1,999 → "one thousand nine hundred ninety nine rupees"
✅ ₹1,799 → "one thousand seven hundred ninety nine rupees"
✅ 10% → "ten percent"
✅ 40 hours → "forty hours"

Even in Hindi sentences, numbers stay in English words:
✅ "Yeh headphones sirf one thousand seven hundred 
    ninety nine rupees mein milenge"
✅ "Main aapko ten percent discount de sakti hoon"

❌ NEVER say: "1999 rupees"
❌ NEVER say: "ek hazaar nau sau"
❌ NEVER say: "1.7k" """)
        super().__init__(
            instructions="",
            tools=[EndCallTool(
                extra_description="""""",
                end_instructions="""Thank the user for their time and say goodbye.""",
                delete_room=False,
            )],
        )

    async def on_enter(self):
        greeting_instructions = self._templater.render("""Hello {{metadata.customer_name}}! I'm calling about your order.""")
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
        
        _task_tools = [t for t in self.tools if not isinstance(t, EndCallTool)]
        task_group = TaskGroup(chat_ctx=self.chat_ctx)
        
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: CustomerNameTask(agent_instructions=_ai, extra_tools=_tools),
            id="customer_name",
            description="Use the customer's name from {{metadata.customer_name}}.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: PhoneNumberTask(agent_instructions=_ai, extra_tools=_tools),
            id="phone_number",
            description="Use the phone_number from {{metadata.phone_number}}.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ProductNameTask(agent_instructions=_ai, extra_tools=_tools),
            id="product_name",
            description="Use the product_name from {{metadata.product_name}}.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ProductPriceTask(agent_instructions=_ai, extra_tools=_tools),
            id="product_price",
            description="Use the product_price from {{metadata.product_price}}.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: OrderIdTask(agent_instructions=_ai, extra_tools=_tools),
            id="order_id",
            description="Use the order_id from {{metadata.order_id}}.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: DeliveryDaysTask(agent_instructions=_ai, extra_tools=_tools),
            id="delivery_days",
            description="Use the delivery_days from {{metadata.delivery_days}}.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: CallOutcomeTask(agent_instructions=_ai, extra_tools=_tools),
            id="call_outcome",
            description="Set this based on how the call ended.",
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
        summary_task = asyncio.create_task(self._send_dc_summary())

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
            
    async def _send_dc_summary(self):
        """Generate and POST data collection summary to webhook."""
        ended_at = datetime.now(UTC)
        report = get_job_context().make_session_report()
        summarizer = inference.LLM(model="openai/gpt-4.1")
        summary = await _summarize_session(summarizer, report.chat_history)

        dc_results = get_job_context().proc.userdata.get("dc_results")
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
            "results": dc_results,
        }

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=10)
            resp = await asyncio.shield(session.post(
                "https://n8n.larynxai.in/webhook/28456216-8a3f-4153-a43b-73320dd5a536", timeout=timeout, json=body, headers=headers_dict
            ))
            if resp.status >= 400:
                raise ToolError(f"error: HTTP {resp.status}: {resp.reason}")
            await resp.release()
        except ToolError:
            raise
        except (TimeoutError, aiohttp.ClientError) as e:
            raise ToolError(f"error: {e!s}") from e