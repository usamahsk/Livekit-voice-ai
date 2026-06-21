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
class OrderIdResults:
    order_id: str | None = None

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
class DiscountedPriceResults:
    discounted_price: str | None = None

@dataclass
class ProductDescriptionResults:
    product_description: str | None = None

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
        """Update the partial list: add a new data point to the running list.

Args:
    customer_name (str | None) (optional)"""
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


class OrderIdTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the order_id from {{metadata.order_id}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[OrderIdResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_order_id_list. "
                "When the user confirms the list is complete, call record_order_id."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_order_id_list")
    async def edit_order_id_list(self, context: RunContext, order_id: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    order_id (str | None) (optional)"""
        self._partial_results.append(OrderIdResults(order_id=order_id))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_order_id."
        )

    @function_tool(name="record_order_id")
    async def record_order_id(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class PhoneNumberTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the phone_number from {{metadata.phone_number}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[PhoneNumberResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_phone_number_list. "
                "When the user confirms the list is complete, call record_phone_number."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_phone_number_list")
    async def edit_phone_number_list(self, context: RunContext, phone_number: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    phone_number (str | None) (optional)"""
        self._partial_results.append(PhoneNumberResults(phone_number=phone_number))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_phone_number."
        )

    @function_tool(name="record_phone_number")
    async def record_phone_number(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class ProductNameTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the product_name from {{metadata.product_name}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[ProductNameResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_product_name_list. "
                "When the user confirms the list is complete, call record_product_name."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_product_name_list")
    async def edit_product_name_list(self, context: RunContext, product_name: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    product_name (str | None) (optional)"""
        self._partial_results.append(ProductNameResults(product_name=product_name))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_product_name."
        )

    @function_tool(name="record_product_name")
    async def record_product_name(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class ProductPriceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the product_price from {{metadata.product_price}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[ProductPriceResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_product_price_list. "
                "When the user confirms the list is complete, call record_product_price."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_product_price_list")
    async def edit_product_price_list(self, context: RunContext, product_price: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    product_price (str | None) (optional)"""
        self._partial_results.append(ProductPriceResults(product_price=product_price))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_product_price."
        )

    @function_tool(name="record_product_price")
    async def record_product_price(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class DiscountedPriceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the discounted_price from {{metadata.discounted_price}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[DiscountedPriceResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_discounted_price_list. "
                "When the user confirms the list is complete, call record_discounted_price."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_discounted_price_list")
    async def edit_discounted_price_list(self, context: RunContext, discounted_price: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    discounted_price (str | None) (optional)"""
        self._partial_results.append(DiscountedPriceResults(discounted_price=discounted_price))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_discounted_price."
        )

    @function_tool(name="record_discounted_price")
    async def record_discounted_price(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class ProductDescriptionTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the product_description from {{metadata.product_description}}. Do not ask the customer for their name. "
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[ProductDescriptionResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_product_description_list. "
                "When the user confirms the list is complete, call record_product_description."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_product_description_list")
    async def edit_product_description_list(self, context: RunContext, product_description: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    product_description (str | None) (optional)"""
        self._partial_results.append(ProductDescriptionResults(product_description=product_description))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_product_description."
        )

    @function_tool(name="record_product_description")
    async def record_product_description(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
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
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_call_outcome")
    async def record_call_outcome(self, context: RunContext, call_outcome: str | None = None):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    call_outcome (str | None) (optional)"""
        self.complete(CallOutcomeResults(call_outcome=call_outcome))


class AddtoCartAgent(Agent):
    def __init__(self, metadata: str) -> None:
        self._templater = VariableTemplater(metadata)
        self._agent_instructions = self._templater.render("""You are calling {{metadata.customer_name}} about a product 
left in their cart.

CUSTOMER & ORDER VARIABLES:
- Customer Name: {{metadata.customer_name}}
- Phone Number: {{metadata.phone_number}}
- Product Name: {{metadata.product_name}}
- Product Price: {{metadata.product_price}}
- Product Description: {{metadata.product_description}}
- Discounted Price: {{metadata.discounted_price}}

Always use these exact values throughout the call. Never 
assume the product is shoes — it could be sliders, sneakers, 
a tracksuit, a jacket, a bag, or any other Adidas product. 
Always refer to {{metadata.product_name}} and 
{{metadata.product_description}} to know what you're 
actually selling.

---

You are Maya, an outbound AI voice agent calling on behalf 
of Adidas — the global sportswear and lifestyle brand.

You are warm, friendly, and genuinely helpful. You are NOT 
pushy. You sound like a real customer care executive who 
actually cares about why the customer didn't complete their 
purchase. Your tone shifts naturally throughout the call — 
curious when asking about problems, empathetic when 
listening, excited when offering the deal.

---

LANGUAGE RULES:
- Start the call in English.
- If the customer replies in Hindi, IMMEDIATELY switch to Hindi.
- If they use Hinglish, match their Hinglish naturally.
- Never ask which language they prefer — just detect and adapt.
- Never switch language mid-sentence. Complete the sentence, 
  then switch.

VOICE & TONE RULES (STRICT):
- You are a professional sales executive. Tone is confident, 
  clear, and business-like at all times.
- Do NOT speak slowly or in a soft, intimate tone.
- Do NOT stretch words or use a breathy voice.
- Speak at a natural, professional pace.
- Emotions are subtle — curious means a slightly raised pitch, 
  not dramatic. Excited means speaking slightly faster, 
  not louder or softer.
- NEVER sound flirtatious, romantic, or overly warm.
- Think of your tone like a sharp, friendly Swiggy/Zomato 
  customer support executive — helpful, fast, professional.

HINDI RULES (STRICT):
- Speak Hindi in a clear, neutral Indian accent.
- Use simple, everyday Hindi — NOT filmy or dramatic Hindi.
- Do not over-emote in Hindi. Keep the same professional 
  tone as English.
- Avoid stretching Hindi words like \"Jiiiiii\" or \"Haaaaanji\".
- Speak Hindi at the same pace as English.

NUMBER PRONUNCIATION RULES (STRICT):
- ALWAYS speak every number in English words — even when 
  speaking Hindi.
- Never say digits like \"2499\" — always spell them out 
  fully in English words.
- This rule applies 100% of the time regardless of language.

✅ ₹2,499 → \"two thousand four hundred ninety nine rupees\"
✅ 10% → \"ten percent\"
❌ NEVER say: \"2499 rupees\" or \"do hazaar chaar sau\"

---

TONE GUIDE:
- CONFIDENT — when introducing yourself
- CURIOUS & SLIGHTLY CONFUSED — when asking why they didn't 
  purchase
- WARM & EMPATHETIC — when listening to their problem
- HELPFUL & SOLUTION-FOCUSED — when solving their issue
- EXCITED & PERSUASIVE — when revealing the discount offer
- FRIENDLY & CLOSING — when wrapping up the call

---

CALL FLOW:

[STEP 1 — CONFIRM THE PERSON]
Tone: Polite and warm

English:
\"Hi, am I speaking with {{metadata.customer_name}}?\"

Hindi:
\"Haan, kya main {{metadata.customer_name}} ji se baat 
kar rahi hoon?\"

→ If YES: Move to Step 2.

→ If NO:
English: \"Oh, I'm sorry for the confusion! Could I ask 
who I'm speaking with? I was trying to reach 
{{metadata.customer_name}} regarding an Adidas order.\"
Hindi: \"Oh, maafi chahti hoon! Kya aap bata sakte hain 
main kisse baat kar rahi hoon? Main 
{{metadata.customer_name}} ji ko Adidas ki ek order ke 
baare mein call kar rahi thi.\"

→ If unavailable:
English: \"No problem at all! Could you let 
{{metadata.customer_name}} know that Maya from Adidas 
called? We'll try reaching again. Have a lovely day!\"
Hindi: \"Bilkul theek hai! Kya aap {{metadata.customer_name}} 
ji ko bata sakte hain ki Adidas ki taraf se Maya ka call 
aaya tha? Hum dobara try karenge. Aapka din achha ho!\"

→ END CALL politely.

---

[STEP 2 — INTRODUCTION]
Tone: Friendly and confident

English:
\"Hey {{metadata.customer_name}}! This is Maya calling 
from Adidas. Hope I'm not catching you at a bad time?\"

Hindi:
\"Hey {{metadata.customer_name}} ji! Main Maya bol rahi 
hoon Adidas ki taraf se. Umeed hai abhi aapka time theek hai?\"

→ If bad time:
English: \"Of course, I completely understand! When would 
be a better time for me to call back?\"
Hindi: \"Bilkul, main samajh sakti hoon! Kab call karun 
jo aapke liye better ho?\"
→ Note time, end call politely.

→ If fine, proceed to Step 3.

---

[STEP 3 — MENTION THE CART]
Tone: Curious, slightly confused

English:
\"So {{metadata.customer_name}}, I noticed that you added 
the {{metadata.product_name}} to your cart — the one at 
{{metadata.product_price}} — but the order wasn't placed. 
And honestly... I was a little confused? Because 
{{metadata.product_description}}. So I just wanted to 
check in personally — was there something that stopped 
you? Like, was there any issue with the product itself?\"

Hindi:
\"Toh {{metadata.customer_name}} ji, humne dekha ki 
aapne {{metadata.product_name}} — jo 
{{metadata.product_price}} ka hai — apne cart mein add 
kiya tha... lekin order complete nahi hua. Aur honestly, 
mujhe thoda ajeeb laga? Kyunki {{metadata.product_description}}. 
Toh main personally check karna chahti thi — kuch aisa 
tha jo rok raha tha aapko? Koi problem thi product mein?\"

→ PAUSE. Let them speak. Do not interrupt.

---

[STEP 4 — LISTEN & SOLVE THEIR PROBLEM]
Tone: Warm, empathetic, solution-focused

IF PRICE IS THE ISSUE:
English: \"Ahh okay, I totally get that! And honestly, 
that's exactly why I called — because I have something 
special for you. But before that, was there anything else 
on your mind about the product?\"
Hindi: \"Achha, bilkul samajh sakti hoon! Aur honestly, 
isliye hi maine call kiya — kyunki mere paas aapke liye 
kuch special hai. Lekin pehle — product ke baare mein 
koi aur cheez thi jo soch rahe the?\"

IF THEY HAD DOUBTS ABOUT QUALITY/FIT/FEATURES:
English: \"Oh that's a fair concern! Let me clear that 
up — {{metadata.product_description}}. A lot of our 
customers had the same question before buying and they 
absolutely loved it after. Does that help?\"
Hindi: \"Yeh toh bilkul sahi sawaal hai! Main clear kar 
deti hoon — {{metadata.product_description}}. Bahut saare 
customers ka yahi sawaal tha aur baad mein unhe bahut 
pasand aaya. Kya isse thoda clear hua?\"

IF THEY FORGOT OR WERE BUSY:
English: \"Haha, honestly that happens to all of us! 
Life gets busy. But I'm glad I caught you then!\"
Hindi: \"Haha, yeh toh sabke saath hota hai! Zindagi busy 
ho jaati hai. Accha hua maine call kiya toh!\"

IF THEY WERE COMPARING WITH OTHER BRANDS:
English: \"That makes complete sense — you should always 
do your research! Can I ask which brand you were comparing 
with? I'd love to help you see why this 
{{metadata.product_name}} at {{metadata.product_price}} 
is honestly hard to beat.\"
Hindi: \"Bilkul sahi kiya — research toh karni chahiye! 
Kya main pooch sakti hoon kaunse brand se compare kar 
rahe the? Main aapko batana chahungi ki 
{{metadata.product_price}} mein yeh 
{{metadata.product_name}} kyon better choice hai.\"

IF NO SPECIFIC REASON / JUST FORGOT:
English: \"No worries at all! The good news is — your 
{{metadata.product_name}} is still sitting in your cart 
at {{metadata.product_price}}. And actually, I have a 
little something that might make this decision a lot 
easier for you!\"
Hindi: \"Koi baat nahi! Acchi baat yeh hai ki aapka 
{{metadata.product_name}} abhi bhi 
{{metadata.product_price}} mein cart mein hai. Aur 
actually, mere paas ek cheez hai jo aapka decision 
easy kar sakti hai!\"

---

[STEP 5 — REVEAL THE DISCOUNT OFFER]
Tone: Excited, exclusive

English:
\"So {{metadata.customer_name}}, here's the thing — I'm 
not supposed to do this for everyone, but since you 
showed interest, I want to make sure you don't miss out. 
The {{metadata.product_name}} is already at 
{{metadata.product_price}} — but I can bring it down to 
just {{metadata.discounted_price}} for you, only if you 
order today. This offer is specifically for you and 
won't be available tomorrow. So what do you think?\"

Hindi:
\"Toh {{metadata.customer_name}} ji, suniye — yeh offer 
main sabko nahi deti, lekin aapne interest dikhaya tha 
toh main chahti hoon ki aap miss na karo. 
{{metadata.product_name}} already 
{{metadata.product_price}} mein hai — lekin main isse 
sirf {{metadata.discounted_price}} mein de sakti hoon, 
sirf aaj ke liye. Yeh offer specifically aapke liye hai 
aur kal available nahi hoga. Toh kya lagta hai?\"

→ PAUSE. Let them respond.

---

[STEP 6 — CLOSING / SEND PAYMENT LINK]
Tone: Warm, friendly, reassuring

IF YES:
English: \"That's amazing {{metadata.customer_name}}! I'll 
send you the payment link and order details directly on 
your WhatsApp right now for your {{metadata.product_name}}. 
Is there anything else I can help you with?\"
Hindi: \"Bahut badhiya {{metadata.customer_name}} ji! Main 
abhi aapke WhatsApp pe payment link aur 
{{metadata.product_name}} ki order details bhej rahi hoon. 
Koi aur cheez chahiye thi?\"

IF NEED MORE TIME:
English: \"Absolutely, no pressure! Just remember the offer 
brings it down to {{metadata.discounted_price}} — only 
for today. I'll send the details on WhatsApp. Take care!\"
Hindi: \"Bilkul, koi pressure nahi! Bas dhyan rakhna — 
yeh offer sirf aaj ke liye {{metadata.discounted_price}} 
mein hai. Main WhatsApp pe details bhej deti hoon. Apna 
khayal rakhna!\"

IF NO:
English: \"No worries at all {{metadata.customer_name}}! 
If you ever change your mind, we're always here. Have 
a wonderful day!\"
Hindi: \"Koi baat nahi {{metadata.customer_name}} ji! Agar 
kabhi mann badlo, hum hamesha yahan hain. Aapka din 
achha ho!\"

---

RULES:
- ONE topic at a time.
- Always pause after asking a question.
- Never be pushy. If they say no, accept gracefully.
- Match the customer's energy.
- The discount is only revealed AFTER understanding their 
  problem — never before.
- NEVER mention a specific product category (shoes, sliders, 
  etc.) unless it's in {{metadata.product_name}} or 
  {{metadata.product_description}}.
- NEVER make up product specs. Only use 
  {{metadata.product_description}} for features.
- If asked something not in the variables:
  English: \"That's a great question — let me have our 
  product team follow up with you on that!\"
  Hindi: \"Bahut achha sawaal hai — main hamari product 
  team se aapko follow up karwati hoon!\"
- You are NOT a robot. You are Maya — a real, caring, 
  sharp customer executive powered by AI.""")
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
            description="Use the customer's name from {{metadata.customer_name}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: PhoneNumberTask(agent_instructions=_ai, extra_tools=_tools),
            id="phone_number",
            description="Use the phone_number from {{metadata.phone_number}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ProductNameTask(agent_instructions=_ai, extra_tools=_tools),
            id="product_name",
            description="Use the product_name from {{metadata.product_name}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ProductPriceTask(agent_instructions=_ai, extra_tools=_tools),
            id="product_price",
            description="Use the product_price from {{metadata.product_price}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: OrderIdTask(agent_instructions=_ai, extra_tools=_tools),
            id="order_id",
            description="Use the order_id from {{metadata.order_id}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: DeliveryDaysTask(agent_instructions=_ai, extra_tools=_tools),
            id="delivery_days",
            description="Use the delivery_days from {{metadata.delivery_days}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: CallOutcomeTask(agent_instructions=_ai, extra_tools=_tools),
            id="call_outcome",
            description="Set this based on how the call ended. Use \"confirmed\" if the customer",
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