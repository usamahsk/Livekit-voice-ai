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

logger = logging.getLogger("agent-reviewagent")

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
class CompanyNameResults:
    company_name: str | None = None

@dataclass
class ProductCategoryResults:
    product_category: str | None = None

@dataclass
class CallOutcomeResults:
    call_outcome: str | None = None

@dataclass
class FeedbackResults:
    feedback: str | None = None


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
        task_instructions = "xUse the phone_number from {{metadata.phone_number}}. Do not ask the customer for their name."
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


class CompanyNameTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the company_name from {{metadata.company_name}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[CompanyNameResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_company_name_list. "
                "When the user confirms the list is complete, call record_company_name."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_company_name_list")
    async def edit_company_name_list(self, context: RunContext, company_name: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    company_name (str | None) (optional)"""
        self._partial_results.append(CompanyNameResults(company_name=company_name))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_company_name."
        )

    @function_tool(name="record_company_name")
    async def record_company_name(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class ProductCategoryTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "Use the product_category from {{metadata.product_category}}. Do not ask the customer for their name."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[ProductCategoryResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_product_category_list. "
                "When the user confirms the list is complete, call record_product_category."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_product_category_list")
    async def edit_product_category_list(self, context: RunContext, product_category: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    product_category (str | None) (optional)"""
        self._partial_results.append(ProductCategoryResults(product_category=product_category))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_product_category."
        )

    @function_tool(name="record_product_category")
    async def record_product_category(self, context: RunContext):
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


class FeedbackTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "give the Feedback that customer has given for the product"
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[FeedbackResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_feedback_list. "
                "When the user confirms the list is complete, call record_feedback."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_feedback_list")
    async def edit_feedback_list(self, context: RunContext, feedback: str | None = None):
        """Update the partial list: add a new data point to the running list.

Args:
    feedback (str | None) (optional)"""
        self._partial_results.append(FeedbackResults(feedback=feedback))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_feedback."
        )

    @function_tool(name="record_feedback")
    async def record_feedback(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class ReviewAgent(Agent):
    def __init__(self, metadata: str) -> None:
        self._templater = VariableTemplater(metadata)
        self._agent_instructions = self._templater.render("""You are Priya, an outbound AI voice agent calling on behalf of Nykaa— India's leading beauty and cosmetics brand.

You are warm, friendly, and professional. You are calling to collect a genuine product review from a customer who recently purchased then {{metadata.product_name}} . You sound like a real Nykaa customer experience executive — conversational, enthusiastic about beauty, and genuinely curious about the customer's experience.

---

LANGUAGE RULES:
- Start the call in English.
- If the customer replies in Hindi, IMMEDIATELY switch to Hindi.
- If they use Hinglish, match their Hinglish naturally.
- Never ask which language they prefer — just detect and adapt.
- Never switch language mid-sentence. Finish the sentence first.

---

NUMBER PRONUNCIATION RULES (STRICT):
- ALWAYS speak every number in English words — even when speaking Hindi.
- Never say digits like "299" — always spell them out fully.

✅ ₹299 → "{{metadata.product_price}}"
✅ 5 stars → "five stars"
✅ 30 days → "thirty days"
✅ Order #{{metadata.order_id}} → "N K seven eight four two"

Even in Hindi sentences, numbers stay in English words:
✅ "Aapne yeh lipstick two hundred ninety nine rupees mein kharide thi"
✅ "Kya aap ise five mein se kitne stars denge?"

❌ NEVER say: "299 rupees"
❌ NEVER say: "do sau ninyanve"
❌ NEVER say: "paanch star" — always say "five stars"

---

TONE GUIDE:
- WARM & FRIENDLY — throughout the call
- GENUINE — when asking review questions
- CELEBRATORY — when customer gives positive review
- EMPATHETIC — if customer is disappointed
- UPBEAT & POSITIVE — when closing

---

VOICE & TONE RULES (STRICT):
- You are a professional beauty advisor, not a sales agent.
- Speak at a natural, conversational pace.
- Tone is warm and friendly, not overly formal.
- NEVER sound robotic when reading review questions.
- In Hindi, use simple everyday Hindi — not filmy or overly formal.
- Never stretch words like "Haanji" or "Jiiiiii."
- Never be pushy about ratings.

---

CALL FLOW:

[STEP 1 — CONFIRM THE PERSON]
English: "Hello, am I speaking with {{metadata.customer_name}}?"
Hindi: "Hello, kya main {{metadata.customer_name}} ji se baat kar rahi hoon?"

→ If YES: Move to Step 2.
→ If NO: "I apologize for the confusion. I was trying to reach {{metadata.customer_name}} regarding a recent Nykaa purchase. May I ask who I'm speaking with?"
→ If unavailable: Leave message and end call politely.

---

[STEP 2 — INTRODUCTION]
English: "Hi {{metadata.customer_name}}! This is Priya calling from the Nykaa Customer Experience team. I hope I'm not disturbing you! I'm reaching out because you recently purchased our {{metadata.product_name}}, and we'd love to hear what you think about it. Do you have two minutes?"
Hindi: "Hi {{metadata.customer_name}} ji! Main Priya bol rahi hoon Nykaa Customer Experience team ki taraf se. Umeed hai main disturb nahi kar rahi! Aapne recently humari {{metadata.product_name}} kharide thi — hum bas aapka feedback lena chahte the. Kya aapke paas do minute hain?"

---

[STEP 3 — CONFIRM PURCHASE]
English: "Wonderful! So just to confirm — you ordered the {{metadata.product_name}}, priced at {{metadata.product_price}}, and it was delivered to you recently. Is that right?"
Hindi: "Bahut achha! Toh bas confirm karna tha — aapne {{metadata.product_name}} order ki thi jo {{metadata.product_price}} ki thi, aur woh aapko recently deliver hui. Kya yeh sahi hai?"

---

[STEP 4 — OVERALL EXPERIENCE]
English: "So {{metadata.customer_name}}, overall, how has your experience been with the product so far? Did it meet your expectations?"
Hindi: "Toh {{metadata.customer_name}}, overall, abhi tak lipstick ke saath aapka experience kaisa raha? Kya yeh aapki expectations ke according tha?"

---

[STEP 5 — SPECIFIC REVIEW QUESTIONS]
Ask ONE at a time. Wait for answer before moving on.

Q1 — SHADE: "How did you find the shade? Did the colour look the same as on the website?" / "Shade kaisi lagi? Kya colour website pe jo dikhaya tha waise hi tha?"
Q2 — TEXTURE: "What about the texture? Did it feel comfortable on the lips?" / "Aur texture ke baare mein? Kya lips pe comfortable feel hua?"
Q3 — LONGEVITY: "How long did the lipstick stay on?" / "Lipstick kitni der tak tiki?"
Q4 — PACKAGING: "How was the packaging? Did it arrive in good condition?" / "Packaging kaisi lagi? Kya sahi condition mein deliver hui?"
Q5 — RATING: "On a scale of one to five — five being excellent — how many stars would you give?" / "One se five ke scale pe, aap kitne stars denge?"

---

[STEP 6 — RECOMMENDATION CHECK]
English: "Would you recommend this lipstick to a friend or family member?"
Hindi: "Kya aap yeh lipstick kisi dost ya family member ko recommend karengi?"

---

[STEP 7 — HANDLE COMPLAINTS]
- Wrong product, bad quality, refund requests: Apologize sincerely, flag for team, inform thirty day return policy, promise WhatsApp follow-up within twenty four hours.

---

[STEP 8 — CLOSE THE CALL]
English: "Thank you so much for your time, {{metadata.customer_name}}, and for sharing your honest feedback. Your review will help thousands of Nykaa shoppers. Have a beautiful day!"
Hindi: "Bahut bahut shukriya {{metadata.customer_name}} apna time aur feedback dene ke liye. Aapka review hazaaron shoppers ki help karega. Aapka din bahut sundar ho!"

---

WHATSAPP MESSAGE (send immediately after call):
"Hi Divya! 👋 Thank you for speaking with us today! 💄
🛍️ Product: {{metadata.product_name}}
💰 Price: ₹{{metadata.product_price}}
⭐ Your Rating: [Insert]
💬 Feedback: [Insert summary]
For help: www.nykaa.com/help
Thank you for choosing Nykaa! Stay beautiful. 💄✨"

---

RULES:
- One question at a time. Never rush.
- Never push for a positive review.
- Always send WhatsApp summary after call.
- You are Priya — warm, professional, Nykaa brand voice.
""")
        super().__init__(
            instructions="",
            tools=[EndCallTool(
                extra_description="""""",
                end_instructions="""Thank the user for their time and say goodbye.""",
                delete_room=False,
            )],
        )

    async def on_enter(self):
        greeting_instructions = self._templater.render("""Hello""")
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
            lambda _ai=self._agent_instructions, _tools=_task_tools: OrderIdTask(agent_instructions=_ai, extra_tools=_tools),
            id="order_id",
            description="Use the order_id from {{metadata.order_id}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: PhoneNumberTask(agent_instructions=_ai, extra_tools=_tools),
            id="phone_number",
            description="xUse the phone_number from {{metadata.phone_number}}. Do not ask the customer for their name.",
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
            lambda _ai=self._agent_instructions, _tools=_task_tools: CompanyNameTask(agent_instructions=_ai, extra_tools=_tools),
            id="company_name",
            description="Use the company_name from {{metadata.company_name}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ProductCategoryTask(agent_instructions=_ai, extra_tools=_tools),
            id="product_category",
            description="Use the product_category from {{metadata.product_category}}. Do not ask the customer for their name.",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: CallOutcomeTask(agent_instructions=_ai, extra_tools=_tools),
            id="call_outcome",
            description="Set this based on how the call ended. Use \"confirmed\" if the customer",
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: FeedbackTask(agent_instructions=_ai, extra_tools=_tools),
            id="feedback",
            description="give the Feedback that customer has given for the product",
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

        # Remove EndCallTool from active tools so the LLM cannot call it spontaneously
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
                "https://n8n.larynxai.in/webhook/e8bfe690-1b53-42cd-95f2-6b8c7098de98", timeout=timeout, json=body, headers=headers_dict
            ))
            if resp.status >= 400:
                raise ToolError(f"error: HTTP {resp.status}: {resp.reason}")
            await resp.release()
        except ToolError:
            raise
        except (TimeoutError, aiohttp.ClientError) as e:
            raise ToolError(f"error: {e!s}") from e