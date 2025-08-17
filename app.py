import re
import json
import os
import streamlit as st
from typing import Dict, Any, Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# =========================
# Streamlit Page Settings
# =========================
st.set_page_config(page_title="Pakistan Law Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Pakistan Law Chatbot")
st.caption("Answers about the Constitution and laws of Pakistan. Structured. Reliable. Useful.")


# =========================
# Session State
# =========================
if "history" not in st.session_state:
    st.session_state.history = []


# =========================
# Secrets / API Key
# =========================
if "api_key" not in st.secrets:
    st.warning("Missing `api_key` in Streamlit secrets. Add it as `api_key` to use the LLM fallback.")
GOOGLE_API_KEY = st.secrets.get("api_key", "")


# =========================
# Knowledge Base (Fallback)
# =========================
# Minimal built-in KB to prevent classic mix-ups. You can replace/extend this via constitution.json
DEFAULT_KB: Dict[str, Dict[str, Any]] = {
    "176": {
        "title": "Constitution of Supreme Court",
        "text": ("The Supreme Court shall consist of a Chief Justice, to be known as the Chief Justice of Pakistan, "
                 "and so many other Judges as may be determined by Act of Majlis-e-Shoora (Parliament) or, "
                 "until so determined, as may be fixed by the President."),
        "summary": "Sets the composition of the Supreme Court (CJP + other judges determined by Parliament or temporarily by the President).",
        "examples": [
            "Parliament may increase the number of Supreme Court judges to reduce backlog—permitted under Article 176.",
            "Until Parliament fixes a number, the President may set the number of judges."
        ],
        "related": ["175", "177", "178"]
    },
    "89": {
        "title": "Power of President to promulgate Ordinances",
        "text": ("The President may promulgate an Ordinance when the National Assembly is not in session if satisfied "
                 "that circumstances exist which render it necessary to take immediate action. Ordinances have the force "
                 "of law but must be laid before the Assembly and lapse after set durations unless extended or replaced by Act."),
        "summary": "Allows temporary legislation by the President when NA is not in session, subject to duration/laying/approval rules.",
        "examples": [
            "During an urgent fiscal situation while NA is not in session, the President may issue a tax-related Ordinance.",
            "On Assembly resumption, the Ordinance must be laid; if disapproved or time lapses, it ceases to operate."
        ],
        "related": ["70", "75", "127", "128"]
    },
    "128": {
        "title": "Power of Governor to promulgate Ordinances",
        "text": ("The Governor of a Province may promulgate Ordinances when the Provincial Assembly is not in session, "
                 "subject to conditions similar to those for Presidential Ordinances."),
        "summary": "Provincial analogue of Article 89 for Governors when Provincial Assemblies are not in session.",
        "examples": [
            "A provincial public health emergency may be addressed via a Governor’s Ordinance pending Assembly session.",
            "If the Provincial Assembly disapproves, the Ordinance ceases."
        ],
        "related": ["89", "130"]
    },
    "175": {
        "title": "Establishment and jurisdiction of courts",
        "text": "There shall be a Supreme Court of Pakistan, a High Court for each Province, and such other courts as may be established by law.",
        "summary": "Establishes the court system and judicial structure in Pakistan.",
        "examples": [
            "Inter-provincial disputes fall within the higher judiciary’s constitutional scheme framed by Article 175.",
            "Law students use 175 as the starting point for the hierarchy of courts."
        ],
        "related": ["176", "191"]
    },
    "177": {
        "title": "Appointment of Supreme Court Judges",
        "text": "(Appointment provisions for the Chief Justice and other judges of the Supreme Court—see full text for detail).",
        "summary": "Covers appointment of the CJP and other Supreme Court judges.",
        "examples": [
            "New judges of the Supreme Court are appointed under Article 177.",
            "Bar associations track appointments through Article 177 procedures."
        ],
        "related": ["176", "178"]
    }
}


@st.cache_data(show_spinner=False)
def load_constitution_json(path: str = "constitution.json") -> Dict[str, Dict[str, Any]]:
    """
    Loads a full constitution KB if available; otherwise returns the DEFAULT_KB.
    Expected JSON format:
    {
      "176": {
        "title": "...",
        "text": "...",
        "summary": "...",
        "examples": ["...", "..."],
        "related": ["177", "178"]
      },
      ...
    }
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge: file entries override defaults; defaults fill gaps.
            merged = {**DEFAULT_KB, **data}
            return merged
        except Exception as e:
            st.error(f"Failed to load {path}: {e}")
            return DEFAULT_KB
    return DEFAULT_KB


KB = load_constitution_json()


# =========================
# Utilities
# =========================
ARTICLE_PATTERN = re.compile(r"\barticle\s+(\d{1,3})\b", re.IGNORECASE)

def extract_article_number(text: str) -> Optional[str]:
    m = ARTICLE_PATTERN.search(text or "")
    if m:
        return str(int(m.group(1)))  # normalize like "001"->"1"
    # fallback: if user typed just a number
    only_num = re.search(r"\b(\d{1,3})\b", text or "")
    if only_num and "article" in text.lower():
        return str(int(only_num.group(1)))
    return None


def render_article_response(num: str, entry: Dict[str, Any]) -> str:
    title = entry.get("title", "").strip()
    text = entry.get("text", "").strip()
    summary = entry.get("summary", "").strip()
    examples: List[str] = entry.get("examples", [])
    related: List[str] = entry.get("related", [])

    lines = []
    lines.append(f"### 1) Correct Article/Law\n**Article {num} – {title}**")
    if text:
        lines.append("\n### 2) Authentic Wording or Summary")
        # If the official wording is short, show it; else show the summary first.
        if len(text) <= 600:
            lines.append(f"**Authentic Wording:** {text}")
            if summary:
                lines.append(f"\n**Summary:** {summary}")
        else:
            if summary:
                lines.append(f"**Summary:** {summary}")
            lines.append("\n*(Full text is lengthy; refer to the official Gazette text.)*")

    lines.append("\n### 3) Simple Explanation")
    # A compact, user-friendly explanation (kept generic here; you can specialize per entry in KB)
    if num == "176":
        lines.append("This Article sets how the Supreme Court is composed: one Chief Justice of Pakistan plus other judges. "
                     "Parliament decides the number of judges; until it does, the President may fix the number. "
                     "*(سادہ اردو: اس آرٹیکل میں سپریم کورٹ کی تشکیل کا طریقہ بتایا گیا ہے—چیف جسٹس اور دیگر ججز کی تعداد پارلیمنٹ یا عارضی طور پر صدر مقرر کر سکتے ہیں.)*")
    elif num == "89":
        lines.append("This Article empowers the President to issue temporary laws (Ordinances) when the National Assembly is not in session, "
                     "subject to time limits and laying before the Assembly for approval/disapproval. "
                     "*(سادہ اردو: جب قومی اسمبلی کا اجلاس نہ ہو تو صدر وقتی قانون بطور آرڈیننس جاری کر سکتے ہیں، جو مخصوص مدت کے بعد ختم ہو سکتا ہے جب تک پارلیمنٹ اسے منظور نہ کر دے.)*")
    else:
        lines.append(summary or "This provision applies within Pakistan’s constitutional framework. See examples below.")

    if examples:
        lines.append("\n### 4) Practical Example(s) / Scenario(s)")
        for i, ex in enumerate(examples[:2], start=1):
            lines.append(f"- {ex}")

    if related:
        lines.append("\n### 5) Related Provisions")
        lines.append(", ".join([f"Article {r}" for r in related]))

    return "\n".join(lines)


def handle_article_query(user_input: str) -> Optional[str]:
    """
    If the input refers to an Article and exists in KB → return formatted response.
    If input refers to an Article but not in KB → return a polite uncertainty message.
    Otherwise return None (so the caller can route to LLM).
    """
    num = extract_article_number(user_input)
    if not num:
        return None

    # If user references an Article explicitly, we must verify before denying existence.
    entry = KB.get(num)
    if entry:
        return render_article_response(num, entry)
    else:
        # Polite uncertainty with repair suggestions (common confusions: 89 vs 176 vs 128)
        suggestions = []
        if num in {"175", "176", "177", "178"}:
            suggestions = ["176", "177", "175", "178"]
        else:
            suggestions = ["89", "128", "175", "176", "177"]

        hint = ", ".join([f"Article {s}" for s in suggestions])
        return (
            "I’m not fully certain without checking the official text for that Article number. "
            "If you can share the subject matter (e.g., ‘ordinances’, ‘composition of Supreme Court’), "
            f"I can match it precisely. Likely relevant: {hint}."
        )


# =========================
# LLM Model & Prompt
# =========================
# Use a low temperature for consistency
chat_model = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    temperature=0.2
)

system_prompt = SystemMessage(
    content=(
        """ROLE
You are a legal assistant specialized only in the Constitution and laws of Pakistan.

SCOPE (hard limit)
- If the question is not about Pakistan’s law or Constitution, reply exactly:
  "Sorry, I can only provide information related to laws in Pakistan."

VERIFICATION (very important)
- Before stating that an Article does not exist, first double-check against the Constitution of the Islamic Republic of Pakistan, 1973 (as amended).
- If you cannot reliably verify, say: "I’m not fully certain without checking the text," then ask for the subject matter and suggest likely Article(s) with brief reasoning.
- Never invent section numbers, case names, dates, or figures.

WHEN ASKED ABOUT A CONSTITUTIONAL ARTICLE
1) Confirm or repair the Article number (e.g., user confuses 176 with 89).
2) Provide the exact wording (if short) or an authentic, faithful summary (if long).
3) Give a simple explanation (use plain English; add Urdu gloss where helpful).
4) Provide 1–2 practical scenarios useful for advocates, judges, or law students.
5) If the user’s number is wrong, politely correct it and then answer with the correct Article.

WHEN ASKED ABOUT A PAKISTANI STATUTE/LAW (criminal, family, property, labor, cyber, contract, etc.)
- Explain in simple language, note key elements/thresholds, and typical remedies/penalties.
- Add 1–2 practical scenarios (common applications, pitfalls, practice tips).
- If multiple interpretations exist, note the main views and controlling provisions.

FORMAT (always use this structure)
1. Correct Article/Law
2. Authentic Wording or Summary
3. Simple Explanation
4. Practical Example(s) / Scenario(s)
5. Related Provisions (if any)

STYLE
- Polite, professional, concise.
- Prefer numbered or bulleted structure.
- If unsure on any fact, explicitly say you’re not fully certain rather than guessing.
"""
    )
)

chat_template = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
])

output_parser = StrOutputParser()

def get_history(_):
    return st.session_state.get("history", [])

runnable_get_history = RunnableLambda(get_history)

chain = RunnablePassthrough.assign(
    chat_history=runnable_get_history
) | chat_template | chat_model | output_parser


# =========================
# UI: Sidebar History
# =========================
st.sidebar.title("Chat History")
if st.session_state.history:
    for msg in st.session_state.history[-20:]:
        role = "🧑‍💼 User" if isinstance(msg, HumanMessage) else "🤖 AI"
        st.sidebar.write(f"{role}: {msg.content}")


# =========================
# Main Input
# =========================
user_input = st.text_input("Enter your question (e.g., 'Article 176', 'What is khula under Pakistani law?')")

col1, col2 = st.columns([1, 3])
with col1:
    ask = st.button("Ask")

with col2:
    st.write("")  # just spacing


# =========================
# Routing & Answer
# =========================
if ask:
    if user_input and user_input.strip():
        with st.spinner("Thinking..."):
            # First, try the Article KB path for reliability
            kb_answer = handle_article_query(user_input)
            if kb_answer is not None:
                response = kb_answer
            else:
                # Non-article (or general law) → go to LLM with strong guardrails
                # Hard scope check: if clearly out of Pakistan law, reply with scope message (extra safety)
                if not re.search(r"\b(pakistan|pakistani|constitution|article|ppc|crpc|family|nikah|khula|court|high court|supreme court|ordinance|act|law|bylaws|labour|cyber|pta|fbr|nab|ipc)\b", user_input, re.IGNORECASE):
                    response = 'Sorry, I can only provide information related to laws in Pakistan.'
                else:
                    response = chain.invoke({"human_input": user_input})

        # Display response
        st.success("Answer:")
        st.markdown(response)

        # Save history
        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=response))
    else:
        st.warning("Please enter a question.")


# =========================
# Helper: How to expand KB
# =========================
with st.expander("ℹ️ How to add/expand the Constitution knowledge base"):
    st.markdown(
        """
**Option A (Recommended):** Create a `constitution.json` in the app folder with entries like:

```json
{
  "176": {
    "title": "Constitution of Supreme Court",
    "text": "The Supreme Court shall consist of a Chief Justice of Pakistan and so many other Judges as may be determined by Act of Majlis-e-Shoora (Parliament) or, until so determined, as may be fixed by the President.",
    "summary": "Sets the composition of the Supreme Court.",
    "examples": [
      "Parliament may increase the number of Supreme Court judges to reduce backlog.",
      "President may temporarily fix the number until Parliament decides."
    ],
    "related": ["175", "177", "178"]
  }
}
"""
    )


