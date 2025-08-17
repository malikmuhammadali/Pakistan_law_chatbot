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
# Knowledge Base (Fallback for Constitution)
# =========================
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
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
        return str(int(m.group(1)))
    only_num = re.search(r"\b(\d{1,3})\b", text or "")
    if only_num and "article" in text.lower():
        return str(int(only_num.group(1)))
    return None


def render_article_response(num: str, entry: Dict[str, Any]) -> str:
    """Format Article response in one single heading block"""
    title = entry.get("title", "").strip()
    text = entry.get("text", "").strip()
    summary = entry.get("summary", "").strip()
    examples: List[str] = entry.get("examples", [])
    related: List[str] = entry.get("related", [])

    details = f"**Article {num} – {title}**\n\n"

    if text:
        details += f"**Authentic Wording:** {text}\n\n"
    if summary:
        details += f"**Detailed Explanation:** {summary}\n\n"
    if examples:
        details += "**Practical Example(s) / Scenario(s):**\n"
        for ex in examples[:2]:
            details += f"- {ex}\n"
        details += "\n"
    if related:
        details += f"**Related Provisions:** {', '.join([f'Article {r}' for r in related])}\n"

    return "### Article Information\n\n" + details


def handle_article_query(user_input: str) -> Optional[str]:
    num = extract_article_number(user_input)
    if not num:
        return None

    entry = KB.get(num)
    if entry:
        return render_article_response(num, entry)
    else:
        suggestions = ["89", "128", "175", "176", "177"]
        hint = ", ".join([f"Article {s}" for s in suggestions])
        return (
            "I’m not fully certain without checking the official text for that Article number. "
            f"If you can share the subject matter (e.g., ‘ordinances’, ‘composition of Supreme Court’), "
            f"I can match it precisely. Likely relevant: {hint}."
        )


# =========================
# LLM Model & Prompt
# =========================
chat_model = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    temperature=0.2
)

system_prompt = SystemMessage(
    content=(
        """ROLE
You are a legal assistant specialized only in the Constitution and laws of Pakistan.

SCOPE
- If the question is not about Pakistan’s law or Constitution, reply exactly:
  "Sorry, I can only provide information related to laws in Pakistan."

VERIFICATION
- Before stating that an Article does not exist, double-check against the Constitution of Pakistan (1973).
- If you cannot reliably verify, say: "I’m not fully certain without checking the text," then ask for the subject matter and suggest likely Article(s).
- Never invent section numbers, case names, dates, or figures.

WHEN ASKED ABOUT A CONSTITUTIONAL ARTICLE
- Provide authentic wording or summary
- Provide 1–2 practical scenarios
- Mention related provisions (if any)

WHEN ASKED ABOUT OTHER PAKISTANI LAWS (PPC, CrPC, Family Law, Cyber, etc.)
- Explain simply
- Note key elements, remedies/penalties
- Provide 1–2 practical scenarios

FORMAT
- Always return under **one main heading** (e.g., "Article Information" or "Law Information")
- Use bold labels for sections inside.

STYLE
- Polite, professional, concise
- Bulleted or short paragraphs
- If unsure, say so explicitly
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
user_input = st.text_input("Enter your question ")

col1, col2 = st.columns([1, 3])
with col1:
    ask = st.button("Ask")

with col2:
    st.write("")


# =========================
# Routing & Answer
# =========================
if ask:
    if user_input and user_input.strip():
        with st.spinner("Thinking..."):
            kb_answer = handle_article_query(user_input)
            if kb_answer is not None:
                response = kb_answer
            else:
                if not re.search(r"\b(pakistan|pakistani|constitution|article|ppc|crpc|family|nikah|khula|court|supreme court|ordinance|act|law|bylaws|labour|cyber|pta|fbr|nab)\b", user_input, re.IGNORECASE):
                    response = 'Sorry, I can only provide information related to laws in Pakistan.'
                else:
                    response = chain.invoke({"human_input": user_input})

        st.success("Answer:")
        st.markdown(response)

        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=response))
    else:
        st.warning("Please enter a question.")

