import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ✅ Initialize session state right at the top
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------
# Load API key
# -------------------

GOOGLE_API_KEY = st.secrets["api_key"]
# -------------------
# Create Chat Model
# -------------------
chat_model = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash"
)

# -------------------
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


# -------------------
# Chat Template
# -------------------
chat_template = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
])

# -------------------
# Output Parser
# -------------------
output_parser = StrOutputParser()

# -------------------
# Memory Buffer
# -------------------
def get_history(_):
    # ✅ Safe access
    return st.session_state.get("history", [])

runnable_get_history = RunnableLambda(get_history)

# -------------------
# Build Chain
# -------------------
chain = RunnablePassthrough.assign(
    chat_history=runnable_get_history
) | chat_template | chat_model | output_parser

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Pakistan Law Chatbot", page_icon="⚖️")
st.title("⚖️ Pakistan Law Chatbot")
st.write("Ask me anything about laws in Pakistan.")

# Sidebar Chat History
st.sidebar.title("Chat History")
if st.session_state.history:
    for msg in st.session_state.history:
        role = "🧑‍💼 User" if isinstance(msg, HumanMessage) else "🤖 AI"
        st.sidebar.write(f"{role}: {msg.content}")

# Input Box
user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = chain.invoke({"human_input": user_input})
        
        # Display response
        st.success("Answer:")
        st.write(response)

        # Save to history
        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=response))
    else:
        st.warning("Please enter a question.")






