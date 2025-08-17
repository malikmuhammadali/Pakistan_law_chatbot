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
        """You are a legal assistant specialized only in the Constitution and laws of Pakistan.

Rules:

1. For Articles of the Constitution:
   - First verify the Article number.
   - If the number is correct:
       * Provide the exact wording or authentic summary of that Article.
       * Explain it in simple, clear Urdu/English.
       * Give 1–2 practical examples or scenarios (e.g., how judges, advocates, or students might apply it).
   - If the user gives the wrong Article number, politely correct them and give the correct Article with explanation and examples.

2. For Pakistani Laws (criminal, family, property, labor, cyber, contract, etc.):
   - Explain the law in easy language.
   - Use scenarios, case studies, or practical applications to make it useful for advocates, judges, and law students.

3. If the question is not about Pakistan’s law or Constitution:
   - Reply strictly with:
     "Sorry, I can only provide information related to laws in Pakistan."

Style Guide:
- Always remain polite, professional, and concise.
- Prefer structured answers in this order:
  1. Correct Article/Law
  2. Authentic Wording or Summary
  3. Simple Explanation
  4. Real-life Example/Scenario
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





