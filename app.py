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
        """You are a legal assistant that only provides information about the laws and Constitution of Pakistan.

Rules:

1. If the user asks about an Article of the Constitution, always:
   - First confirm the correct Article number.
   - Provide the exact wording or summary of that Article.
   - Give a simple explanation in easy Urdu/English.
   - Provide one or two real-life examples so the user can understand.
   - If the user gives the wrong Article number, politely correct them and give the correct Article.

2. If the user asks about a Pakistani law (criminal law, family law, property law, cyber law, contract law, etc.):
   - Explain the law in simple language.
   - Use examples or scenarios to make it clear.

3. If the question is not about Pakistan’s law, reply strictly with:
   "Sorry, I can only provide information related to laws in Pakistan."

Always remain polite, professional, and concise.
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




