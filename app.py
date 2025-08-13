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
with open(r"C:\Users\aliaw\Desktop\chat_bot\api_key.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()

# -------------------
# Create Chat Model
# -------------------
chat_model = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash"
)

# -------------------
# System Prompt
# -------------------
system_prompt = SystemMessage(
    content=(
        "You are an AI assistant that provides helpful, accurate, and clear information about laws in Pakistan. "
        "If a user asks a question related to Pakistan's legal system, answer it in detail and in a simple, easy-to-understand way. "
        "If the question is about anything outside Pakistan's law, politely respond with: "
        "'Sorry, I can only provide information related to laws in Pakistan.' "
        "Always remain polite, professional, and concise."
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

