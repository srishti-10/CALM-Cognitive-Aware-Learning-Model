import streamlit as st
import time
from generate_text import generate_llm_like_response

st.set_page_config(page_title="Multi-Conversation Chatbot", layout="wide")

st.title("🤖 Multi-Conversation Chatbot")

# Initialize session state for conversations store and current selected conversation
if "conversations" not in st.session_state:
    st.session_state.conversations = {
        "Conversation 1": []  # start with one empty conversation
    }
if "current_conv" not in st.session_state:
    st.session_state.current_conv = "Conversation 1"

def clear_current_conversation():
    st.session_state.conversations[st.session_state.current_conv] = []

def add_new_conversation():
    # Generate a new unique conversation name
    existing = st.session_state.conversations.keys()
    i = 1
    while f"Conversation {i}" in existing:
        i += 1
    new_name = f"Conversation {i}"
    st.session_state.conversations[new_name] = []
    st.session_state.current_conv = new_name

# Sidebar for selecting conversations
st.sidebar.header("Conversations")

conversation_names = list(st.session_state.conversations.keys())
selected_conv = st.sidebar.radio("Select Conversation", conversation_names, index=conversation_names.index(st.session_state.current_conv))
st.session_state.current_conv = selected_conv

if st.sidebar.button("New Conversation"):
    add_new_conversation()

if st.sidebar.button("Clear Current Conversation"):
    clear_current_conversation()

# Display the messages of the current conversation
st.subheader(f"Chat - {st.session_state.current_conv}")

messages = st.session_state.conversations[st.session_state.current_conv]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is your question?")

if prompt:
    # Add user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate simulated response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(1.5)
            response = generate_llm_like_response(min_words=10, max_words=80)
            st.markdown(response)
    messages.append({"role": "assistant", "content": response})

# Save back messages to state (actually it's mutable so not strictly necessary)
st.session_state.conversations[st.session_state.current_conv] = messages
