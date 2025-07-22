import streamlit as st
import time
from inference_pipeline import run_inference
from sentence_transformers import SentenceTransformer, util
import re

# Initialize session state variables
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = None
if "topic_model" not in st.session_state:
    st.session_state.topic_model = SentenceTransformer('all-MiniLM-L6-v2')

print("DEBUG: app.py loaded")
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
    print("DEBUG: User prompt is", prompt)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using inference_pipeline
    with st.chat_message("assistant"):
        print("DEBUG: Assistant block entered")

        with st.spinner("Thinking..."):
            # Call your backend inference function
            result = run_inference(prompt)
            response = f"**Emotion:** {result['emotion']}\n\n**Strategy:** {result['strategy']}\n\n**Answer:** {result['answer']}"
            st.markdown(response)
    messages.append({"role": "assistant", "content": response})

# Save back messages to state (actually it's mutable so not strictly necessary)
st.session_state.conversations[st.session_state.current_conv] = messages

shorten_response = None
if messages and messages[-1]["role"] == "assistant":
    last_answer = messages[-1]["content"]
    if st.button("Shorten"):
        try:
            print("DEBUG: Shorten button clicked")
            topic = st.session_state.current_topic or "the previous topic"
            clean_answer = re.sub(r'\*\*.*?\*\*', '', last_answer)
            prompt = (
                f"Topic: {topic}\n\n"
                f"Answer: {clean_answer}\n\n"
                "Summarize the above answer in 3-4 sentences using the CAG strategy. Be concise and do not repeat details. Only output the summary."
            )
            result = run_inference(prompt)
            summary = result['answer']
            st.markdown(
                f'''
                <div style="background-color: #fff; padding: 1em; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 1em;">
                    {summary}
                </div>
                ''',
                unsafe_allow_html=True
            )
        except Exception as e:
            print("ERROR in shorten button:", e)
            st.error(f"Error: {e}")
