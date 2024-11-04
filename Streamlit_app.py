__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from extInfo2 import qa_chain, process_llm_response, store_feedback

# Setting page configuration must be the first Streamlit command used.
st.set_page_config(page_title="Fils Chatbot", page_icon=":robot_face:", layout='wide')

# Custom CSS to style the chatbot interface with chat bubbles
st.markdown("""
<style>
.chatbox {
    max-width: 80%;
    margin: 10px;
    padding: 10px;
    border-radius: 18px;
    color: #fff;
    font-size: 16px;
    line-height: 24px;
}
.chatbot {
    background-color: #007bff;
    align-self: flex-start;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
}
.user {
    background-color: #4caf50;
    align-self: flex-end;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
}
.scrollable-container {
    height: 500px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}
.fixed-input {
    position: fixed;
    bottom: 20px;
    left: 0;
    right: 0;
    padding: 10px;
    background: white;
    box-shadow: 0 -1px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("Fils Chatbot")
st.sidebar.header("Quick Actions")
if st.sidebar.button("Say Hello!"):
    st.sidebar.write("Hello! How can I help you today?")

# Initialize chat history in session state if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def add_to_chat(user_input, bot_response):
    st.session_state.chat_history.append({'user': user_input, 'bot': bot_response})

def get_contextual_prompt(user_input):
    max_tokens_for_context = 4096 - len(user_input.split()) - 1500
    context = ""
    accumulated_tokens = 0

    for turn in reversed(st.session_state.chat_history[-5:]):
        turn_text = f"User: {turn['user']} Bot: {turn['bot']}"
        turn_tokens = len(turn_text.split())
        if accumulated_tokens + turn_tokens > max_tokens_for_context:
            break
        context = turn_text + " " + context
        accumulated_tokens += turn_tokens

    return context.strip()

# Message display area
message_container = st.empty()
with message_container.container():
    # Display previous messages
    for message in st.session_state.chat_history:
        st.markdown(f'<div class="chatbox user">You: {message["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chatbox chatbot">Bot: {message["bot"]}</div>', unsafe_allow_html=True)

# Fixed input area
# Fixed input area
with st.container():
    st.markdown('<div class="fixed-input">', unsafe_allow_html=True)

    # Use a separate key for managing the display of input to avoid conflict
    if 'temp_input' not in st.session_state:
        st.session_state.temp_input = ""

    # Capture input using the temporary state variable
    user_input = st.text_input("", value=st.session_state.temp_input, key="user_input",
                               placeholder="Type your message here...")
    submit_button = st.button("Send")

    if submit_button:
        if user_input:  # Process the input if it is not empty
            context = get_contextual_prompt(user_input)
            full_prompt = context + f" User: {user_input}" if context else user_input
            llm_response = qa_chain(full_prompt)
            bot_response = process_llm_response(llm_response)

            add_to_chat(user_input, bot_response)

            # Reset the temporary input after processing to clear the field
            st.session_state.temp_input = ""

            # Collect feedback right after displaying the response
            feedback = st.radio("Was this answer helpful?", ('Yes', 'No'), key="feedback")
            if st.button("Submit Feedback"):
                store_feedback(user_input, bot_response, feedback)
                st.success("Feedback submitted!")

            # Redisplay the updated chat history
            message_container.empty()
            with message_container.container():
                for message in st.session_state.chat_history:
                    st.markdown(f'<div class="chatbox user">You: {message["user"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chatbox chatbot">Bot: {message["bot"]}</div>', unsafe_allow_html=True)
        else:
            st.session_state.temp_input = ""  # Ensure input is cleared if 'send' is hit without input

    st.markdown('</div>', unsafe_allow_html=True)
