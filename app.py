
import streamlit as st
from typing import TypedDict
import re
import json
import os
import sqlite3
import uuid
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# --- IMPORTANT: Set up API Key and Database ---
os.environ["GROQ_API_KEY"] = "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"
DB_FILE = "hiring_assistant_gdpr.db"

# Initialize ChatGroq LLM
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)

# Connect to SQLite
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS hiring_state (
    id TEXT PRIMARY KEY,
    state_json TEXT
)
""")
conn.commit()

# ---------------------------
# Typed state definition
# ---------------------------
class HiringState(TypedDict, total=False):
    ful_name: str
    email_id: str
    phone_number: str
    yoe: int
    desired_posiiton: str
    current_location: str
    tech_stack: list[str]
    messages: list[dict]
    end: bool
    generated_questions: dict
    generated_questions_list: list[str]
    responses: list[dict]
    question_index: int

# ---------------------------
# Configuration
# ---------------------------
CONVERSATION_END_KEYWORDS = {"exit", "quit", "bye", "goodbye", "thanks", "thank you", "stop"}
MAX_RETRIES = 2

# ---------------------------
# Utility helpers
# ---------------------------
def is_end_message(text: str) -> bool:
    lower = text.strip().lower()
    return any(tok in lower for tok in CONVERSATION_END_KEYWORDS)

def simple_email_validator(text: str) -> bool:
    pattern = r"[^@\s]+@[^@\s]+\.[^@\s]+"
    return re.search(pattern, text) is not None

def simple_phone_validator(text: str) -> bool:
    digits = re.sub(r"[^0-9]", "", text)
    return 7 <= len(digits) <= 15

def parse_yoe(text: str) -> int | None:
    m = re.search(r"(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def call_llm(prompt: str, system: str | None = None) -> str:
    msg_seq = []
    if system:
        msg_seq.append(AIMessage(content=system))
    msg_seq.append(HumanMessage(content=prompt))
    try:
        result = llm.invoke(msg_seq)
        if hasattr(result, "content"):
            return getattr(result, "content")
        return str(result)
    except Exception as e:
        return f"(LLM error) {e}"

def save_state_gdpr(state: dict) -> str:
    state_copy = dict(state)
    if state_copy.get("email_id"):
        state_copy["email_id"] = None
    if state_copy.get("phone_number"):
        state_copy["phone_number"] = None
    record_id = str(uuid.uuid4())
    state_json = json.dumps(state_copy)
    c.execute("INSERT INTO hiring_state (id, state_json) VALUES (?, ?)", (record_id, state_json))
    conn.commit()
    return record_id

# ---------------------------
# Node implementations (modified for Streamlit)
# ---------------------------
def get_fullname(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    if not user_input:
        state["messages"].append({"role": "assistant", "content": "Please provide your full name (first and last)."})
        return state, False
    state["ful_name"] = user_input
    state["messages"].append({"role": "assistant", "content": f"Thank you, {user_input}."})
    return state, True

def get_email(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    if not user_input:
        return state, False
    if not simple_email_validator(user_input):
        state["messages"].append({"role": "assistant", "content": "That doesn't look like a valid email address. Please provide an email like 'name@example.com'."})
        return state, False
    state["email_id"] = user_input
    return state, True

def get_phoneno(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    # Added a check to prevent AttributeError if user_input is None
    if not user_input:
        return state, False
    if user_input.lower() == "skip":
        state["phone_number"] = ""
        state["messages"].append({"role": "assistant", "content": "Understood. Skipping phone number."})
        return state, True
    if not simple_phone_validator(user_input):
        state["messages"].append({"role": "assistant", "content": "That phone number doesn't look valid. Please enter digits with optional + and spaces, or type 'skip'."})
        return state, False
    state["phone_number"] = user_input
    return state, True

def get_yoe(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    # Added a check to prevent TypeError if user_input is None
    if not user_input:
        return state, False
    y = parse_yoe(user_input)
    if y is None:
        state["messages"].append({"role": "assistant", "content": "I couldn't extract a number of years. Please enter a single integer like '2' or '5'."})
        return state, False
    state["yoe"] = y
    return state, True

def get_desrired_pos(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    if not user_input:
        state["messages"].append({"role": "assistant", "content": "Please enter your desired position."})
        return state, False
    state["desired_posiiton"] = user_input
    return state, True

def get_current_loc(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    if not user_input:
        state["messages"].append({"role": "assistant", "content": "Please enter your current location."})
        return state, False
    
    parts = [p.strip() for p in user_input.split(',')]
    if len(parts) < 2:
        state["messages"].append({"role": "assistant", "content": "Please provide a valid location in the format 'City, Country'."})
        return state, False

    state["current_location"] = user_input
    return state, True

def get_tech_stack(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    # Added a check to prevent TypeError if user_input is None
    if not user_input:
        return state, False
    items = [t.strip() for t in re.split(r",|;|\\n", user_input) if t.strip()]
    if not items:
        state["messages"].append({"role": "assistant", "content": "Please provide a comma-separated list of technologies you're proficient in."})
        return state, False
    state["tech_stack"] = items
    return state, True

def ask_and_record_questions(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    questions = state.get("generated_questions_list", [])
    q_idx = state.get("question_index", 0)

    # First turn: Generate and ask the first question
    if q_idx == 0 and not questions:
        if not state.get("tech_stack"):
            state["messages"].append({"role": "assistant", "content": "No tech stack provided — skipping technical questions."})
            return state, True
        
        techs = ", ".join(state["tech_stack"])
        prompt = (
            f"You are an expert technical interviewer. Based on the candidate's tech stack: {techs}. "
            "Generate exactly 3 diverse technical interview questions that together assess the candidate's practical skills and depth across the listed technologies. "
            "Output ONLY the questions, numbered 1, 2, and 3. Do not include any introductory or concluding text, or any thoughts or conversational phrases."
        )
        
        # Add a message to indicate the generation is starting
        state["messages"].append({"role": "assistant", "content": "I've collected your information. I will now generate technical questions for you."})
        
        # This is where the loading spinner is displayed
        with st.spinner("Generating questions..."):
            q_text = call_llm(prompt)
        
        state["generated_questions"] = {"raw": q_text, "for_techs": state.get("tech_stack", [])}

        parsed = [line.strip() for line in re.findall(r"^\s*\d+[\).:-]?\s*(.+)", q_text, re.MULTILINE)]
        if len(parsed) < 3:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", q_text) if s.strip()]
            parsed = [s for s in sentences if re.search(r"\w", s)][:3]
        while len(parsed) < 3:
            parsed.append("(No question generated)")

        state["generated_questions_list"] = parsed
        state["responses"] = []
        questions = state.get("generated_questions_list", [])
        
        if questions:
            state["messages"].append({"role": "assistant", "content": f"Question 1: {questions[0]}"})
            state["question_index"] = 1
        return state, False
    
    # Subsequent turns: Record the answer and ask the next question
    if user_input:
        state["responses"].append({"question": questions[q_idx - 1], "answer": user_input})

    if state["question_index"] >= len(questions):
        state["messages"].append({"role": "assistant", "content": "Thanks — I've recorded your answers."})
        return state, True
    
    next_q = questions[state["question_index"]]
    state["messages"].append({"role": "assistant", "content": f"Question {state['question_index'] + 1}: {next_q}"})
    state["question_index"] += 1
    return state, False

def conclude_conversation(state: HiringState, user_input: str) -> tuple[HiringState, bool]:
    # Changed the final message to be a hardcoded string as requested.
    state["messages"].append({"role": "assistant", "content": "Thanks — I've recorded your answers."})
    state["end"] = True
    return state, True

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="TalentScout Hiring Assistant", 
    page_icon="🤖",
    layout="wide"
)
st.title("TalentScout Hiring Assistant 🤖")

# Custom CSS for UI enhancements
st.markdown("""
<style>
    /* General body and container styling */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }

    /* Chat message styling */
    .st-chat-message-container {
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 16px;
    }

    /* Assistant messages */
    .st-chat-message-container.assistant {
        background-color: #e3f2fd;
        border-top-right-radius: 0;
    }

    /* User messages */
    .st-chat-message-container.user {
        background-color: #bbdefb;
        border-top-left-radius: 0;
        text-align: right;
    }

    /* Chat input styling */
    .st-chat-input-container {
        padding: 10px;
        background-color: #ffffff;
        border-top: 1px solid #e0e0e0;
        position: sticky;
        bottom: 0;
        z-index: 10;
        box-shadow: 0 -4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize chat history and state in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "hiring_state" not in st.session_state:
    st.session_state.hiring_state = {
        "messages": [],
        "ful_name": "",
        "email_id": "",
        "phone_number": "",
        "yoe": 0,
        "desired_posiiton": "",
        "current_location": "",
        "tech_stack": [],
        "end": False,
        "generated_questions": {},
        "generated_questions_list": [],
        "responses": [],
        "question_index": 0,
    }
if "current_step_index" not in st.session_state:
    st.session_state.current_step_index = 0
if "conversation_complete" not in st.session_state:
    st.session_state.conversation_complete = False

# Mapping of steps and prompts for a guided conversation
steps = [
    ("ful_name", "Hi! To get started, may I please have your full name (first + last)?", get_fullname),
    ("email_id", "Please share your email address so we can contact you about opportunities.", get_email),
    ("phone_number", "Could you provide a phone number we can reach you on? Include country code if applicable.", get_phoneno),
    ("yoe", "How many years of professional experience do you have? (e.g., 3)", get_yoe),
    ("desired_posiiton", "What role(s) are you interested in? (e.g., Backend Engineer, Data Scientist)", get_desrired_pos),
    ("current_location", "Where are you currently located? (City, Country)", get_current_loc),
    ("tech_stack", "Please list the technologies you're proficient in — programming languages, frameworks, databases, and tools. Separate items with commas. Example: Python, Django, PostgreSQL, Docker", get_tech_stack),
    ("ask_and_record_questions", None, ask_and_record_questions),
    ("conclude_conversation", None, conclude_conversation),
]

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initial welcome message
if not st.session_state.chat_history:
    welcome_msg = "👋 Welcome to TalentScout! I'm your intelligent hiring assistant. I'll collect some basic details and then ask tailored technical questions. Let's get started! 🚀"
    st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
    st.session_state.chat_history.append({"role": "assistant", "content": steps[0][1]})
    st.rerun()

# Get user input
user_input = st.chat_input("Enter your response...")
last_step_completed = st.session_state.get("last_step_completed", False)

if user_input or last_step_completed:
    # Append user message to the history if available
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Check for end keywords
    if user_input and is_end_message(user_input):
        st.session_state.conversation_complete = True
        st.session_state.chat_history.append({"role": "assistant", "content": "Thanks for your time. Goodbye!"})
        st.rerun()

    h_state = st.session_state.hiring_state
    
    # Process steps in a loop until a step requires user input or the conversation ends
    while not st.session_state.conversation_complete and (user_input or last_step_completed):
        step_key, step_prompt, step_func = steps[st.session_state.current_step_index]
        
        new_state, step_completed = step_func(h_state, user_input)
        st.session_state.hiring_state = new_state
        
        new_messages = [msg for msg in st.session_state.hiring_state["messages"] if msg not in st.session_state.chat_history]
        if new_messages:
            for msg in new_messages:
                st.session_state.chat_history.append(msg)
            st.session_state.hiring_state["messages"] = []
        
        st.session_state.last_step_completed = step_completed
        user_input = None # Consume the user input so it's not reused in the loop

        if step_completed:
            st.session_state.current_step_index += 1

            if st.session_state.current_step_index >= len(steps):
                st.session_state.conversation_complete = True
            
            if not st.session_state.conversation_complete:
                next_step_key, next_step_prompt, next_step_func = steps[st.session_state.current_step_index]
                
                if next_step_prompt:
                    st.session_state.chat_history.append({"role": "assistant", "content": next_step_prompt})
                else:
                    # If the next step has no prompt, it means it should execute immediately
                    # We will continue the while loop to process it
                    continue
        break # Exit the loop after processing one step, whether completed or not

    st.rerun()

# Save the final state if the conversation is complete
if st.session_state.conversation_complete and not st.session_state.get("state_saved"):
    try:
        record_id = save_state_gdpr(st.session_state.hiring_state)
        st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Conversation state stored safely with UUID: `{record_id}`"})
        st.session_state.state_saved = True
        st.rerun()
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"Failed to save state: {e}"})
        st.rerun()
app.py
Displaying app.py.