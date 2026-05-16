import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🤖",
    layout="centered"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

html, body, [class*="css"]  {

    font-family: 'Segoe UI', sans-serif;

    background: linear-gradient(
        135deg,
        #0F172A,
        #111827,
        #1E293B
    );

    color: white;
}

.main {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
}

.ai-box {

    background: rgba(255,255,255,0.06);

    padding: 35px;

    border-radius: 24px;

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.1);

    box-shadow: 0 8px 32px rgba(0,0,0,0.4);

}

.title {

    text-align: center;

    font-size: 52px;

    font-weight: bold;

    background: linear-gradient(
        90deg,
        #60A5FA,
        #A78BFA,
        #22D3EE
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

}

.subtitle {

    text-align: center;

    color: #CBD5E1;

    margin-top: -10px;

    margin-bottom: 25px;

}

.stTextArea textarea {

    background: rgba(255,255,255,0.05) !important;

    color: white !important;

    border-radius: 18px !important;

    border: 1px solid rgba(255,255,255,0.15) !important;

    padding: 18px !important;

    font-size: 16px !important;

}

.stButton>button {

    width: 100%;

    border-radius: 18px;

    padding: 14px;

    border: none;

    background: linear-gradient(
        90deg,
        #3B82F6,
        #8B5CF6
    );

    color: white;

    font-size: 18px;

    font-weight: bold;

    transition: 0.3s;

}

.stButton>button:hover {

    transform: scale(1.02);

    box-shadow: 0 0 25px rgba(139,92,246,0.6);

}

.summary-box {

    margin-top: 25px;

    padding: 22px;

    border-radius: 18px;

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

}

.summary-title {

    color: #60A5FA;

    font-size: 24px;

    margin-bottom: 10px;

}

.footer {

    text-align: center;

    margin-top: 30px;

    color: #94A3B8;

    font-size: 14px;

}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown('<div class="ai-box">', unsafe_allow_html=True)

st.markdown(
    '<div class="title">🤖 AI Text Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by Transformer AI Technology</div>',
    unsafe_allow_html=True
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    model = T5ForConditionalGeneration.from_pretrained(
        "t5-small"
    )

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small"
    )

    return model, tokenizer

model, tokenizer = load_model()

# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

# =====================================
# CLEAN FUNCTION
# =====================================

def clean_data(text):

    text = re.sub(r"\r\n", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"<.*?>", "", text)

    return text.strip().lower()

# =====================================
# SUMMARIZER FUNCTION
# =====================================

def summarize_dialogue(dialogue):

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)

    attention_mask = inputs["attention_mask"].to(device)

    targets = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=40,
        num_beams=2,
        early_stopping=True
    )

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary

# =====================================
# TEXT AREA
# =====================================

user_input = st.text_area(
    "Enter your long text below",
    height=260,
    placeholder="Paste your article, notes, story or paragraph here..."
)

# =====================================
# BUTTON
# =====================================

# =====================================
# BUTTON
# =====================================

if st.button("⚡ Generate AI Summary"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        with st.spinner("AI is generating summary..."):

            summary = summarize_dialogue(user_input)

        st.markdown(
            """
            <div class="summary-box">
                <div class="summary-title">
                    🧠 AI Generated Summary
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(summary)
