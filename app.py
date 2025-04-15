import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import urllib.parse

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./decor-gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./decor-gpt2")

st.set_page_config(page_title="DecoStyles-GPT")

# Session state tracking
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Landing Page
if not st.session_state.show_chat:
    st.markdown("""
    # Welcome to DecoStyles-GPT

    This custom-trained AI assistant helps you plan and visualize your perfect event decor.

    Whether you're organizing a wedding, birthday, baby shower, or corporate event, simply describe your vision in plain language, and DecoStyles-GPT will generate personalized style suggestions — plus show you matching inspiration on Pinterest.
    """)
    if st.button("Start Styling"):
        st.session_state.show_chat = True
        st.rerun()

# Chat Interface Page
else:
    st.title("DecoStyles-GPT: Event Decor Style Assistant")

    if not st.session_state.user_input:
        user_input = st.text_input("Describe your event style (e.g., *bohemian baby shower in peach and cream*):")
        if user_input:
            st.session_state.user_input = user_input
            st.rerun()
    else:
        user_input = st.session_state.user_input

        def generate_decor_suggestion(prompt, max_length=100):
            prompt_formatted = prompt.strip() + " ->"
            inputs = tokenizer(prompt_formatted, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=60,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt_formatted):].strip()

        with st.spinner("Generating your custom decor suggestion..."):
            suggestion = generate_decor_suggestion(user_input)

        st.markdown("### Suggested Decor Style")
        st.success(suggestion)

        # Pinterest link
        search_query = urllib.parse.quote(user_input)
        pinterest_url = f"https://www.pinterest.com/search/pins/?q={search_query}"
        st.markdown("### See Visual Inspiration")
        st.markdown(f"[Click here to view Pinterest results for **{user_input}**]({pinterest_url})", unsafe_allow_html=True)

        # Feedback Section
        st.markdown("---")
        st.markdown("### Feedback on Suggestion")

        col1, col2, col3 = st.columns(3)

        with col1:
            acc_feedback = st.radio("Accuracy", ["👍", "👎"], key="acc")
        with col2:
            rel_feedback = st.radio("Relevance", ["👍", "👎"], key="rel")
        with col3:
            sat_feedback = st.radio("Satisfaction", ["👍", "👎"], key="sat")

        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")
            print(f"Feedback -> Accuracy: {acc_feedback}, Relevance: {rel_feedback}, Satisfaction: {sat_feedback}")

        st.markdown("---")
        if st.button("Try a New Style"):
            st.session_state.user_input = ""
            st.rerun()

    if st.button("Back to Landing Page"):
        st.session_state.show_chat = False
        st.session_state.user_input = ""
        st.rerun()
