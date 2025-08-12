import streamlit as st
import json
import os
from rapidfuzz import process
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Dynamically resolve path
file_path = os.path.join(os.path.dirname(__file__), "..", "data", "schemes.json")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
except Exception as e:
    st.error(f"Failed to load FAQ data: {e}")
    faq_data = []

st.title("Beej : किसान सलाहकार")

st.markdown("""
    **पूछें:** कोई भी सवाल जैसे "पीएम-किसान योजना क्या है?"  
    **उत्तर:** हम आपको योजनाओं की जानकारी देंगे!
""")

query = st.text_input("Ask a question about PM-KISAN or Soil Health Card")

import speech_recognition as sr

def transcribe_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(" Speak now...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language="hi-IN")  # Hindi support
        st.success(f" You said: {query}")
        return query
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand your voice.")
    except sr.RequestError:
        st.error("Could not connect to the speech recognition service.")
    return ""



# Using Indic BERT for more complex queries
def get_llm_answer(query, faq_data):
    questions = [item["question"] for item in faq_data]
    match, score, index = process.extractOne(query, questions)
    if score > 60:
        context = faq_data[index].get("context", "")
        if context:
            result = qa_pipeline(question=query, context=context)
            return result["answer"]
        else:
            return faq_data[index]["answer"]
    return "माफ़ कीजिए, मुझे आपके सवाल का उत्तर नहीं मिला।"

if query:
    answer = get_llm_answer(query, faq_data)
    st.success(f"🗨️ {answer}")