import streamlit as st
from PIL import Image
import json
import os

from model import Asclepius

@st.cache_data
def get_model():
    model = Asclepius()
    return model

model = get_model()

st.set_page_config(page_title="Streamlit demo for asclepius", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Streamlit demo for asclepius")

with st.form("Sample"):
    sample_note = st.text_area('Sample Clinical Notes:', '''Discharge Summary: Patient: 69-year-old male Hospital Course: The patient was admitted to the ICU due to COVID-19 pneumonia, which was diagnosed after he experienced a dry cough for 2 weeks and showed poor oxygenation. Initially, he was given lung-protective ventilation and targeted sedation, and he remained stable. However, his condition worsened over the next days, and he developed hemodynamic instability and severe Acute Respiratory Distress Syndrome (ARDS). The patient underwent intermittent prone positioning and continuous renal replacement therapy because of these complications. Physical therapists were involved in this process, and they were responsible for ensuring the correct positioning of the patient's joints and preventing secondary complications, such as pressure ulcers, nerve lesions and contractures. After the tracheostomy, the patient underwent passive range-of-motion exercises and passive side-edge mobilization. However, asynchronous ventilation and hemodynamic instability persisted, inhibiting active participation. After spending 24 days in the ICU, the patient showed severe signs of muscle loss and scored 1/50 points on the Chelsea Critical Care Physical Assessment Tool (CPAx). The patient died soon after the withdrawal of life support. Hospital Stay: The patient was in the ICU for 24 days, where he received treatment interventions, including passive range of motion and positioning, intermittent prone positioning, and continuous renal replacement therapy. While in the ICU, the patient showed severe signs of muscle loss and developed pressure ulcer on the forehead. Discharge Instruction: The patient has been discharged in a stable condition. Although the patient was not discharged, we offer our sincere condolences to his family''')
    sample_question = st.text_area('Sample Question:', '''What roles did physical therapists have in the patient's treatment, and what secondary complications did they work to prevent? (Related to the task of relation extraction)''')     

note = st.text_area('Enter a Clinical Notes:')
question = st.chat_input('Enter your question:')

if question:
    result = model.inference(note,question)
    result_text = st.text_area("Result", result)