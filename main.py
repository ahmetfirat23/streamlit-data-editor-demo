import streamlit as st
import pandas as pd
from transformers import pipeline
import random

@st.cache_resource
def load_model():
    return pipeline(model="bert-base-uncased")

fill_masker = load_model()
st.title("Let me guess it for you")
st.text("Write your sentences, write 'GUESSME' on parts that you want me to guess and then press the button.")
col1, col2= st.columns([3,1])
with col2:
    st.video("https://www.youtube.com/watch?v=0DYtkumMEFU",start_time=60)

sentences = [["Let", "me", "GUESSME", "it", "for", "you."],
             ["Didn't", "I", "GUESSME", "it", "for", "you?", ],
             ["Why", "won't", "you", "GUESSME", "it", "for", "me?"],
             ["When", "all", "I", "GUESSME", "is", "for", "you,", "Kermie.", None, None, None]]
df = pd.DataFrame(sentences)
new_df = col1.experimental_data_editor(df, num_rows="dynamic")
btn = st.button(label="Press me")
if btn:
    new_df.fillna("", inplace=True)
    guesses = []
    for i, rows in new_df.iterrows():
        sentence = " ".join(rows.tolist()).replace("GUESSME", fill_masker.tokenizer.mask_token)
        result = fill_masker(sentence)
        idx = random.randint(0, len(result)-2)
        guess = " ".join(rows.tolist()).replace("GUESSME", (result[idx]["token_str"]))
        guesses.append(guess.split())
    st.text("My guess:")
    new_df = pd.DataFrame(guesses)
    st.dataframe(new_df)
