import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache(allow_output_mutation=True)
def load_model():
    '''
    Loads the model and tokenizer from the local directory.
    :return: A list containing the model and the tokenizer.
    '''
    model_name = 'WIP'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return [model, tokenizer]

st.set_page_config(
    page_title="BulgakovLM Example",
    page_icon="👨‍💻",
)

st.markdown("# 👨‍💻 BulgakovLM Example")

txt = st.text_area('Write code here', '''Однажды утром''', height=400)

gen = st.button('Generate')

c = st.code('')

max_length = st.slider('max_length', 1, 1024, 128)
top_k = st.slider('top_k', 0, 100, 50)
top_p = st.slider('top_p', 0.0, 1.0, 0.9)
temperature = st.slider('temperature', 0.0, 1.0, 1.0)
num_beams = st.slider('num_beams', 1, 100, 5)
repetition_penalty = st.slider('repetition_penalty', 1.0, 10.0, 1.0)


if gen:
    c.code('Generating...')
    m = load_model()

    inpt = m[1].encode(txt, return_tensors="pt")
    out = m[0].generate(inpt, max_length=max_length, top_p=top_p, top_k=top_k, temperature=temperature, num_beams=num_beams, repetition_penalty=repetition_penalty)
    res = m[1].decode(out[0])

    print('ok')
    c.code(res)