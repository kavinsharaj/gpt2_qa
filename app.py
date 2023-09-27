import streamlit as st
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
import re



def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return tokenizer

def generate_text(model_path, tokenizer,sequence, max_length):

    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    output=(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
    return output



model_path = "kavin23/qa_gpt2"
model = load_model(model_path)
tokenizer = load_tokenizer(model_path)
user_input = st.text_area('Enquire about the Menu')
button = st.button("Enter")
if user_input and button :
    sequence2 = user_input
    max_len = 50
    answer=generate_text(model, tokenizer, sequence2, max_len)
    answers = re.findall(r'\[A\] (.*?)\n', answer)
    result = '\n'.join(answers)
    st.write("Answer: ",result)