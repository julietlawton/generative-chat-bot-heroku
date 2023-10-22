import re
import torch
import yake
import string
from yake import KeywordExtractor
from flask import Flask, render_template, request, jsonify 
from flask_bootstrap import Bootstrap
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from context import ContextWindow

app = Flask(__name__)
bootstrap = Bootstrap(app)

model = GPT2LMHeadModel.from_pretrained("model/")
tokenizer = GPT2Tokenizer.from_pretrained("model/")
generator_max_length = 35
max_context_tokens = 100
contextwindow = ContextWindow(max_tokens=max_context_tokens)
kw_extractor = KeywordExtractor(lan="en", n=3, top=10)
temperature = 0.5
top_p = 0.7

def structure_prompt(prompt):
    normalized = ' '.join(prompt.split())
    structured = f"<USER> {normalized} <AGENT>"
    return structured

def remove_unfinished_sentences(response):
    punctuation = set(".!?\"")
    if response[-1] in punctuation:
        return response
    else:
        last_punctuation = max(response.rfind(p) for p in punctuation)
        pruned = response[:last_punctuation+1]
    return pruned

def generate(prompt, temperature, max_retries=3): 
    input = structure_prompt(prompt)
    input_ids = tokenizer.encode(input, return_tensors="pt")
    input_length = input_ids.shape[1]

    contextwindow.add(input, input_length)
    current_tokens = contextwindow.get_current_token_count()
    context = contextwindow.get_conversation_history()
    print(f"Context: {context}")

    if current_tokens + generator_max_length > 1024:
        max_length = 1024
    else:
        max_length = current_tokens + generator_max_length

    # Tokenize conversation history
    context_ids = tokenizer.encode(context, return_tensors="pt")
    context_length = context_ids.shape[1]

    attention_mask = torch.ones(context_ids.shape)

    output = model.generate(
        context_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        max_length=max_length,
        num_return_sequences=4,
        repetition_penalty=1.5,
        top_p=top_p,
        do_sample=True
    )

    keywords = kw_extractor.extract_keywords(context)
    keyword_list = [item[0].lower() for item in keywords]
    response_scores = []

    for i, sequence in enumerate(output):
        generated_text = tokenizer.decode(sequence, skip_special_tokens=False)

        normalized_text = ' '.join(generated_text.split())
        without_prompt = normalized_text.replace(context, "").strip()
        proper_punctuation = without_prompt.replace('í', "'")
        cleaned_response = remove_unfinished_sentences(re.sub(r'( <USER>).*$', '', proper_punctuation))
        no_punc = cleaned_response.translate(str.maketrans('', '', string.punctuation))

        score = sum([1 for word in no_punc.lower().split() if word in keyword_list])
        response_scores.append((score, cleaned_response))
        response_scores.sort(key=lambda s: s[0], reverse=True)
    
    chosen_response = response_scores[0][1]

    if len(chosen_response) < 2:
        if max_retries > 0:
            print("Response is empty. Trying again...")
            return generate(prompt, temperature, max_retries-1)
        else:
            print("Max retries reached. Returning default response.")
            return "Sorry, I couldn't generate a proper response. Please try again."

    output_ids = tokenizer.encode(chosen_response, return_tensors="pt")
    output_length = output_ids.shape[1]
    contextwindow.add(chosen_response, output_length)
    
    return chosen_response 

  
@app.route("/", methods=["POST", "GET"]) 
def index(): 
    if request.method == "POST": 
        prompt = request.form["prompt"] 
        temperature_selection = request.form["temperature"]
        temperature = float(temperature_selection)
        response = generate(prompt, temperature) 
  
        return jsonify({"response": response}) 
    return render_template("index.html") 
  
if __name__ == "__main__": 
    app.run(debug=True) 