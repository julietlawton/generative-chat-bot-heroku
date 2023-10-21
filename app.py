import re
import os
import threading
import torch
import boto3
import sys
from flask import Flask, render_template, request, jsonify 
from flask_bootstrap import Bootstrap
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from context import ContextWindow

app = Flask(__name__)
bootstrap = Bootstrap(app)

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = 497813341
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100 
            sys.stdout.write("\r%s  %s / %s  %s  (%.2f%%)" % (
                self._filename, self._seen_so_far, self._size, "bytes", percentage))
            sys.stdout.flush()

s3 = boto3.client(
    "s3", 
    aws_access_key_id=aws_access_key, 
    aws_secret_access_key=aws_secret_key
)

# Retrieve the list of existing buckets
response = s3.list_buckets()

# Output the bucket names
print("Downloading model...")

# Define the bucket and folder path
bucket_name = "cinebot-model"
model_folder_path = os.path.join(os.getcwd(), "tmp", bucket_name)

# Ensure local directory exists

if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

# List all files in the folder
objects = s3.list_objects_v2(Bucket=bucket_name)

# Download each file
for obj in objects.get('Contents', []):
    file_path_in_bucket = obj['Key']
    print(file_path_in_bucket)
    
    # Save the file
    local_file_path = os.path.join(model_folder_path, os.path.basename(file_path_in_bucket))
    
    s3.download_file(bucket_name, file_path_in_bucket, local_file_path, Callback=ProgressPercentage(file_path_in_bucket))

print("\nModel download complete.")

model = GPT2LMHeadModel.from_pretrained(model_folder_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_folder_path)
generator_max_length = 35
max_context_tokens = 100
contextwindow = ContextWindow(max_tokens=max_context_tokens)
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
        num_return_sequences=1,
        repetition_penalty=1.5,
        top_p=top_p,
        do_sample=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    #print(generated_text)
    normalized_text = ' '.join(generated_text.split())
    without_prompt = normalized_text.replace(context, "").strip()
    proper_punctuation = without_prompt.replace('í', "'")
    cleaned_response = remove_unfinished_sentences(re.sub(r'( <USER>).*$', '', proper_punctuation))

    if len(cleaned_response) < 2:
        if max_retries > 0:
            print("Response is empty. Trying again...")
            return generate(prompt, temperature, max_retries-1)
        else:
            print("Max retries reached. Returning default response.")
            return "Sorry, I couldn't generate a proper response. Please try again."

    output_ids = tokenizer.encode(cleaned_response, return_tensors="pt")
    output_length = output_ids.shape[1]
    contextwindow.add(cleaned_response, output_length)
    
    return cleaned_response 

  
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