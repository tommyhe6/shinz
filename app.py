import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from flask import Flask, render_template, request

app = Flask(__name__)
app.debug = True

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

f = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["GET"])
def get():
    return summarize(f)
    

@app.route("/", methods=["POST"])
def post():
    f = request.form["filename"]
    return request.form["filename"] # request: {"file_name": name_of_file}


def summarize(text_file):
    with open(text_file, "r") as f:
        text = f.read()

    preprocess_text = text.strip().replace("\n"," ")
    t5_prepared_Text = "summarize: "+preprocess_text

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=100,
                                        early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    app.run(debug=True)

