from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    # Get the blog title from the form
    blog_title = request.form['blog_title']
    
    # Tokenize the input text
    input_ids = tokenizer.encode(blog_title, return_tensors="pt", max_length=100, truncation=True)
    
    # Generate text using the GPT-2 model
    output = model.generate(input_ids, max_length=400, num_return_sequences=1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return render_template('display.html', blog_title=blog_title, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
