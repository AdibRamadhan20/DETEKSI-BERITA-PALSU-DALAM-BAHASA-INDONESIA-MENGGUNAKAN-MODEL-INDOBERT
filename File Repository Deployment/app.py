from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load tokenizer dan model IndoBERT hasil fine-tune
model_path = "model"  # ganti sesuai lokasi folder model kamu
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    judul = request.form['judul']
    konten = request.form['konten']
    text = judul + " " + konten

    # Tokenisasi teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = probs[0][pred_class].item()

    result = {
        "label": "BENAR (BUKAN HOAX)" if pred_class == 0 else "PALSU (HOAX)",
        "is_true": pred_class == 0,
        "confidence": f"{confidence:.2%}"
    }

    return render_template('index.html', result=result, judul=judul, konten=konten)

if __name__ == '__main__':
    app.run(debug=True)
