from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer from saved directory
model = AutoModelForSequenceClassification.from_pretrained("Ali508208/emotion-model")
tokenizer = AutoTokenizer.from_pretrained("Ali508208/emotion-model")

# Create pipeline
emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Default route
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Emotion Detection API!",
        "usage": "Send a POST request to /predict with JSON: { 'text': 'your input text here' }"
    })


@app.route('/predict', methods=['POST'])
def predict_emotion():
    data = request.get_json()

    # Validate input
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    text = data['text']
    predictions = emotion_pipeline(text)
    sorted_predictions = sorted(predictions[0], key=lambda x: x['score'], reverse=True)
    top_prediction = sorted_predictions[0]

    return jsonify({
        'text': text,
        'emotion': top_prediction['label'],
        'confidence': round(top_prediction['score'], 3)
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
