from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Charger le modèle et le vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['email']
    transformed_data = vectorizer.transform([data])
    prediction = model.predict(transformed_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
