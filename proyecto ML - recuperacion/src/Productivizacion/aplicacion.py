from flask import Flask, render_template, request
from flask import jsonify
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

path = r"C:\Users\Amy\OneDrive\Escritorio\Curso de programacion\The bridge\septiembre\proyecto ML - recuperacion\src\ML\air_model.pkl"
model = joblib.load(path)
scaler = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature = float(request.form['feature'])
    
    scaled_feature = scaler.transform([[feature]])
    
    prediction = model.predict(scaled_feature)[0]
    
    categories = ["Baja", "Moderada", "Alta", "Muy alta"]
    predicted_category = categories[prediction]
    
    return jsonify({'prediction': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)