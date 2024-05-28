from flask import Flask,render_template,request,redirect,jsonify
import pandas as pd
import requests
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

def predict_survival(data):
    data.columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if model.predict(data)[0] ==0:
        return False
    else:
        return True


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    survived = predict_survival(df)
    return jsonify({'survived': survived})

if __name__ == '__main__':
    app.run(debug=True)





