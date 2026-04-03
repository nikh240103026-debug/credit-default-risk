from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('credit_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Credit Default Risk Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 650px; margin: 50px auto; background-color: #f4f4f4; }
        h1 { text-align: center; color: #333; }
        form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        label { font-weight: bold; display: block; margin-top: 15px; }
        input { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 5px; }
        button { width: 100%; padding: 12px; margin-top: 20px; background-color: #333; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        .result { text-align: center; font-size: 20px; margin-top: 20px; padding: 15px; background: white; border-radius: 10px; font-weight: bold; }
        .safe { color: green; }
        .risk { color: red; }
    </style>
</head>
<body>
    <h1>🏦 Credit Default Risk Predictor</h1>
    {% if prediction %}
    <div class="result {{ css_class }}">{{ prediction }}</div>
    {% endif %}
    <form action="/predict" method="post">
        <label>Age:</label>
        <input type="number" name="age" placeholder="e.g. 45" required>
        <label>Monthly Income ($):</label>
        <input type="number" name="income" placeholder="e.g. 5000" required>
        <label>Debt Ratio (0 to 1):</label>
        <input type="number" step="0.01" name="debtratio" placeholder="e.g. 0.35" required>
        <label>Revolving Utilization (0 to 1):</label>
        <input type="number" step="0.01" name="utilization" placeholder="e.g. 0.5" required>
        <label>Number of Dependents:</label>
        <input type="number" name="dependents" placeholder="e.g. 2" required>
        <label>Open Credit Lines:</label>
        <input type="number" name="creditlines" placeholder="e.g. 5" required>
        <label>Real Estate Loans:</label>
        <input type="number" name="realestateloans" placeholder="e.g. 1" required>
        <label>Times 30-59 Days Late:</label>
        <input type="number" name="late3059" placeholder="e.g. 0" required>
        <label>Times 60-89 Days Late:</label>
        <input type="number" name="late6089" placeholder="e.g. 0" required>
        <label>Times 90+ Days Late:</label>
        <input type="number" name="late90" placeholder="e.g. 0" required>
        <button type="submit">Predict Risk</button>
    </form>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    features = pd.DataFrame([[
        float(data['utilization']),
        int(data['age']),
        int(data['late3059']),
        float(data['debtratio']),
        float(data['income']),
        int(data['creditlines']),
        int(data['late90']),
        int(data['realestateloans']),
        int(data['late6089']),
        int(data['dependents'])
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ])

    prediction = model.predict(features)[0]

    if prediction == 0:
        result = "✅ Low Risk — Likely to repay loan"
        css_class = "safe"
    else:
        result = "⚠️ High Risk — Likely to default"
        css_class = "risk"

    return render_template_string(HTML, prediction=result, css_class=css_class)

if __name__ == '__main__':
    app.run(debug=True)