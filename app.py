from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the AdaBoost Classifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'adaboost_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit & Loan Risk Assessment</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        body {
            background: radial-gradient(circle at top left, #0f172a, #1e1b4b, #020617);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 32px 16px;
        }
        .container {
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            width: 100%;
            max-width: 880px;
            padding: 40px 36px;
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        .header h1 {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p {
            color: #94a3b8;
            font-size: 1rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 16px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(1, 1fr);
            gap: 18px;
        }
        @media (min-width: 640px) {
            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (min-width: 860px) {
            .form-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input, select {
            width: 100%;
            padding: 11px 13px;
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid #334155;
            border-radius: 10px;
            color: #ffffff;
            font-size: 0.92rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }
        .btn-submit {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%);
            border: none;
            border-radius: 10px;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            margin-top: 24px;
            transition: transform 0.15s ease, opacity 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        }
        .btn-submit:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.55);
        }
        .btn-submit:active {
            transform: scale(0.98);
        }
        .result-box {
            margin-top: 32px;
            padding: 24px;
            border-radius: 14px;
            text-align: center;
            background: rgba(15, 23, 42, 0.9);
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #a5b4fc;
            margin-bottom: 6px;
        }
        .result-score {
            font-size: 2rem;
            font-weight: 800;
        }
        .status-positive {
            color: #f87171;
            border: 1px solid #dc2626;
        }
        .status-negative {
            color: #34d399;
            border: 1px solid #059669;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💳 Loan Risk Assessor</h1>
            <p>AdaBoost Ensemble Model for Loan Approval & Risk Assessment</p>
        </div>
        
        <form action="/predict" method="POST">
            <div class="section-title">Personal Profile</div>
            <div class="form-grid" style="margin-bottom: 24px;">
                <div class="form-group">
                    <label for="person_age">Age</label>
                    <input type="number" id="person_age" name="person_age" step="1" min="18" max="100" placeholder="e.g. 28" required>
                </div>
                <div class="form-group">
                    <label for="person_gender">Gender (Encoded)</label>
                    <input type="number" id="person_gender" name="person_gender" step="any" placeholder="e.g. 0 or 1" required>
                </div>
                <div class="form-group">
                    <label for="person_education">Education (Encoded)</label>
                    <input type="number" id="person_education" name="person_education" step="any" placeholder="e.g. 0, 1, or 2" required>
                </div>
                <div class="form-group">
                    <label for="person_income">Annual Income ($)</label>
                    <input type="number" id="person_income" name="person_income" step="any" placeholder="e.g. 55000" required>
                </div>
                <div class="form-group">
                    <label for="person_emp_exp">Employment Exp (Years)</label>
                    <input type="number" id="person_emp_exp" name="person_emp_exp" step="0.1" placeholder="e.g. 4.0" required>
                </div>
                <div class="form-group">
                    <label for="person_home_ownership">Home Ownership (Encoded)</label>
                    <input type="number" id="person_home_ownership" name="person_home_ownership" step="any" placeholder="e.g. 0, 1, or 2" required>
                </div>
            </div>

            <div class="section-title">Loan Details</div>
            <div class="form-grid" style="margin-bottom: 24px;">
                <div class="form-group">
                    <label for="loan_amnt">Loan Amount ($)</label>
                    <input type="number" id="loan_amnt" name="loan_amnt" step="any" placeholder="e.g. 10000" required>
                </div>
                <div class="form-group">
                    <label for="loan_intent">Loan Intent (Encoded)</label>
                    <input type="number" id="loan_intent" name="loan_intent" step="any" placeholder="e.g. 0, 1, 2" required>
                </div>
                <div class="form-group">
                    <label for="loan_int_rate">Interest Rate (%)</label>
                    <input type="number" id="loan_int_rate" name="loan_int_rate" step="0.01" placeholder="e.g. 11.25" required>
                </div>
                <div class="form-group">
                    <label for="loan_percent_income">Loan % of Income</label>
                    <input type="number" id="loan_percent_income" name="loan_percent_income" step="0.01" placeholder="e.g. 0.18" required>
                </div>
            </div>

            <div class="section-title">Credit History</div>
            <div class="form-grid">
                <div class="form-group">
                    <label for="cb_person_cred_hist_length">Credit History (Years)</label>
                    <input type="number" id="cb_person_cred_hist_length" name="cb_person_cred_hist_length" step="0.1" placeholder="e.g. 5.0" required>
                </div>
                <div class="form-group">
                    <label for="credit_score">Credit Score</label>
                    <input type="number" id="credit_score" name="credit_score" step="1" min="300" max="850" placeholder="e.g. 680" required>
                </div>
                <div class="form-group">
                    <label for="previous_loan_defaults_on_file">Previous Defaults</label>
                    <select id="previous_loan_defaults_on_file" name="previous_loan_defaults_on_file" required>
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </div>
            </div>
            
            <button type="submit" class="btn-submit">Assess Risk Level</button>
        </form>

        {% if prediction_text is not none %}
            <div class="result-box {% if prediction_text == 1 %}status-positive{% else %}status-negative{% endif %}">
                <div class="result-title">Model Classification Prediction</div>
                <div class="result-score">Class Outcome: {{ prediction_text }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Match feature order:
        # ['person_age', 'person_gender', 'person_education', 'person_income', 'person_emp_exp',
        #  'person_home_ownership', 'loan_amnt', 'loan_intent', 'loan_int_rate', 'loan_percent_income',
        #  'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']
        
        input_data = pd.DataFrame([{
            'person_age': float(request.form['person_age']),
            'person_gender': float(request.form['person_gender']),
            'person_education': float(request.form['person_education']),
            'person_income': float(request.form['person_income']),
            'person_emp_exp': float(request.form['person_emp_exp']),
            'person_home_ownership': float(request.form['person_home_ownership']),
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_intent': float(request.form['loan_intent']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length']),
            'credit_score': float(request.form['credit_score']),
            'previous_loan_defaults_on_file': float(request.form['previous_loan_defaults_on_file'])
        }])
        
        prediction = int(model.predict(input_data)[0])
        return render_template_string(HTML_TEMPLATE, prediction_text=prediction)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
