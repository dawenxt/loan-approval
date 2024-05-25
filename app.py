import flask
import pickle
import numpy as np
import csv
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load models at the top of the app to load into memory only one time
with open('models/loan_application_model_lr.pickle', 'rb') as f:
    clf_lr = pickle.load(f)

ss = StandardScaler()

genders_to_int = {'MALE': 1, 'FEMALE': 0}
married_to_int = {'YES': 1, 'NO': 0}
education_to_int = {'GRADUATED': 1, 'NOT GRADUATED': 0}
dependents_to_int = {'>=2': 0, '<2': 1}
self_employment_to_int = {'YES': 1, 'NO': 0}
property_area_to_int = {'RURAL': 0, 'URBAN': 1}

app = flask.Flask(__name__, template_folder='templates')

CSV_FILE_PATH = 'loan_applications.csv'

def generate_loan_id():
    now = datetime.now()
    return f"LOAN{now.strftime('%Y%m%d%H%M%S')}"

def save_to_csv(data, file_path=CSV_FILE_PATH):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        fieldnames = list(data.keys()) 
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/loan')
def loan():
    return flask.render_template('loan.html')

@app.route('/admin1')
def admin():
    return flask.render_template('admin.html')


@app.route("/Loan_Application", methods=['GET', 'POST'])
def Loan_Application():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    
    if flask.request.method == 'POST':
        # Get input from form
        genders_type = flask.request.form['genders_type']
        marital_status = flask.request.form['marital_status']
        dependents = flask.request.form['dependents']
        education_status = flask.request.form['education_status']
        self_employment = flask.request.form['self_employment']
        applicantIncome = float(flask.request.form['applicantIncome'])
        coapplicantIncome = float(flask.request.form['coapplicantIncome'])
        loan_amnt = float(flask.request.form['loan_amnt'])
        term_d = int(flask.request.form['term_d'])
        credit_history = int(flask.request.form['credit_history'])
        property_area = flask.request.form['property_area']

        # Generate loan ID
        loan_id = generate_loan_id()

        # Create original output dict
        output_dict = {
            'Loan ID': loan_id,
            'Applicant Income': applicantIncome,
            'Co-Applicant Income': coapplicantIncome,
            'Loan Amount': loan_amnt,
            'Loan Amount Term': term_d,
            'Credit History': credit_history,
            'Gender': genders_type,
            'Marital Status': marital_status,
            'Education Level': education_status,
            'No of Dependents': dependents,
            'Self Employment': self_employment,
            'Property Area': property_area,
        }

        # Prepare input for prediction
        x = np.zeros(21)
        x[0] = applicantIncome
        x[1] = coapplicantIncome
        x[2] = loan_amnt
        x[3] = term_d
        x[4] = credit_history

        # Make prediction
        pred = clf_lr.predict([x])[0]
        result = 'Approved' if pred == 1 else 'Denied'

        # Add Loan Status and Loan ID to output_dict
        output_dict['Loan Status'] = result

        # Save the user input to CSV file
        save_to_csv(output_dict)

        # Render different templates based on the prediction result
        if pred == 1:
            return flask.render_template('approved.html', result=result, original_input=output_dict)
        else:
            return flask.render_template('denied.html', result=result, original_input=output_dict)

if __name__ == '__main__':
    app.run(debug=True)
