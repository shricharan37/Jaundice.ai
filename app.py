# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from xhtml2pdf import pisa
from io import BytesIO

app = Flask(__name__)

# Load models
color_model = joblib.load('color_sensor_model.joblib')
temp_pulse_model = joblib.load('temp_pulse_model.joblib')

def rgb_to_ycrcb(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    return y, cr, cb

@app.route('/')
def index():
    return render_template('index_.html')  # updated filename

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('data_file')
    if not uploaded_file:
        return '⚠️ Please upload a data file.'

    filename = uploaded_file.filename
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        return '⚠️ Only .csv, .xlsx, or .xls files are supported.'

    df.columns = df.columns.str.strip().str.lower()
    required_cols = ['red', 'green', 'blue', 'temp', 'pulse']
    if not all(col in df.columns for col in required_cols):
        return "⚠️ File must include columns: {}".format(', '.join(required_cols))

    results_clean = []
    results_with_emoji = []
    for _, row in df.iterrows():
        r, g, b = row['red'], row['green'], row['blue']
        y, cr, cb = rgb_to_ycrcb(r, g, b)
        color_pred = color_model.predict([[y, cr, cb]])[0]
        temp = row['temp']
        pulse = row['pulse']
        temp_pulse_pred = temp_pulse_model.predict([[temp, pulse]])[0]

        if color_pred == 1 and temp_pulse_pred == 1 and max(r, g, b) >= 30:
            results_clean.append("Jaundiced")
            results_with_emoji.append("✅ Jaundiced")
        else:
            results_clean.append("Not Jaundiced")
            results_with_emoji.append("❌ Not Jaundiced")

    df['Jaundice'] = results_clean
    df.insert(0, "Patient Index", [f"Patient {i+1}" for i in range(len(df))])  # Add to PDF data

    # For HTML (with emojis)
    html_df = df.copy()
    html_df['Jaundice'] = results_with_emoji

    chart_data = {
        'color_data': df[['red', 'green', 'blue']].to_dict(orient='records'),
        'temp_pulse_data': df[['temp', 'pulse']].to_dict(orient='records')
    }

    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pdf_filename = f"report_{timestamp_str}.pdf"
    pdf_path = os.path.join("static", pdf_filename)
    generate_pdf_from_html(df, pdf_path)

    return render_template(
        'result-Copy.html',
        patient_table=html_df.to_html(classes='table table-bordered', index=False),
        result_summary="Prediction completed for all patients.",
        chart_data=json.dumps(chart_data),
        pdf_path=pdf_filename
    )


def generate_pdf_from_html(df, path):
    html_content = """
    <html>
    <head>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
                font-size: 10pt;
                text-align: center;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 6px;
                text-align: center;
            }}
            h1 {{
                text-align: center;
                font-family: Arial, sans-serif;
            }}
        </style>
    </head>
    <body>
        <h1>Jaundice Detection Report</h1>
        <p>Generated on: {timestamp}</p>
        {table}
    </body>
    </html>
    """.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        table=df.to_html(index=False, classes="center-table")
    )

    with open(path, "wb") as f:
        pisa.CreatePDF(src=html_content, dest=f)


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
