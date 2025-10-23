"""
Estimate race and gender for readings in course syllabi
Flask code for web deployment
"""

import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, url_for
import pandas as pd
from werkzeug.utils import secure_filename

import analyze_readings

app = Flask(__name__)

CSVTEMPLATES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv_templates')
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

app.config['CSVTEMPLATES_FOLDER'] = CSVTEMPLATES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB limit

@app.route('/')
@app.route('/home')
def home(error=None):
    return render_template('home.html', upload_error=request.args.get('upload_error'))

@app.route('/csv_templates/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory=app.config['CSVTEMPLATES_FOLDER'], filename=filename)

# from https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':       
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file: 
            file.seek(0)
            try:
                trows = process_uploaded_file(file)
                return render_template('analysis.html', trows=trows)
            except:
                return redirect(url_for('home', upload_error=True))

def process_uploaded_file(file):
    """Takes a werkzeug secure file object as the argument
    Runs the syllabus analyzer code on this file
    Returns a list of dicts with the table values"""
    sa = analyze_readings.syllabusAnalyzer(secure_filename(file.filename), None)
    sa.loadData(file)
    sa.addRaceGender()
    sa.outDf.rename(columns={'pc_female':'female'}, inplace=True)
    cols = ['female','asian','hispanic','nh_black','nh_white']
    summary = (sa.outDf.groupby('courseid')[cols].mean()*100).astype(int)
    summary['N_authors'] = sa.outDf.groupby('courseid').N_authors.sum().astype(int)
    summary.reset_index(inplace=True)

    return summary[['courseid','N_authors','female','asian','hispanic','nh_black','nh_white']].values.tolist()

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)