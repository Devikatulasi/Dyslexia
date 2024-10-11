from flask import Flask, render_template, request, redirect, url_for

import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))


@app.route('/')
def start():
    return render_template('home.html')


@app.route('/About')
def about():
    return render_template('about.html')


@app.route('/index3')
def quiz():
    return render_template('index3.html')

@app.route('/index2')
def quiz1():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        q1 = int(request.form.get('q1', 0))
        q2 = int(request.form.get('q2', 0))
        q3 = int(request.form.get('q3', 0))
        q4 = int(request.form.get('q4', 0))
        q5 = int(request.form.get('q5', 0))
        q6 = int(request.form.get('q6', 0))
        q7 = int(request.form.get('q7', 0))
        q8 = int(request.form.get('q8', 0))
        q9 = int(request.form.get('q9', 0))
        q10 = int(request.form.get('q10', 0))
        q11 = int(request.form.get('q11', 0))
        q12 = int(request.form.get('q12', 0))
        q13 = int(request.form.get('q13', 0))
        lang_vocb_score = (q1 + q2 + q3 + q4) / 16
        memory_score = (q2 + q6 + q7) / 12
        speed_score = 0.5
        visual_score = (q5 + q8 + q9 + q10 + q11) / 20
        audio_score = (q12 + q13) / 8
        all_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13
        total_score = all_score / 52

    scores = np.array([[lang_vocb_score, memory_score, speed_score, visual_score, audio_score, total_score]])

  
    label = int(model.predict(scores)[0])

    if label == 0:
        output = "There is a high chance of the applicant to have dyslexia."
    elif label == 1:
        output = "There is a  chance of the applicant to have dyslexia."
    else:
        output = "There is a no chance of the applicant to have dyslexia."

    return render_template('result1.html', lang_vocb_score=lang_vocb_score, memory_score=memory_score,
                           speed_score=speed_score, visual_score=visual_score, audio_score=audio_score,
                           total_score=total_score, result=output)

if __name__ == '__main__':
    app.run(debug=True)
