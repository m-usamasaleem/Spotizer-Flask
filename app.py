from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap4
from static.TextToSpeechModel import tts_bp

# from flask_bootstrap import Bootstrap
from flask_gtts import gtts
import io
import urllib.request
from scipy.io import wavfile
from werkzeug.utils import secure_filename
import os
from static import diarization_and_recognition as dr
from static.noise_removal import noise_removal
from static. plotting_waveform import plotting_waveform
from flask import render_template, flash, request, redirect, url_for, session, send_from_directory
from flask import Flask, Response
import speech_recognition as sr
from flask import Flask, render_template, request, redirect
from flask import Flask, render_template
from flask import send_file
from scipy.io.wavfile import write
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from collections import OrderedDict
from operator import getitem
import random
import matplotlib.pyplot as plt
import wave
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
cv = CountVectorizer()
le = LabelEncoder()
app.config["CACHE_TYPE"] = "null"
cwd = os.getcwd()
print(cwd)


@app.route('/language')
def language():
    return render_template('language.html')


@app.route("/noiseremoval", methods=["GET", "POST"])
def noiseremoval():
    try:
        path_save = ""
        if request.method == "POST":
            print("FORM DATA RECEIVED")
            start = np.int64(request.form['start'])
            end = np.int64(request.form['end'])
            if "file" not in request.files:
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)
            if file:

                file.save(os.path.join(
                    os.path.join(cwd, "static/audios/"), secure_filename(file.filename)))
                path_save = os.path.join(
                    os.path.join(cwd, "static/audios/"), file.filename)
                noise_removal(path_save, start, end)
    except:
        print("Error in language")
    return render_template('noiseremoval.html', transcript={"transcript": path_save})


@app.route('/language_predict', methods=['POST'])
def language_predict():
    try:
        if request.method == "POST":
            # Read data
            df = pd.read_csv("Language Detection.csv")
            # feature and label extraction
            X = df["Text"]
            y = df["Language"]

            # Label encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

            # cleaning the data
            text_list = []

            # iterating through all the text
            for text in X:
                # removes all the symbols and numbers
                text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
                text = re.sub(r'[[]]', ' ', text)
                text = text.lower()          # converts all the text to lower case
                # appends the text to the text_list
                text_list.append(text)

            # Encode the feature(text)

            cv = CountVectorizer()
            X = cv.fit_transform(text_list).toarray()

            model = pickle.load(open("langdetectmodel.pkl", "rb"))
            print("predicting")
            if request.method == 'POST':
                txt = request.form['text']
                # convert text to bag of words model (Vector)
                t_o_b = cv.transform([txt]).toarray()
                language = model.predict(t_o_b)  # predict the language
                # find the language corresponding with the predicted value
                corr_language = le.inverse_transform(language)

                output = corr_language[0]

            return render_template('language.html', prediction='Language is in {}'.format(output))
    except:
        print("Error in language")
    return render_template('language.html')


@app.route('/')
def splash():
    return render_template('splash.html')


@app.route('/presignup')
def pre_signup():
    return render_template('presignup.html')


@app.route("/home", methods=["GET", "POST"])
def home_page():
    transcription = ""
    path = "audios"
    path_save = ""
    try:
        if request.method == "POST":
            print("FORM DATA RECEIVED")
            print("request.form['nospeakers']", request.form['nospeakers'])
            if "file" not in request.files:
                return redirect(request.url)

            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)

            if file:
                file.save(os.path.join(
                    os.path.join(cwd, "static/audios/"), secure_filename(file.filename)))
                path_save = os.path.join(
                    os.path.join(cwd, "static/audios/"), file.filename)

                plotting_waveform(path_save)
                print(path_save)
                transcription = dr.analyze_audio(
                    path_save, output_folder=os.path.join(cwd, "static/audios/out/"), num_speakers=np.int64(request.form['nospeakers']))

                transcription = json.loads(transcription)
                transcription = dict(OrderedDict(sorted(transcription.items(),
                                                        key=lambda x: getitem(x[1], 'start_time'))))
                for key in transcription.keys():
                    transcription[key]['audio_file'] = transcription[key]['audio_file'].split(
                        "/")[-1]
    except:
        print("Error in home")

    return render_template('speakers.html', transcript={"transcript": transcription, "path": path_save})


@ app.route('/<audio_file_name>')
def returnAudioFile(audio_file_name):

    try:
        path_to_audio_file = os.path.join(
            cwd, "static/audios/out/") + audio_file_name
        print(path_to_audio_file)
        print()
        print()
        return send_file(
            path_to_audio_file,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename="test.wav")
    except:
        print("Error in returnAudioFile")


gtts(app)
bootsrap = Bootstrap4(app)

# Registering the Text to Speech blueprint
app.register_blueprint(tts_bp)


@app.route("/tts")
def tts():
    return render_template('tts/texttospeech.html')
