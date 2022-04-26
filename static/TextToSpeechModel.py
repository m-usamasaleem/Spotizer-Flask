from flask import Blueprint, json, request, render_template

# Initializing the Text to Speech blueprint
tts_bp = Blueprint('texttospeech', __name__, url_prefix='/texttospeech')


@tts_bp.route('/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return render_template('tts/texttospeech.html')
    if request.method == 'POST':
        user_input = request.form
        return render_template('tts/texttospeech_result.html', user_input=user_input)
