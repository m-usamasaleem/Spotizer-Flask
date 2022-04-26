# pip install spectralcluster SpeechRecognition pydub
from spectralcluster import SpectralClusterer
import speech_recognition as sr
import pydub

import torchaudio
import torch

import numpy as np
import json
import os

def analyze_audio(input_path, output_folder, path_to_model, num_speakers, verbose=False):
  '''
  Function to analyze audio and return diarization and speech content.

  **Arguments**
  input_path : str
    A path to the input audio file (something.wav)
  output_folder : str
    The folder that the audio should be saved in. This should be unique for each new recording. The folder will be created
    if it does not exist
  path_to_model : str
    A path to the 'ecapa_model.ptl' file that contains the ECAPA model weights
  num_speakers : int
    The number of speakers in the recording
  verbose: bool
    Will print what the function is doing

  **Returns**
  return_dict : dict
    A dictionary containing the diarization and speech content

  -------
  Example
  -------
  returned_dict = analyze_audio('/content/full_audio.wav', '/content/', '/content/ecapa_model.ptl', num_speakers=4, verbose=True)
  '''
  assert os.path.exists(input_path)
  assert os.path.exists(path_to_model)
  os.makedirs(output_folder, exist_ok=True) # attempt to create output_folder if it does not exist

  model = torch.jit.load(path_to_model)

  signal, sr = torchaudio.load(input_path)
  
  '''
  Compute segment embeddings
  '''
  # Compute embeddings
  if verbose: print('Computing speaker embeddings...')
  WINDOW_SIZE = 3 # in seconds
  HOP_LENGTH = WINDOW_SIZE # in seconds
  SAMPLING_RATE = 16000

  if sr != SAMPLING_RATE:
    print(f'Resampling audio from {sr}Hz to {SAMPLING_RATE}Hz...')
    resampler = torchaudio.transforms.Resample(sr, 16000)
    resampler(signal)
    print(f'Resampling finished.')

  num_segments = signal.shape[1]//int(HOP_LENGTH*SAMPLING_RATE)-1
  segment_embeddings = np.empty((num_segments, 192))

  for i in range(num_segments):
    start_idx = int(i*SAMPLING_RATE*HOP_LENGTH)
    end_idx = start_idx+(SAMPLING_RATE*WINDOW_SIZE)

    segment_signal = signal[:,start_idx:end_idx]
    spectr = gen_ECAPA_input(segment_signal)

    segment_embeddings[i] = model(spectr).detach().numpy()[0][0]


  '''
  Compute affinity matrix
  '''
  dotted = segment_embeddings @ segment_embeddings.T
  normalized = np.linalg.norm(segment_embeddings, axis=1)**2

  affinity_matrix = dotted/normalized # cosine similarities between each segment

  
  '''
  Clustering
  '''
  clusterer = SpectralClusterer(
      min_clusters=num_speakers,
      max_clusters=num_speakers,
      laplacian_type=None,
      refinement_options=None,
      custom_dist="cosine")

  labels = clusterer.predict(segment_embeddings)
  labels = np.abs(labels-labels.max())


  '''
  Segment audio and generate return JSON
  '''
  if verbose: print('Segmenting Audio...')
  return_dict = {}
  n_segment = 0
  find_consecutive = lambda x: np.split(x, np.where(np.diff(x) != 1)[0]+1)

  for i in range(num_speakers):
    idxs = np.where(labels==i)[0]
    segments = find_consecutive(idxs)

    for j, segment in enumerate(segments):
      if segment.shape[0] < 3:
        continue

      seg_start = segment.min()
      seg_start_time = WINDOW_SIZE * seg_start
      seg_end = segment.max()
      seg_end_time = WINDOW_SIZE * seg_end

      start_idx_signal = seg_start*SAMPLING_RATE*WINDOW_SIZE
      end_idx_signal = seg_end*SAMPLING_RATE*WINDOW_SIZE

      seg_audio = signal[:,start_idx_signal:end_idx_signal]
      audio_out_path = f'{output_folder}/seg{n_segment}_speaker{i}_speakerseg{j}.wav'
      torchaudio.save(audio_out_path, seg_audio, SAMPLING_RATE)

      # everything must be a string in order to get converted to JSON
      return_dict[f'segment_{n_segment}'] = {
        'global_segment': str(n_segment),
        'speaker_segment': str(j),
        'speaker_id': str(i),
        'start_time': str(seg_start_time),
        'end_time': str(seg_end_time),
        'audio_file': str(audio_out_path),
        'text': -1 # will be filled in by gen_spoken_contents
      }

      n_segment += 1


  '''
  Speech recognition
  '''
  if verbose: print('Generating transcripts...')
  return_dict = gen_spoken_contents(return_dict)

  return json.dumps(return_dict)

def gen_ECAPA_input(signal):
  # STFT hyperparameters
  hop_length_ms = 10 # hop length in ms
  win_length_ms = 32 # win length in ms

  hop_length_samples = round((16000 / 1000.0) * hop_length_ms) # hop length in samples
  win_length_samples = round((16000 / 1000.0) * win_length_ms) # win length in samples

  window_fn = torch.hamming_window(win_length_samples) # hamming window function

  # Perform STFT
  torch_stft = torch.stft(signal, 512, hop_length_samples, win_length_samples, window_fn, pad_mode='constant', center=True)
  torch_stft = torch_stft.transpose(2, 1)
  torch_spectr = (torch_stft**2).sum(-1) # this is the input to the model, shape is (1, _, 257) the "_" depends on signal length

  return torch_spectr

def gen_spoken_contents(return_dict):
  recognizer = sr.Recognizer()
  
  for k in return_dict.keys():

    v = return_dict[k]
    
    # convert to PCM WAV
    sound = pydub.AudioSegment.from_wav(v['audio_file'])
    sound.export(v['audio_file'], format="wav")

    # load audio
    audio = sr.AudioFile(v['audio_file'])
    with audio as source:
      audio_obj = recognizer.record(source)

    # do speech recognition
    predicted_text = recognizer.recognize_google(audio_obj)

    v['text'] = predicted_text

  return return_dict