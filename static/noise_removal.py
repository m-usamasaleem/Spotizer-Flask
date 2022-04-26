
from noisereduce.noisereduce import SpectralGateStationary

import soundfile as sf
import matplotlib.pyplot as plt


def noise_removal(path, start, end):

    audio_clip_cafe, rate = sf.read(
        path)

    sg = SpectralGateStationary(
        y=audio_clip_cafe,
        sr=rate,
        y_noise=None,
        prop_decrease=1.0,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        n_std_thresh_stationary=1.5,
        tmp_folder=None,
        chunk_size=600000,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        clip_noise_stationary=True,
        use_tqdm=False,
        n_jobs=1,
    )

    subset_noise_reduce = sg.get_traces(start_frame=start, end_frame=end)
    sf.write("/media/msaleem2/data/Development/IS APPS/FLASK/static/audios/withoutnoise.wav",
             subset_noise_reduce, rate)
    plt.figure(figsize=(20, 5))
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'

    plt.plot(audio_clip_cafe)
    plt.plot(subset_noise_reduce)
    plt.yticks([])
    plt.xlabel('', fontsize=20)
    plt.xticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(
        "/media/msaleem2/data/Development/IS APPS/FLASK/static/img/noise_removal.jpg")
