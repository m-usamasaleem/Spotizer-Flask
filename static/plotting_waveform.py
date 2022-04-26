

import wave
import matplotlib.pyplot as plt
import numpy as np

import random


def plotting_waveform(path_save):
    try:
        spf = wave.open(path_save, "r")
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, "int16")
        fs = spf.getframerate()

        times = np.linspace(0, len(signal) / fs, num=len(signal))
        print(signal.shape, times.shape)
        plt.figure(figsize=(15, 5))
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['savefig.facecolor'] = 'black'

        color_list = ['gray', 'red', 'yellow', 'purple', "green", "olive", "aquamarine",
                      "mediumseagreen", "xkcd:sky blue", "xkcd:eggshell", "cyan", "blue"]
        for i in range(0, len(times), 100000):
            r = random.randint(0, 1)
            g = random.randint(0, 1)
            b = random.randint(0, 1)
            rgb = [r, g, b]

            plt.plot(times[i:i*100000], signal[i:i*100000],
                     color=np.random.choice(color_list, 1)[0])

        # plt.axvline(x=0.5, color='w', linestyle='--', linewidth=2,)
        # plt.axvline(x=10, color='w', linestyle='--', linewidth=2,)
        # plt.axvline(x=15, color='w', linestyle='--', linewidth=2,)

        plt.yticks([])
        plt.xlabel('Time (s)', fontsize=20)
        plt.xticks(fontsize=20)

        plt.tight_layout()
        plt.savefig(
            "/media/msaleem2/data/Development/IS APPS/FLASK/static/img/waveform.jpg")
    except:
        print("Error in plotting waveform")
