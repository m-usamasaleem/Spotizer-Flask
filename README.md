# Spotizer

Speaker diarization refers to the process of separating an audio file with various speakers into distinct audio file for each speaker. The partitioning of audio files based on distinct users have various applications across domains. Over the years, various methods have been proposed for speaker diarization. Deep Neural networks in conjunction with deep embeddings such as x-vector have been proven fruitful in such systems as they are able to distinctly identify each user. These systems use time delay neural network architecture, which aids in classifying patterns despite of shift invariance to identify different users. In this project, we have extended the use of the TDNN architecture to perform speaker diarization by implementing ECAPA-TDNN which is an improvement on the previous model. Our results successfully show that the ECAPA-TDNN outperforms the TDNN model.


## Installation

Create a python virtual enviroment

[Python3 Vitual Enviroments](https://docs.python.org/3/library/venv.html)

After your enviroment is created and activated, run:
```bash
  pip install -r requirements.txt
```
Once the requirements are installed, run:
```bash
  cd app/
  python3 app.py
```
