import contextlib
import wave
import os
import math
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo

labels = list(range(1, 11))
cwd = os.getcwd() + '\\dataset\\'
cwd_output = os.getcwd() + '\\output_dataset\\'
signals = dict()
for label in labels:
    directory_input = cwd + str(label)
    directory_output = cwd_output + str(label)

    for filename in os.listdir(directory_input):
        if filename.endswith(".wav"):
            print("Filename: " + filename)
            in_wav = os.path.join(directory_input, filename)
            out_wav = os.path.join(directory_output, filename)
            with contextlib.closing(wave.open(in_wav, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = int(((frames / float(rate))) * 1000)  # input file duration

                if (duration < 1000):  # if its less then 1s
                    newd = 1000 - duration
                    if (newd % 2 == 0):
                        pair = 0
                    else:
                        pair = 1

                    durbeg = (newd) / 2 + pair  # the time of silence in the begging

                    durend = (newd) / 2  # the time of silence in the end

                    # create silence audio segment of begging
                    beg_sec_segment = AudioSegment.silent(duration=durbeg)  # duration in milliseconds
                    # create of silence audio segment of end
                    end_sec_segment = AudioSegment.silent(duration=durend)  # duration in milliseconds

                    song = AudioSegment.from_wav(in_wav)
                    # Add above two audio segments
                    final_song = beg_sec_segment + song + end_sec_segment
                    try:
                        final_song.export(out_wav, format="wav")
                        print("Success: " + out_wav)

                    except IOError:
                        print("Failed: " + out_wav)

                else:
                    print("Duration is greater than 1: " + out_wav)

                    y, sr = librosa.load(in_wav)

                    # Trim the beginning and ending silence
                    yt, index = librosa.effects.trim(y)

                    # Duration of the signal in seconds
                    duration = librosa.get_duration(y=yt, sr=sr)
                    if duration < 1:
                        print("Duration after trim : " + str(duration))
                        excess_duration = (1 - duration) / 2
                        silence_size = round(excess_duration * sr)
                        # create silence audio segment of begging
                        beg_sec_segment = np.zeros(silence_size)  # duration in milliseconds
                        # create of silence audio segment of end
                        end_sec_segment = np.zeros(silence_size)  # duration in milliseconds

                        yt = np.concatenate((beg_sec_segment, yt, end_sec_segment))
                        sf.write(out_wav, yt, sr)
                    else:
                        print("Duration before: " + str(duration))
                        # Excessive duration at the start and the end of the signal that should be removed
                        excess_duration = (duration - 1) / 2
                        starting_index = math.floor(excess_duration * sr)
                        ending_index = starting_index + sr

                        # Signal wth duration of 1 sec
                        yt = yt[starting_index:ending_index]
                        print(yt.shape)
                        sf.write(out_wav, yt, sr)
