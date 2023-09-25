import numpy as np
import librosa
import matplotlib.pyplot as plt
import datasets

PIECE_TITLE = "Air for Band Bass Line"
PLAYING_SAMPLE_PATH = "/cs/home/stu/greer2jl/Documents/tele.band-playing-samples/"
PROCESSED_DATASET_PATH = "processed_dataset/teleband_dataset_mp3"
MP3_PATH = "/cs/home/stu/greer2jl/Documents/teleband-export/teleband-export/teleband_wavs"


#loading and getting everything set with the sample piece
p_data, p_sr = librosa.load(f'{PLAYING_SAMPLE_PATH}{PIECE_TITLE}.mp3')

dataset = datasets.load_from_disk(PROCESSED_DATASET_PATH)
dataset = dataset.filter(lambda a : a["title"] == PIECE_TITLE and a["rhythm"] == 3)



#loading the actual mp3 file with libroas
s_data, s_sr = dataset[1]["key"]["array"], dataset[1]["key"]["sampling_rate"]



hop_length = 1024

#This holds the result of the chromagram comp. It is a matrix representing the chromograph over time
p_chroma = librosa.feature.chroma_cqt(y=p_data, sr=p_sr,
                                         hop_length=hop_length)
s_chroma = librosa.feature.chroma_cqt(y=s_data, sr=p_sr,
                                         hop_length=hop_length)


# --- Dynamic Time Warping --- 

D, wp = librosa.sequence.dtw(X=p_chroma, Y=s_chroma, metric='cosine')


wp_s = librosa.frames_to_time(wp, sr=p_sr, hop_length=hop_length)

fig, ax = plt.subplots()


img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=p_sr,
                               cmap='gray_r', hop_length=hop_length, ax=ax)

ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
ax.set(title='Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
fig.colorbar(img, ax=ax)


# -- Alternative Visualization -- 

from matplotlib.patches import ConnectionPatch

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8,4))

# Plot x_2
librosa.display.waveshow(p_data, sr=p_sr, ax=ax2)
ax2.set(title='Sample Piece')

# Plot x_1
librosa.display.waveshow(s_data, sr=s_sr, ax=ax1)
ax1.set(title='Student Performance')
ax1.label_outer()


n_arrows = 20
for tp1, tp2 in wp_s[::len(wp_s)//n_arrows]:
    # Create a connection patch between the aligned time points
    # in each subplot
    con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0),
                          axesA=ax1, axesB=ax2,
                          coordsA='data', coordsB='data',
                          color='r', linestyle='--',
                          alpha=0.5)
    con.set_in_layout(False)  # This is needed to preserve layout
    ax2.add_artist(con)

plt.show()