import numpy as np
import librosa
import matplotlib.pyplot as plt
import datasets

PIECE_TITLE = "Air for Band Bass Line"
PLAYING_SAMPLE_PATH = "/cs/home/stu/greer2jl/Documents/tele.band-playing-samples/"
PROCESSED_DATASET_PATH = "/cs/home/stu/greer2jl/Documents/teleband-export/teleband-export/teleband_dataset_mp3"
MP3_PATH = "/cs/home/stu/greer2jl/Documents/teleband-export/teleband-export/teleband_wavs"


#loading and getting everything set with the sample piece
playing_sample_data, playing_sample_sr = librosa.load(f'{PLAYING_SAMPLE_PATH}{PIECE_TITLE}.mp3')

dataset = datasets.load_from_disk(PROCESSED_DATASET_PATH)
bad_score = dataset.filter(lambda a : a["title"] == PIECE_TITLE and a["rhythm"] == 2)[0]
good_score = dataset.filter(lambda a : a["title"] == PIECE_TITLE and a["rhythm"] == 5)[0]
print(dataset)

#loading the actual mp3 file with libroas
bad_data, bad_sr = bad_score["key"]["array"], bad_score["key"]["sampling_rate"]
good_data, good_sr = good_score["key"]["array"], good_score["key"]["sampling_rate"]



hop_length = 1024

#This holds the result of the chromagram comp. It is a matrix representing the chromograph over time
playing_sample_chroma = librosa.feature.chroma_cqt(y=playing_sample_data, sr=playing_sample_sr,
                                         hop_length=hop_length)
bad_chroma = librosa.feature.chroma_cqt(y=bad_data, sr=bad_sr,
                                         hop_length=hop_length)
good_chroma = librosa.feature.chroma_cqt(y=good_data, sr=good_sr,
                                         hop_length=hop_length)


# --- Dynamic Time Warping --- 

dynamic_fig, (bad_ax, good_ax) = plt.subplots(nrows=2)

# -----

bad_D, bad_wp = librosa.sequence.dtw(X=playing_sample_chroma, Y=bad_chroma, metric='cosine')


bad_wp_s = librosa.frames_to_time(bad_wp, sr=playing_sample_sr, hop_length=hop_length)

bad_img = librosa.display.specshow(bad_D, x_axis='time', y_axis='time', sr=playing_sample_sr,
                               cmap='gray_r', hop_length=hop_length, ax=bad_ax)

bad_ax.plot(bad_wp_s[:, 1], bad_wp_s[:, 0], marker='o', color='r')
bad_ax.set(title='Bad Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
dynamic_fig.colorbar(bad_img, ax=bad_ax)

# -- -

good_D, good_wp = librosa.sequence.dtw(X=playing_sample_chroma, Y=good_chroma, metric='cosine')


good_wp_s = librosa.frames_to_time(good_wp, sr=playing_sample_sr, hop_length=hop_length)

good_img = librosa.display.specshow(good_D, x_axis='time', y_axis='time', sr=playing_sample_sr,
                               cmap='gray_r', hop_length=hop_length, ax=good_ax)

good_ax.plot(good_wp_s[:, 1], good_wp_s[:, 0], marker='o', color='r')
good_ax.set(title='Good Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
dynamic_fig.colorbar(good_img, ax=good_ax)


# -- Alternative Visualization -- 

# from matplotlib.patches import ConnectionPatch

# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8,4))

# # Plot x_2
# librosa.display.waveshow(playing_sample_data, sr=playing_sample_sr, ax=ax2)
# ax2.set(title='Sample Piece')

# # Plot x_1
# librosa.display.waveshow(bad_data, sr=bad_sr, ax=ax1)
# ax1.set(title='Student Performance')
# ax1.label_outer()


# n_arrows = 20
# for tp1, tp2 in bad_wp_s[::len(bad_wp_s)//n_arrows]:
#     # Create a connection patch between the aligned time points
#     # in each subplot
#     con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0),
#                           axesA=ax1, axesB=ax2,
#                           coordsA='data', coordsB='data',
#                           color='r', linestyle='--',
#                           alpha=0.5)
#     con.set_in_layout(False)  # This is needed to preserve layout
#     ax2.add_artist(con)

plt.show()