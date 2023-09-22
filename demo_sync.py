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
print(dataset[0])

dataset = dataset.filter(lambda a : a["title"] == PIECE_TITLE and a["rhythm"] == 5)


print(dataset[0])
#loading the actual mp3 file with libroas
s_data, s_sr = dataset[1]["key"]["array"], dataset[1]["key"]["sampling_rate"]



hop_length = 1024

#This holds the result of the chromagram comp. It is a matrix representing the chromograph over time
p_chroma = librosa.feature.chroma_cqt(y=p_data, sr=p_sr,
                                         hop_length=hop_length)
s_chroma = librosa.feature.chroma_cqt(y=s_data, sr=p_sr,
                                         hop_length=hop_length)


D, wp = librosa.sequence.dtw(X=p_chroma, Y=s_chroma, metric='cosine')


wp_s = librosa.frames_to_time(wp, sr=p_sr, hop_length=hop_length)

fig, ax = plt.subplots()


img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=p_sr,
                               cmap='gray_r', hop_length=hop_length, ax=ax)

#wp_s[:, 1] represents the second column of the wp_s array. This likely corresponds to the time values of the warping path.
#wp_s[:, 0] represents the first column of the wp_s array. This likely corresponds to the time values of the warping path.
#So this is actually whats plotting the stuff on the grid
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
ax.set(title='Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
fig.colorbar(img, ax=ax)



# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# librosa.display.waveshow(p_data, sr=p_sr, ax=ax[0])
# ax[0].set(title='Slower Version $X_1$')
# ax[0].label_outer()

# # librosa.display.waveshow(data, sr=sr, ax=ax[1])
# # ax[1].set(title='Faster Version $X_2$')
# # plt.show()
# print(data[0])
plt.show()