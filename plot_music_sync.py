# coding: utf-8
"""
https://librosa.org/doc/main/auto_examples/plot_music_sync.html
===============================================
Music Synchronization with Dynamic Time Warping
===============================================

In this short tutorial, we demonstrate the use of dynamic time warping (DTW) for music synchronization
which is implemented in `librosa`.

We assume that you are # coding: utf-8

https://librosa.org/doc/main/auto_examples/plot_music_sync.html
===============================================
Music Synchronization with Dynamic Time Warping
===============================================

In this short tutorial, we demonstrate the use of dynamic time warping (DTW) for music synchronization
which is implemented in `librosa`.

We assume that you are familiar with the algorithm and focus on the application. Further information about
the algorithm can be found in the literature, e. g. [1]_.

Our example consists of two recordings of the first bars of the famous
brass section lick in Stevie Wonder's rendition of "Sir Duke".
Due to differences in tempo, the first recording lasts for ca. 7 seconds and the second recording for ca. 5 seconds.
Our objective is now to find an alignment between these two recordings by using DTW.

License:
Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

# Code source: Stefan Balke
# License: ISC
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa


############################################################
# ---------------------
# Load Audio Recordings
# ---------------------
# First, let's load a first version of our audio recordings.
#audio data slow is a varaible that holds an audio signal, its an array of audio samples from the music file.
#fs is the rate of the audio signal. how many samples per second were used to record
audio_data_slow, fs = librosa.load('audio/longer.mp3')
# And a second version, slightly faster.
audio_data_fast, fs = librosa.load('audio/tune.mp3')

# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# librosa.display.waveshow(x_1, sr=fs, ax=ax[0])
# ax[0].set(title='Slower Version $X_1$')
# ax[0].label_outer()

# librosa.display.waveshow(x_2, sr=fs, ax=ax[1])
# ax[1].set(title='Faster Version $X_2$')

#########################
# -----------------------
# Extract Chroma Features
# -----------------------
hop_length = 1024

#This holds the result of the chromagram comp. It is a matrix representing the chromograph over time
x_1_chroma = librosa.feature.chroma_cqt(y=audio_data_slow, sr=fs,
                                         hop_length=hop_length)
x_2_chroma = librosa.feature.chroma_cqt(y=audio_data_fast, sr=fs,
                                         hop_length=hop_length)

print("This is x 1", len(audio_data_slow))
print("This is x 2", len(audio_data_fast))
print("This is x 1chom", x_1_chroma)
print("This is x 2chom", x_2_chroma)
# fig, ax = plt.subplots(nrows=2, sharey=True)
# img = librosa.display.specshow(x_1_chroma, x_axis='time',
#                                y_axis='chroma',
#                                hop_length=hop_length, ax=ax[0])
# ax[0].set(title='Chroma Representation of $X_1$')
# librosa.display.specshow(x_2_chroma, x_axis='time',
#                          y_axis='chroma',
#                          hop_length=hop_length, ax=ax[1])
# ax[1].set(title='Chroma Representation of $X_2$')
# fig.colorbar(img, ax=ax)


########################
# ----------------------
# Align Chroma Sequences
# ----------------------

#X and Y are what is compared
#cosine is just used here cause we are comparing.
#D will hold the cost matrix between the two chormas
#wp will hold the optimal alignment from chorma 1 to chroma 2
D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')

#wp_s is a varialbe that will stop the wp as mentioned above in terms of time. It will hold the time values associated with the frames of the wp

wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

fig, ax = plt.subplots()


img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=fs,
                               cmap='gray_r', hop_length=hop_length, ax=ax)

#wp_s[:, 1] represents the second column of the wp_s array. This likely corresponds to the time values of the warping path.
#wp_s[:, 0] represents the first column of the wp_s array. This likely corresponds to the time values of the warping path.
#So this is actually whats plotting the stuff on the grid
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
ax.set(title='Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
fig.colorbar(img, ax=ax)

##############################################
# --------------------------------------------
# Alternative Visualization in the Time Domain
# --------------------------------------------
#
# We can also visualize the warping path directly on our time domain signals.
# Red lines connect corresponding time positions in the input signals.
# (Thanks to F. Zalkow for the nice visualization.)
from matplotlib.patches import ConnectionPatch

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8,4))

# Plot x_2
librosa.display.waveshow(audio_data_fast, sr=fs, ax=ax2)
ax2.set(title='Faster Version $X_2$')

# Plot x_1
librosa.display.waveshow(audio_data_slow, sr=fs, ax=ax1)
ax1.set(title='Slower Version $X_1$')
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
########################### coding: utf-8
"""
https://librosa.org/doc/main/auto_examples/plot_music_sync.html
===============================================
Music Synchronization with Dynamic Time Warping
===============================================

In this short tutorial, we demonstrate the use of dynamic time warping (DTW) for music synchronization
which is implemented in `librosa`.

We assume that you are familiar with the algorithm and focus on the application. Further information about
the algorithm can be found in the literature, e. g. [1]_.

Our example consists of two recordings of the first bars of the famous
brass section lick in Stevie Wonder's rendition of "Sir Duke".
Due to differences in tempo, the first recording lasts for ca. 7 seconds and the second recording for ca. 5 seconds.
Our objective is now to find an alignment between these two recordings by using DTW.

License:
Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

# Code source: Stefan Balke
# License: ISC
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa


############################################################
# ---------------------
# Load Audio Recordings
# ---------------------
# First, let's load a first version of our audio recordings.
#audio data slow is a varaible that holds an audio signal, its an array of audio samples from the music file.
#fs is the rate of the audio signal. how many samples per second were used to record
audio_data_slow, fs = librosa.load('audio/longer.mp3')
# And a second version, slightly faster.
audio_data_fast, fs = librosa.load('audio/tune.mp3')

# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# librosa.display.waveshow(x_1, sr=fs, ax=ax[0])
# ax[0].set(title='Slower Version $X_1$')
# ax[0].label_outer()

# librosa.display.waveshow(x_2, sr=fs, ax=ax[1])
# ax[1].set(title='Faster Version $X_2$')

#########################
# -----------------------
# Extract Chroma Features
# -----------------------
hop_length = 1024

#This holds the result of the chromagram comp. It is a matrix representing the chromograph over time
x_1_chroma = librosa.feature.chroma_cqt(y=audio_data_slow, sr=fs,
                                         hop_length=hop_length)
x_2_chroma = librosa.feature.chroma_cqt(y=audio_data_fast, sr=fs,
                                         hop_length=hop_length)

print("This is x 1", len(audio_data_slow))
print("This is x 2", len(audio_data_fast))
print("This is x 1chom", x_1_chroma)
print("This is x 2chom", x_2_chroma)
# fig, ax = plt.subplots(nrows=2, sharey=True)
# img = librosa.display.specshow(x_1_chroma, x_axis='time',
#                                y_axis='chroma',
#                                hop_length=hop_length, ax=ax[0])
# ax[0].set(title='Chroma Representation of $X_1$')
# librosa.display.specshow(x_2_chroma, x_axis='time',
#                          y_axis='chroma',
#                          hop_length=hop_length, ax=ax[1])
# ax[1].set(title='Chroma Representation of $X_2$')
# fig.colorbar(img, ax=ax)


########################
# ----------------------
# Align Chroma Sequences
# ----------------------

#X and Y are what is compared
#cosine is just used here cause we are comparing.
#D will hold the cost matrix between the two chormas
#wp will hold the optimal alignment from chorma 1 to chroma 2
D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')

#wp_s is a varialbe that will stop the wp as mentioned above in terms of time. It will hold the time values associated with the frames of the wp

wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

fig, ax = plt.subplots()


img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=fs,
                               cmap='gray_r', hop_length=hop_length, ax=ax)

#wp_s[:, 1] represents the second column of the wp_s array. This likely corresponds to the time values of the warping path.
#wp_s[:, 0] represents the first column of the wp_s array. This likely corresponds to the time values of the warping path.
#So this is actually whats plotting the stuff on the grid
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
ax.set(title='Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
fig.colorbar(img, ax=ax)

##############################################
# --------------------------------------------
# Alternative Visualization in the Time Domain
# --------------------------------------------
#
# We can also visualize the warping path directly on our time domain signals.
# Red lines connect corresponding time positions in the input signals.
# (Thanks to F. Zalkow for the nice visualization.)
from matplotlib.patches import ConnectionPatch

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8,4))

# Plot x_2
librosa.display.waveshow(audio_data_fast, sr=fs, ax=ax2)
ax2.set(title='Faster Version $X_2$')

# Plot x_1
librosa.display.waveshow(audio_data_slow, sr=fs, ax=ax1)
ax1.set(title='Slower Version $X_1$')
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
###########################################################
# -------------
# Next steps...
# -------------
#
# Alright, you might ask where to go from here.
# Once we have the warpin# coding: utf-8