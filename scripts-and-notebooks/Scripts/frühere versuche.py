import librosa
import numpy as np
import math
from scipy.fft import fft, ifft
import scipy as scipy
from pydub import AudioSegment
import sys
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import spectrogram



def function_1():
        # Load the audio file
    audio_file = "P:\\Projekte\\29452_Kopi3\\Audio\\Testpressung\\29452-A.wav"
    y, sr = librosa.load(audio_file)

    # Preprocess the audio data
    y = librosa.to_mono(y)
    y /= max(abs(y))
    y, _ = librosa.effects.trim(y)

    # Compute the peak amplitude of the audio
    peak = max(abs(y))

    # Set the threshold value for clipping detection
    threshold = 0.95

    # Detect clipping
    clip_frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    clip_mask = (abs(clip_frames) >= threshold * peak).any(axis=0)
    clip_frames = clip_frames[:, clip_mask]
    clip_times = librosa.frames_to_time(clip_frames.T, sr=sr)

    # Convert the time stamps to a readable format
    clip_times_formatted = []
    for t in clip_times:
        m, s = divmod(t, 60)
        try:
            clip_times_formatted.append(f"{int(m)}:{int(s):02d}")
        except TypeError as err:
            print(err)

    # Print the time stamps when clipping occurs
    if clip_times.any():
        print("Clipping detected at times:", clip_times_formatted)




def function_2():
    audio_file = "P:\\Projekte\\29452_Kopi3\\Audio\\Testpressung\\29452-A.wav"
    #audio_file = "P:\\Projekte\\29452_Kopi3\\Audio\\Kopi3_Side-A_Master.wav"
    y, sr = librosa.load(audio_file)

    # Preprocess the audio data
    y = librosa.to_mono(y)
    y /= max(abs(y))

    # Compute the peak amplitude of the audio
    peak = max(abs(y))

    # Compute the threshold for clipping detection
    threshold = 0.95

    # Detect clipping
    clip_frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    clip_mask = (abs(clip_frames) >= threshold * peak).any(axis=0)
    clip_times = librosa.frames_to_time(np.arange(clip_frames.shape[-1]), sr=sr)[clip_mask]

    # Convert the clipping time stamps to a readable format
    clip_times_formatted = []
    for t in clip_times:
        m, s = divmod(t, 60)
        clip_times_formatted.append(f"{int(m)}:{int(s):02d}")

    # Compute the threshold for silence detection
    silence_threshold = np.nanpercentile(y, 10)

    # Detect silent sections
    is_silent = y < silence_threshold
    onsets = np.where(np.diff(is_silent.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(is_silent.astype(int)) == -1)[0] + 1

    # Convert the silence time stamps to a readable format
    silence_times_formatted = []
    for onset, offset in zip(onsets, offsets):
        onset_time = librosa.frames_to_time(onset, sr=sr)
        offset_time = librosa.frames_to_time(offset, sr=sr)
        m1, s1 = divmod(onset_time, 60)
        m2, s2 = divmod(offset_time, 60)
        silence_times_formatted.append(f"{int(m1)}:{int(s1):02d} - {int(m2)}:{int(s2):02d}")

    # Print the time stamps of clipping and silent sections
    if clip_times.any():
        print("Clipping detected at times:", clip_times_formatted)
    if onsets.any():
        print("Silent sections detected at times:", silence_times_formatted)





def function_3():
    audio_file = "P:\\Projekte\\29452_Kopi3\\Audio\\Testpressung\\29452-A.wav"

    y, sr = librosa.load(audio_file)

    # Preprocess the audio data
    y = librosa.to_mono(y)
    y /= max(abs(y))

    # Compute the peak amplitude of the audio
    peak = max(abs(y))

    # Compute the threshold for clipping detection
    threshold = 0.95

    # Detect clipping
    clip_frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    clip_mask = (abs(clip_frames) >= threshold * peak).any(axis=0)
    clip_indices = np.where(clip_mask)[0]
    clip_times = librosa.frames_to_time(clip_indices, sr=sr)

    # Convert the clipping time stamps to a readable format
    clip_times_formatted = []
    for t in clip_times:
        m, s = divmod(t, 60)
        clip_times_formatted.append(f"{int(m)}:{int(s):02d}")

    # Compute the threshold for silence detection
    silence_threshold = np.nanpercentile(y, 10)

    # Detect silent sections
    is_silent = y < silence_threshold
    onsets = np.where(np.diff(is_silent.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(is_silent.astype(int)) == -1)[0] + 1

    # Convert the silence time stamps to a readable format
    silence_times_formatted = []
    for onset, offset in zip(onsets, offsets):
        onset_time = onset / sr
        offset_time = offset / sr
        m1, s1 = divmod(onset_time, 60)
        m2, s2 = divmod(offset_time, 60)
        silence_times_formatted.append(f"{int(m1)}:{int(s1):02d} - {int(m2)}:{int(s2):02d}")

    # Print the time stamps of clipping and silent sections
    if clip_times.any():
        print("Clipping detected at times:", clip_times_formatted)
    if onsets.any():
        print("Silent sections detected at times:", silence_times_formatted)



def function_4():

    SAMPLERATE = 44100

    # Load the audio files
    pfad = "P:\\Projekte\\29452_Kopi3\\Audio\\Testpressung\\29452-A.wav"
    audio_original = librosa.load("P:\\Projekte\\29452_Kopi3\\Audio\\Kopi3_Side-A_Master.wav", sr=SAMPLERATE, duration=30)[0]
    audio_digitalisierte_pressung = librosa.load("P:\\Projekte\\29452_Kopi3\\Audio\\Testpressung\\29452-A.wav", sr=SAMPLERATE, duration=30)[0]






    # Pad the shorter audio file with zeros
    if len(audio_original) < len(audio_digitalisierte_pressung):
        audio_original = np.pad(audio_original, (0, len(audio_digitalisierte_pressung) - len(audio_original)), 'constant')
    else:
        audio_digitalisierte_pressung = np.pad(audio_digitalisierte_pressung, (0, len(audio_original) - len(audio_digitalisierte_pressung)), 'constant')

    

    sys.exit()
    # Synchronize the files using cross-correlation
    corr = ifft(fft(audio_original) * np.conj(fft(audio_digitalisierte_pressung)))
    offset = np.argmax(np.abs(corr)) - (len(audio_original) - 1)


    print(audio_original.shape[0])

    # # Trim the longer file to match the length of the shorter file
    # if offset > 0:
    #     audio_digitalisierte_pressung = audio_digitalisierte_pressung[offset:]
    #     audio_original = audio_original[:len(audio_digitalisierte_pressung)]
    # else:
    #     audio_original = audio_original[-offset:]
    #     audio_digitalisierte_pressung = audio_digitalisierte_pressung[:len(audio_original)]


    print(audio_original.shape[0])
    print(audio_digitalisierte_pressung.shape[0])


    # Calculate the mean absolute error between the samples
    mae = np.mean(np.abs(audio_original - audio_digitalisierte_pressung))
    print(mae)

    # Set a threshold for the difference
    threshold = 1

    # Output the timestamps where the difference exceeds the threshold

    for i in range(len(audio_original)):
        if np.abs(audio_original[i] - audio_digitalisierte_pressung[i]) > threshold:
            print(audio_original[i])
            print(audio_digitalisierte_pressung[i])
            print(np.abs(audio_original[i] - audio_digitalisierte_pressung[i]))
            print(f"Difference detected at timestamp {i / SAMPLERATE}s") #librosa.get_samplerate(pfad)



def find_audio_clip(long_audio_clip, short_audio_clip, samplerate):

    #  find macthes
    # match = np.correlate(long_audio_clip, short_audio_clip, mode='valid')  #  mode valid = only perform where signals overlap
    # print(match)
    # print(len(match))
    match_scipy = scipy.signal.correlate(long_audio_clip, short_audio_clip, mode="valid", method="fft")
    # print(match_scipy)
    # print(len(match_scipy))

    # Get the index of the best match
    # best_match_index = np.argmax(match)
    best_match_index_scipy = np.argmax(match_scipy)

    # Calculate the time in seconds where the clip has been found
    # time_found = best_match_index / samplerate
    time_found_scipy = best_match_index_scipy / samplerate
    # print(time_found)
    #print(time_found_scipy)

    return time_found_scipy



    # Pad the shorter audio file with zeros
    # if len(audio_original) < len(audio_digitalisierte_pressung):
    #     audio_original = np.pad(audio_original, (0, len(audio_digitalisierte_pressung) - len(audio_original)), 'constant')
    # else:
    #     audio_digitalisierte_pressung = np.pad(audio_digitalisierte_pressung, (0, len(audio_original) - len(audio_digitalisierte_pressung)), 'constant')

    

    # Synchronize the files using cross-correlation
    # corr = ifft(fft(audio_original) * np.conj(fft(audio_digitalisierte_pressung)))
    # offset = np.argmax(np.abs(corr)) - (len(audio_original) - 1)


    # print(audio_original.shape[0])

    # # Trim the longer file to match the length of the shorter file
    # if offset > 0:
    #     audio_digitalisierte_pressung = audio_digitalisierte_pressung[offset:]
    #     audio_original = audio_original[:len(audio_digitalisierte_pressung)]
    # else:
    #     audio_original = audio_original[-offset:]
    #     audio_digitalisierte_pressung = audio_digitalisierte_pressung[:len(audio_original)]


    print(audio_original.shape[0])
    print(audio_digitalisierte_pressung.shape[0])


    sys.exit()


    # Calculate the mean absolute error between the samples
    mae = np.mean(np.abs(audio_original - audio_digitalisierte_pressung))
    print(mae)

    # Set a threshold for the difference
    threshold = 1

    # Output the timestamps where the difference exceeds the threshold

    for i in range(len(audio_original)):
        if np.abs(audio_original[i] - audio_digitalisierte_pressung[i]) > threshold:
            print(audio_original[i])
            print(audio_digitalisierte_pressung[i])
            print(np.abs(audio_original[i] - audio_digitalisierte_pressung[i]))
            print(f"Difference detected at timestamp {i / SAMPLERATE}s") #librosa.get_samplerate(pfad)


def gabor_transform(clip_1, clip_2) -> None:

    # Gabor Transform (STFT) parameters
    window_length = 1024
    overlap = 0.75

    # Compute spectrograms for both signals
    _, _, spectrogram1 = spectrogram(clip_1, fs=22050, window='hann', nperseg=window_length, noverlap=int(window_length * overlap))
    _, _, spectrogram2 = spectrogram(clip_2, fs=22050, window='hann', nperseg=window_length, noverlap=int(window_length * overlap))

    # Frequency bins and time bins for the spectrograms
    frequency_bins = np.fft.fftfreq(window_length, 1 / 22050)
    time_bins = np.arange(spectrogram1.shape[1])

    # Reshape the spectrogram arrays
    spectrogram1 = spectrogram1[:len(frequency_bins) - 1, :]
    spectrogram2 = spectrogram2[:len(frequency_bins) - 1, :]

    # Plot the spectrograms
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.pcolormesh(time_bins, frequency_bins[:-1], 10 * np.log10(spectrogram1), shading='auto', cmap='inferno')
    plt.colorbar(label='Power Spectral Density (dB)')
    plt.title('Signal 1 Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 1, 2)
    plt.pcolormesh(time_bins, frequency_bins[:-1], 10 * np.log10(spectrogram2), shading='auto', cmap='inferno')
    plt.colorbar(label='Power Spectral Density (dB)')
    plt.title('Signal 2 Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


#  CHroma:
    


# pfad = "P:\\Projekte\\29452_Kopi3\\Audio\\Kopi3_Side-A_Master.wav"
# pfad2 = "P:\\Projekte\\29452_Kopi3\\Audio\\Testpressung\\29452-A.wav"



# # y, samplerate = librosa.load(pfad2, sr=None)
# # print(samplerate)


# import librosa


# y, sr = librosa.load(pfad, sr=22050, duration=60, mono=True)
# #y = librosa.resample(y, orig_sr=sr, target_sr=44100)
# y_2, sr_2 = librosa.load(pfad2, sr=22050, duration=60, mono=True)
# #y_2 = librosa.resample(y_2, orig_sr=sr_2, target_sr=44100)
# # print(sr, sr_2)
# # print(y.shape, y_2.shape)

# # print(y.shape[0] / sr)  # duartion of file in sec
# # print(y_2.shape[0] / sr_2)  # duartion of file in sec



# onset = librosa.onset.onset_detect(y=y_2, sr=22050, units='time')


# print(onset)


def extract_chroma(file_path, save_path):
    y = librosa.load(file_path)[0]
    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=44100)
    np.save(save_path, chroma)
    return chroma



def compute_crp(chroma1, chroma2):
    # Transpose chroma features
    chroma1 = chroma1.T
    chroma2 = chroma2.T

    # Normalize chroma features
    chroma1 = chroma1 / np.sum(chroma1, axis=1)[:, None]
    chroma2 = chroma2 / np.sum(chroma2, axis=1)[:, None]

    # Compute cross recurrence plot
    crp = np.zeros((chroma1.shape[0], chroma2.shape[0]))
    for i in range(chroma1.shape[0]):
        for j in range(chroma2.shape[0]):
            crp[i, j] = np.sum(np.minimum(chroma1[i], chroma2[j]))
            
    # Compute difference
    diff = np.linalg.norm(crp)
    
    return diff


# Example usage
file1 = 'P:\\Projekte\\29167_WRLP003\\Audio\\Testpressung\\segment_original_0.wav'
file2 = 'P:\\Projekte\\29167_WRLP003\\Audio\\Testpressung\\segment_tp_.wav'

chroma1 = extract_chroma(file1, 'chroma1.npy')
chroma2 = extract_chroma(file2, 'chroma2.npy')

# Compute difference
diff = compute_crp(chroma1, chroma2)

print(f"Difference: {diff}")



chroma1 = np.load('chroma1.npy')
chroma2 = np.load('chroma2.npy')




# Load chroma features
chroma1 = np.load('chroma1.npy')
chroma2 = np.load('chroma2.npy')

# Plot chroma features
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(chroma1, y_axis='chroma')
plt.title('Chroma Representation of File 1')
plt.colorbar()

plt.subplot(1, 2, 2)
librosa.display.specshow(chroma2, y_axis='chroma')
plt.title('Chroma Representation of File 2')
plt.colorbar()

plt.tight_layout()
plt.show()