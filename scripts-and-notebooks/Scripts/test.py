import librosa
import scipy
import numpy as np
import matplotlib.pyplot as plt


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