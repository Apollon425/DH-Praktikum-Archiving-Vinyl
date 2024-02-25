import librosa
import numpy as np
import pandas as pd
import math
from scipy.fft import fft, ifft
import scipy as scipy
from scipy.signal import spectrogram
from scipy import signal
from pydub import AudioSegment
import sys
import os
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from dtw import dtw



def compare_audiofiles(path_original_file: str, path_vinyl_digital_file: str, samplerate=44100, segment_duration=0.01, compare_first_x_chunks: int = 500, save_clips: bool = False,
                       window_type='hann', frame_length=256, hop_length=128):
    """
    Implements the original-vinyl comparison pipeline.
    Splits signals to compare into segments and synchronizes them by trying to find the vinyl offset
    by looking at first ten seconds of original and search in vinyl digitalization.
    Then iterates over synchronized segments and compares several audio features (FFT, Power Spectrum, Peaks, Amplitude Envelope).
    Optionally saves plots and segments that were compared.
    """

    #  Load the audio files
    audio_original_first_chunk, _ = librosa.load(path=path_original_file, sr=samplerate, duration=5, mono=True)
    audio_digitalisierte_pressung_first_chunk, _ = librosa.load(path=path_vinyl_digital_file, sr=samplerate, duration=15, mono=True)


    #  synchronize by calculating offset in tp-file
    music_start_vinyl = find_audio_clip(audio_digitalisierte_pressung_first_chunk, audio_original_first_chunk, samplerate)
    print(f"Music starts in Vinyl at: {music_start_vinyl}")


    #  segment them into clips of second_duration
    segments_original = split_audio(path_original_file, segment_duration=segment_duration, samplerate=samplerate)
    segments_vinyl = split_audio(path_vinyl_digital_file, segment_duration=segment_duration, offset=music_start_vinyl, samplerate=samplerate)
    
    def fft_with_windowing(segment):
        window = signal.get_window(window_type, frame_length)
        return np.fft.fft(librosa.stft(segment, n_fft=frame_length, hop_length=hop_length, window=window)[0])

    #  iterate over segments and compute their difference considering different audio features
    error_counter = 0

    columns = ['segment number', 'timecode Vinyl', 'MFCC', 'FFT (MSE)', 'FFT (Corr)', 'Avg. Intensity Diff.']
    df = pd.DataFrame(columns=columns)
    df = df.reindex(range(int(compare_first_x_chunks * segment_duration)))

    for index, segment in enumerate(segments_original):
        #print(index)
        if index == compare_first_x_chunks:
            print(f"index: {index}")
            print(f"{error_counter} segments with problematic artifacts found.")
            df.to_csv("Temp\\Fehlermaße.csv")
            df.to_excel("Temp\\Fehlermaße.xlsx")
            return

        time_start_clip_vinyl = (index * segment_duration) + music_start_vinyl
        #segment = librosa.effects.preemphasis(segment, coef=0.99)
        #segment_vinyl = librosa.effects.preemphasis(segments_vinyl[index], coef=0.99)

        segment_vinyl = make_amplitude_equal(segment, segments_vinyl[index])

        



        if save_clips:
            if not os.path.exists("Temp"):
                os.mkdir(f"Temp")
            
            sf.write(f"Temp\\segment_original_{index}.wav", segment, samplerate)
            sf.write(f"Temp\\segment_vinyl_{index}.wav", segment_vinyl, samplerate)
        
        #  Compare time domain features

        find_peaks(segment, segment_vinyl, index)

        #  MFCC:
        mfcc = calculate_mfcc(segment, segment_vinyl, samplerate, index)



        #  dtw distance:
        #dtw_dist = dtw_distance(segment, segment_vinyl)

        #  intensity difference:
        intensity_diff = intensity_difference_stft(segment_original=segment, segment_vinyl=segment_vinyl)

        # bands = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]
        # intensity_diffs = intensity_difference_bands(segment, segment_vinyl, bands, samplerate=samplerate)
        # print(f"Intensity differences by band: {intensity_diffs}")

        
        #  Compare frequency domain features
        fft_result_original = fft(segment)
        fft_result_vinyl = fft(segment_vinyl)
        # fft_result_original = fft_with_windowing(segment)
        # fft_result_vinyl = fft_with_windowing(segment_vinyl)
        
        fft_distance = 0
        fft_corr = 0
        try:
            threshold = 10

            #  method one: fft + MSE/MAE of absolute differences
            fft_distance = fft_difference(fft_result_original, fft_result_vinyl, "MSE")  
            print(f"Distance FFT: {fft_distance}")
            
            #  method two fft + correlation
            fft_corr = fft_similarity_norm_corr(fft_result_original, fft_result_vinyl)
            print(f"Corr. FFT: {fft_corr}")


            #  plot power spectrum
            extract_power_spectrum(fft_result_original, fft_result_vinyl, index, samplerate, segment_duration)      

            #  count erros
            if fft_distance > threshold:
                #print(f"Error detected at timestamp: {time_start_clip_vinyl}s. Magnitude: {fft_distance}")
                error_counter = error_counter + 1
         



        except ValueError as err:
            print(f"Error for segments of index {index}.")
            print(err)


        df.loc[index] = [index, time_start_clip_vinyl, mfcc, fft_distance, fft_corr, intensity_diff]

    print(f"{error_counter} Fehler gefunden.")
    df.to_csv("Temp\\Fehlermaße.csv")




def split_audio(file_path, segment_duration=0.01, samplerate=44100, offset=0) -> list:
    """
    Split audio files into smaller segments.
    """

    # Load audio file
    audio, sr = librosa.load(file_path, sr=samplerate, offset=offset, mono=True)

    # Calculate segment length in samples
    segment_length = int(segment_duration * sr)


    # Split audio into segments
    if((len(audio) // segment_length) % segment_duration == 0):
        num_segments = len(audio) // segment_length
        #print(num_segments)
    else:
        num_segments = (len(audio) // segment_length) + 1
    

    segments = []
    for i in range(num_segments):
        start_sample = math.floor((i * segment_length))
        end_sample = math.floor(((i + 1) * segment_length))
        segment = audio[start_sample:end_sample]

        segments.append(segment)


    return segments



def find_audio_clip(long_audio_clip, short_audio_clip, samplerate) -> float:
    """
    Finds short_audio_clip in long_audio_clip by using cross-correlation.
    Returns offset by which the music starts later in the vinyl digitalization.
    Used to synchrinize vinyl and original to make comparisons segment by segment possible.

    """

    #  find macthes
    match_scipy = scipy.signal.correlate(long_audio_clip, short_audio_clip, mode="valid", method="fft")  #  mode valid = only perform where signals overlap

    # Get the index of the best match
    best_match_index_scipy = np.argmax(match_scipy)

    # Calculate the time in seconds where the clip has been found
    time_found_scipy = best_match_index_scipy / samplerate
    #print(time_found_scipy)

    return time_found_scipy



def make_amplitude_equal(signal_1: np.ndarray, signal_2: np.ndarray) -> np.ndarray:
    """
    Calculates the Root Mean Square (RMS) of each audio signal, capturing its overall loudness.
    Divides the RMS of the first signal by the RMS of the second, creating a scaling factor.
    Multiplies the second signal by the scaling factor to adjust its loudness.
    """

    rms_audio1 = np.sqrt(np.mean(signal_1**2))
    rms_audio2 = np.sqrt(np.mean(signal_2**2))

    # Compute the scaling factor
    scale = rms_audio1 / rms_audio2

    # Scale the second audio clip
    audio2_scaled = signal_2 * scale

    return audio2_scaled


def make_amplitude_equal2(signal_1: np.ndarray, signal_2: np.ndarray) -> np.ndarray:
    """
    This method normalizes based on the maximum absolute value of each sample, ensuring they don't exceed the dynamic range of the other signal.
    It prevents potential clipping (distortion) caused by simply scaling based on RMS, especially if signals have different peak amplitudes.
    """
    # Normalize based on maximum absolute value
    max_amp_1 = np.max(np.abs(signal_1))
    max_amp_2 = np.max(np.abs(signal_2))
    scale = max_amp_1 / max_amp_2

    # Apply scaling
    audio2_scaled = signal_2 * scale

    return audio2_scaled


def fft_difference(original_fft: np.ndarray, vinyl_fft: np.ndarray, error_measure: str = "MSE") -> float:
    """
    Uses DFT of original and digitized vinyl reproduction.
    Aggregates and returns their differences using the specified distance measure.

    Args:
        original_fft (np.ndarray): First signal in the frequency domain.
        vinyl_fft (np.ndarray): Second signal in the frequency domain.
        error_measure (str): Option for error measure (MSE/MAE/total_energy) 

    Returns:
        float: Normalized cross-correlation similarity score between -1 and 1.


    """

    difference = abs(original_fft) - abs(vinyl_fft)
    if error_measure == "MSE":

        return np.mean(difference**2)
    
    elif error_measure == "MAE":

        return np.mean(np.abs(difference))
    
    elif error_measure == "total_energy":
        return np.sum(difference**2)
    else:
        print("Specified error measure not supported. Return 0 as error value.")
        return 0
    
def extract_power_spectrum(original_fft: np.ndarray, vinyl_fft: np.ndarray, index: int, samplerate: int = 44100, segment_duration: int = 1):
    """
    """

    power_spectrum_orig = np.abs(original_fft)**2
    power_spectrum_vinyl = np.abs(vinyl_fft)**2
    # print(f"poweert spec orig: {power_spectrum_orig}")
    # print(f"poweert spec tp: {power_spectrum_vinyl}")

    #  smoothing through moving average
    power_spectrum_orig_smooth = scipy.signal.convolve(power_spectrum_orig, np.ones(50) / 50, mode='same')
    power_spectrum_vinyl_smooth = scipy.signal.convolve(power_spectrum_vinyl, np.ones(50) / 50, mode='same')

    # difference_powerspec = abs(power_spectrum_orig_smooth) - abs(power_spectrum_vinyl_smooth)
    # error_powerspec = np.mean(difference_powerspec**2)
    # print(f"Power Spec value at timestamp: {time_start_clip_vinyl}s. Magnitude: {error_powerspec}")


    # # Frequency bins for x-axis
    frequency_bins = np.fft.fftfreq(len(power_spectrum_orig), 1 / samplerate)

    # Normalize power values considering symmetry
    #power_spectrum_orig /= 2  # Divide by 2 to compensate for mirrored energy


    plt.figure()


    plt.plot(frequency_bins, power_spectrum_orig_smooth, label='Signal Original')
    plt.plot(frequency_bins, power_spectrum_vinyl_smooth, label='Signal Vinyl Digitalization', alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0, 10000)
    #plt.xlim(-max(frequency_bins), max(frequency_bins))
    plt.ylabel('Power Spectrum')
    plt.yscale('log')
    plt.ylim(bottom=10**-2.5)
    plt.legend()
    plt.grid(True)
    #plt.show()
    if not os.path.exists("Temp"):
        os.mkdir("Temp")
    plt.savefig(f'Temp\\fft_frequency_for_{index}_duration_{str({segment_duration}).replace(".", "pt")}')


def fft_similarity_norm_corr(segment_original: np.ndarray, segment_vinyl: np.ndarray) -> float:
    """
    Calculates the similarity between two FFT-transformed signals using normalized cross-correlation.

    Args:
        segment_original (np.ndarray): First signal in the frequency domain.
        segment_vinyl (np.ndarray): Second signal in the frequency domain.

    Returns:
        float: Normalized cross-correlation similarity score between -1 and 1.
    """

    fft_orig = np.fft.fft(segment_original)
    fft_vinyl = np.fft.fft(segment_vinyl)

    conj_fft_vinyl = np.conj(fft_vinyl)
    corr = np.fft.ifft(fft_orig * conj_fft_vinyl)
    corr = corr.real / np.linalg.norm(fft_orig) / np.linalg.norm(conj_fft_vinyl)

    return np.max(corr)

from dtw import dtw

def dtw_distance(signal1, signal2):
  """
  Calculates the DTW distance between two audio signals.

  Args:
    signal1: NumPy array representing the first audio signal.
    signal2: NumPy array representing the second audio signal.

  Returns:
    The DTW distance between the signals.
  """
  alignment = dtw(signal1, signal2)

  print(f"DTW Distance: {alignment.distance}")
  return alignment.distance

def find_peaks(segment_original, segment_vinyl, index):

    peaks_orig,_ = scipy.signal.find_peaks(segment_original,height=0)
    peaks_tp,_ = scipy.signal.find_peaks(segment_vinyl,height=0)

    print(f"Number of peaks for index {index}: {len(peaks_orig)}")
    print(f"Number of peaks for index {index}: {len(peaks_tp)}")


def calculate_mfcc(segment_original, segment_vinyl, samplerate, index):
    mfcc_orig = np.mean(librosa.feature.mfcc(y=segment_original, sr=samplerate, n_mfcc=40))
    mfcc_digital_vinyl = np.mean(librosa.feature.mfcc(y=segment_vinyl, sr=samplerate, n_mfcc=40))

    # def minmax_normalize(mfcc):
    #     print(np.min(mfcc))
    #     print(np.max(mfcc))
    #     return (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))

    # mfcc_orig_norm = minmax_normalize(mfcc_orig)
    # mfcc_digital_vinyl_norm = minmax_normalize(mfcc_digital_vinyl)
    #euclidean_distance_mfcc_norm = np.linalg.norm(mfcc_orig_norm - mfcc_digital_vinyl_norm)
    #print(f"Euclidean distance between MFCCs at index {index}: {euclidean_distance_mfcc_norm}")

    euclidean_distance_mfcc = np.linalg.norm(mfcc_orig - mfcc_digital_vinyl)

    print(f"Euclidean distance between MFCCs at index {index}: {euclidean_distance_mfcc}")


    return euclidean_distance_mfcc


def intensity_difference_stft(segment_original, segment_vinyl, window_type='hann', frame_length=1024, hop_length=512):
    """
    Calculates the intensity difference between two segments using STFT and PSD.
    """

    def fft_with_windowing(segment):
        window = signal.get_window(window_type, frame_length)
        return np.fft.fft(librosa.stft(segment, n_fft=frame_length, hop_length=hop_length, window=window)[0])

    # Perform STFT for both segments
    stft_original = fft_with_windowing(segment_original)
    stft_vinyl = fft_with_windowing(segment_vinyl)

    # Calculate PSDs
    psd_original = np.abs(stft_original)**2
    psd_vinyl = np.abs(stft_vinyl)**2

    # Compute and average intensity difference across frequency bins
    intensity_diff = psd_vinyl - psd_original
    avg_intensity_diff = np.mean(intensity_diff)

    return avg_intensity_diff

def intensity_difference_bands(segment_original, segment_vinyl, bands, window_type='hann', frame_length=1024, hop_length=512, samplerate=44100):
    """
    Calculates the intensity difference between two segments for specific frequency bands.

    Args:
        segment_original (np.ndarray): First segment.
        segment_vinyl (np.ndarray): Second segment.
        bands (list of tuples): List of frequency ranges (e.g., [(0, 2000), (2000, 4000), ...]).

    Returns:
        list of floats: Intensity difference for each frequency band.
    """

    def fft_with_windowing(segment):
        window = signal.get_window(window_type, frame_length)
        return np.abs(librosa.stft(segment, n_fft=frame_length, hop_length=hop_length, window=window)[0])

    # Perform STFT for both segments
    stft_original = fft_with_windowing(segment_original)
    stft_vinyl = fft_with_windowing(segment_vinyl)

    # Calculate frequency bins based on sampling rate and window parameters
    frequency_bins = np.linspace(0, samplerate / 2, int(frame_length / 2) + 1)

    # Calculate and return intensity differences for each band
    intensity_diffs = []
    for low_freq, high_freq in bands:
        mask_original = (frequency_bins >= low_freq) & (frequency_bins < high_freq)
        mask_vinyl = (frequency_bins >= low_freq) & (frequency_bins < high_freq)
        intensity_diff = np.mean(stft_vinyl[mask_vinyl] ** 2) - np.mean(stft_original[mask_original] ** 2)
        intensity_diffs.append(intensity_diff)

    return intensity_diffs


if __name__=='__main__':


    compare_audiofiles(
        path_original_file="D:\\Audio\\Komplett\\Original\\10253 WRLP003 VINYLMASTER SIDE A.wav", 
        path_vinyl_digital_file="D:\\Audio\\Komplett\\Vinyl\\29167-A Kratzer.wav", 
        segment_duration=0.5, 
        compare_first_x_chunks=20,
        window_type='hann', 
        frame_length=1024, 
        hop_length=512,
        save_clips=True)