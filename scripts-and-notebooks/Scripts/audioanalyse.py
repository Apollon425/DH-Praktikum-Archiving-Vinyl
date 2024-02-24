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
import os



def compare_audiofiles(path_original_file: str, path_vinyl_digital_file: str, samplerate=44100, segment_duration=0.01, compare_first_x_chunks: int = 500, save_clips: bool = False):
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

    #  iterate over segments and compute their difference considering different audio features
    error_counter = 0
    for index, segment in enumerate(segments_original):
        #print(index)
        if index == compare_first_x_chunks:
            print(f"index: {index}")
            print(f"{error_counter} segments with problematic artifacts found.")
            return

        time_start_clip_vinyl = (index * segment_duration) + music_start_vinyl
        segment_vinyl = make_amplitude_equal(segment, segments_vinyl[index])

        if save_clips:

            os.mkdir("..\\Temp")
            sf.write(f"..\\Temp\\Audio_Samples\\segment_original_{index}.wav", segment, samplerate)
            sf.write(f"..\\Temp\\Audio_Samples\\segment_vinyl_{index}.wav", segment_vinyl, samplerate)

        #  try calculate MFCC:
        mfcc_orig = np.mean(librosa.feature.mfcc(y=segment, sr=samplerate, n_mfcc=50))
        mfcc_digital_vinyl = np.mean(librosa.feature.mfcc(y=segment_vinyl, sr=samplerate, n_mfcc=50))
        euclidean_distance = np.linalg.norm(mfcc_orig - mfcc_digital_vinyl)
        print(f"Euclidean distance between MFCCs at index {index}: {euclidean_distance}")

        #  try find peaks:
        peaks_orig,_ = scipy.signal.find_peaks(segment,height=0)
        peaks_tp,_ = scipy.signal.find_peaks(segment_vinyl,height=0)


        print(f"Number of peaks for index {index}: {len(peaks_orig)}")
        print(f"Number of peaks for index {index}: {len(peaks_tp)}")
        


        #  now compute the differences between the x second clip:

        #  method one: fft + MSE/MAE of absolute differences TODO: make threshold dynamic and return it from fft_correlation

        fft_result_original = fft(segment)
        fft_result_vinyl = fft(segment_vinyl)
        

        try:
            threshold = 10
            distance = fft_correlation(fft_result_original, fft_result_vinyl, "MSE")  

            if distance > threshold:
                #print(f"Error detected at timestamp: {time_start_clip_vinyl}s. Magnitude: {distance}")
                error_counter = error_counter + 1

            #TODO: weitere features extrahieren, aggregieren usw.    

            extract_power_spectrum(fft_result_original, fft_result_vinyl, samplerate, segment_duration)
            sys.exit()



            # #  method two (correlation) --> method two im anschluss an method one in eine methode kombinieren?
            # corr = ifft(fft(segment) * np.conj(fft(segment_vinyl)))
            # print(f"corr: {corr}")
            # offset = np.argmax(np.abs(corr)) - (len(segment) - 1)
            # print(f"offset: {offset}")

            # match = np.correlate(segment_vinyl, segment, mode='valid')  #  mode valid = only perform where signals overlap
            # print(f"match: {match}")





        except ValueError as err:
            print(f"Clips with index {index} are not of the same size.")
            print(err)

    print(f"{error_counter} Fehler gefunden.")

           
    #  method 2: Gabor Transform:
    #gabor_transform(segment, segment_vinyl)





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
        print(num_segments)
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

def fft_correlation(original_fft: np.ndarray, vinyl_fft: np.ndarray, error_measure: str = "MSE") -> float:
    """
    Uses FFT algorithm to calculate the DFT of original and digitized vinyl reproduction.
    Aggregates and returns their differences using the specified distance measure.
    """

    match_scipy = scipy.signal.correlate(original_fft, vinyl_fft, mode="valid", method="fft")
    print(match_scipy)
    best_match_index_scipy = np.argmax(match_scipy)
    worst_match_index_scipy = np.argmin(match_scipy)
    print(best_match_index_scipy)
    print(worst_match_index_scipy)


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
    plt.savefig(f'..\\Temp\\fft_frequency_for_second_{index * segment_duration}')
    


if __name__=='__main__':


    compare_audiofiles(path_original_file="D:\\Audio\\Komplett\\Original\\10253 WRLP003 VINYLMASTER SIDE A.wav", path_vinyl_digital_file="D:\\Audio\\Komplett\\Vinyl\\29167-A Kratzer.wav", segment_duration=1, compare_first_x_chunks=20)


















# # Trim the longer file to match the length of the shorter file
# if offset > 0:
#     segment_vinyl = segment_vinyl[offset:]
#     segment = segment[:len(segment_vinyl)]
# else:
#     segment = segment[-offset:]
#     segment_vinyl = segment_vinyl[:len(segment)]

# #mae = np.mean(np.abs(segment - segment_vinyl))
# match = np.correlate(segment_vinyl, segment, mode='valid')  #  mode valid = only perform where signals overlap
# print(f"match 2: {match}")