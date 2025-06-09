import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

music_path_root = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Musics"
# music_path_root = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/Spectros"

def get_spectrogram(audio_name, start_time, end_time, sr=22050):
    audio_path = os.path.join(music_path_root, audio_name)

    # y, sr = librosa.load(audio_path, sr=sr)
    
    # start_sample = int(start_time * sr)
    # end_sample = int(end_time * sr)
    
    # y_segment = y[start_sample:end_sample]
    
    # D = np.abs(librosa.stft(y_segment))
    
    # D_db = librosa.amplitude_to_db(D, ref=np.max)

    sig, fs = librosa.core.load(audio_path, sr=8000)
    D_db = np.abs(librosa.stft(sig))

    print("fs in get_spectrogram:", fs)
    
    return D_db, fs

def plot_spectro(D_db, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(label="dB")
    plt.title("Spectrogram")
    plt.show()

def combine_spectros(music_indices_vector, seq_idx):
    spectros = []

    fs = None

    for music_idx in music_indices_vector[0]:
        # spectro_music_temp = get_spectrogram(f"music{music_idx + 1}.mp3", 5, 10)
        spectro_music_temp, fs = get_spectrogram("music2.mp3", 15 + (5 * music_idx), 15 + (5 * (music_idx + 1)))

        spectros.append(spectro_music_temp)

        print("shape of spectro_music_temp:", spectro_music_temp.shape)
    
    spectros = np.array(spectros)
    spectros = spectros.reshape(spectros.shape[0] * spectros.shape[1], spectros.shape[-1])
    print("shape of spectros combined:", spectros.shape)

    spectrogram_to_audio(spectros, fs=fs, save_name=f"output_{seq_idx}.mp3")


def spectrogram_to_audio(spec, fs=None, sr=22050, save_name="output.mp3"):
    save_path = os.path.join("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/Spectros", save_name)

    spec = librosa.db_to_amplitude(spec)

    # reconstructed_audio = librosa.griffinlim(spec)

    # sf.write(save_path, reconstructed_audio, sr, format='MP3')
    # print(f"Saved reconstructed audio to {save_path}")

    audio_signal = librosa.core.spectrum.griffinlim(spec)
    # print(audio_signal, audio_signal.shape)
    # librosa.output.write_wav(save_path, audio_signal, fs)
    sf.write(save_path, audio_signal, sr, format='MP3')

def isolated():
    spectro = get_spectrogram("output_120.mp3", 45, 50)
    # spectro = get_spectrogram("music2.mp3", 45, 50)
    print(spectro)
    # spectrogram_to_audio(spectro)

    plot_spectro(spectro, 22050)

# isolated()