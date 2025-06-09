from pydub import AudioSegment

def test():
    sound = AudioSegment.from_mp3("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Musics/music2.mp3")

    # len() and slicing are in milliseconds
    halfway_point = len(sound) / 2
    # print("sound length:", type(sound), len(sound))
    second_half = sound[halfway_point:]

    # Concatenation is just adding
    second_half_3_times = second_half + second_half + second_half

    # writing mp3 files is a one liner
    second_half_3_times.export("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/Spectros/output_pydub.mp3", format="mp3")

music_path_root = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Musics"

def combine_musics(music_indices_vector, seq_idx, block_duration=1):
    sounds = []
    music_temp = AudioSegment.from_mp3(f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Musics/music2.mp3")

    for music_idx in music_indices_vector[0]:
        start_temp = (15 + (block_duration * music_idx)) * 1000
        end_temp = (15 + (block_duration * (music_idx + 1))) * 1000
        sounds.append(music_temp[start_temp:end_temp])
    
    sound_final = sounds[0]
    for idx in range(1, len(sounds)):
        sound_final += sounds[idx]
    
    sound_final.export(f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/Spectros/output_{seq_idx}.mp3", format="mp3")

    # spectros = []

    # fs = None

    # for music_idx in music_indices_vector[0]:
    #     # spectro_music_temp = get_spectrogram(f"music{music_idx + 1}.mp3", 5, 10)
    #     spectro_music_temp, fs = get_spectrogram("music2.mp3", 15 + (5 * music_idx), 15 + (5 * (music_idx + 1)))

    #     spectros.append(spectro_music_temp)

    #     print("shape of spectro_music_temp:", spectro_music_temp.shape)
    
    # spectros = np.array(spectros)
    # spectros = spectros.reshape(spectros.shape[0] * spectros.shape[1], spectros.shape[-1])
    # print("shape of spectros combined:", spectros.shape)

    # spectrogram_to_audio(spectros, fs=fs, save_name=f"output_{seq_idx}.mp3")

# test()