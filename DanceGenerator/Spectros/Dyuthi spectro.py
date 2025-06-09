import os
import librosa
import numpy as np

def _get_audio_feature(audio_path):
    """Extract feature from audio file."""
    assert os.path.exists(audio_path), f'File {audio_path} does not exist!'
    _log(f"process music feature {audio_path}.")

    audio_feature = None
    if audio_path.endswith(".npy"):
        # load audio feature from feature data file
        with open(audio_path, 'rb') as f:
            audio_feature = np.load(f)
            audio_feature = np.array(audio_feature)  # (N, 35)
            f.close()
    else:
        # fetch audio feature from wav audio file
        FPS = 60
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH

        data, _ = librosa.load(FLAGS.audio_path, sr=SR)
        # (seq_len,)
        envelope = librosa.onset.onset_strength(y=data, sr=SR)
        # (seq_len, 20)
#        mfcc = librosa.feature.mfcc(data, n_mfcc=20, sr=SR).T
        mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T

        # (seq_len, 12)
#        chroma = librosa.feature.chroma_cens(
#            data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
#        ).T
        chroma = librosa.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T

        # (seq_len,)
        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
        )
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0
        # (seq_len,)
        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH, tightness=100.0
        )
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0
        # concat feature (?, 35)
        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma,
            peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)

    # reshape to (1, ?, 35)
    audio_feature = audio_feature[np.newaxis, :, :]
    _log(audio_feature.shape)

    return audio_feature