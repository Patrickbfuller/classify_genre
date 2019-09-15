import numpy as np
import librosa

def listen(fp:str, genre:str, duration=240, granularity=10):
    """
    Extract audio features from a song. Split a song into segments and 
    extract sonic features for each segment using librosa. Return a list
    of dictionaries of features.

    ---
    Input: 
        fp: Filepath of m4a file from which to extract sonic features.

        genre: The genre of the song.

        duration: The length(seconds) into the song to stop extracting
                sonic features.
                -eg. listens to the first x seconds of a song.
                default: 240
        
        granularity: The length(seconds) of individual segments of the song.
                default: 10

    Output: List of dictionaries containing rows 
    """
    ts, sr = librosa.load(
        path=fp,
        sr=None,
        duration=duration
    )

    song_rows = []
    for y in np.array_split(ts, duration/granularity):
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        
        seg_features = {
            'song': fp,
            'genre': genre,
            'tempo' : tempo,
            'beats' : beats.shape[0],
            'chroma_stft' : chroma_stft.mean(),
            'rmse' : rmse.mean(),
            'spec_cent' : spec_cent.mean(),
            'spec_bw' : spec_bw.mean(), 
            'rolloff' : rolloff.mean(), 
            'zcr' : zcr.mean()
        }
        seg_features.update(
            {f'mfcc{i+1}': mfcc.mean() for i, mfcc in enumerate(mfccs)}
        )
        song_rows.append(seg_features)
    return song_rows