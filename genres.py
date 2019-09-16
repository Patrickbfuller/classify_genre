import numpy as np
import pandas as pd
import librosa
import requests
import bs4
import youtube_dl
from glob import glob
import os
import json

def extract_segment_features(y, sr):
    """
    Extract audio features from a segment of audio using librosa.

    Input: An array of a audiofile.

    Output: Dictionary of segments with keys:
        tempo, beats, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr,
        and mfcc values from 1-12. 
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    
    seg_features = {
        "tempo" : tempo,
        "beats" : beats.shape[0],
        "chroma_stft" : chroma_stft.mean(),
        "rms" : rms.mean(),
        "spec_cent" : spec_cent.mean(),
        "spec_bw" : spec_bw.mean(), 
        "rolloff" : rolloff.mean(), 
        "zcr" : zcr.mean()
    }
    seg_features.update(
        {f"mfcc{i+1}": mfcc.mean() for i, mfcc in enumerate(mfccs)}
    )
    return seg_features

def listen(fp:str, genre:str, duration=240, granularity=10):
    """
    Extract audio features from a song by segment. Split a song into segments
    and extract sonic features for each segment using librosa. Return a pandas
    DataFrame of features.

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
        if np.any(y):
            seg_features = {'song': fp, 'genre': genre}
            seg_features.update(extract_segment_features(y, sr))
            song_rows.append(seg_features)
    return pd.DataFrame(song_rows)

#### Split to separate .py maybe?

def get_urls(url:str):
    """
    Given a url for a youtube playlist, return a list of the urls for 
    the videos in the playlist, using beautiful soup.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return "RESPONSE_ERROR"
    soup = bs4.BeautifulSoup(response.text, features="html5lib")
    url_suffixes = [
        a['href'] for a in soup.find_all('a', class_='pl-video-title-link')
        ]
    return ['https://www.youtube.com' + sfx for sfx in url_suffixes]

def get_m4a(url, i):
    """
    Write Me
    """
    ydl = youtube_dl.YoutubeDL(
        params={
            'format': 'bestaudio/best',
            'quiet': True,
            'ignoreerrors': True,
            'outtmpl': 'data/%(title)s.%(ext)s',
            'playliststart': i,
            'playlistend': i,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '192',
                }],
            }
        )
    ydl.download([url])

def get_rows_from_m4a(url:str, genre:str):
    """
    Given the m4a audio extract sonic features from segments of an audio file
    to dictionaries. Remove the m4a and return a the data in json format.
    """
    m4a_fp = glob('data/*.m4a')[0]
    song_rows = listen(m4a_fp, genre)
    os.remove(m4a_fp)
    return song_rows.to_json(orient='records', lines=True)

def collect_genre_features(genres:dict, data_fp="data/genre_features.json"):
    """
    Given a dictionary with keys and values: name of genre and url of youtube
    playlists corresponding to the genre, extract sonic features for each song
    in segments. Store the features in json file with default as
    'data/genre_features.json'
    """
    with open(data_fp, 'a+') as f:
        for genre in genres:
            urls = get_urls(genres[genre])
            for i, url in enumerate(urls):
                try:
                    get_m4a(url, i+1)
                    song_json = get_rows_from_m4a(url, genre)
                    f.write(song_json)
                    f.write('\n')
                except Exception:
                    pass