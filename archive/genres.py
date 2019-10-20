import numpy as np
import pandas as pd
import seaborn as sns
import librosa
import requests
import bs4
import youtube_dl
from glob import glob
import os
import json
import pickle
from io import BytesIO
import base64
from sklearn.metrics import log_loss, jaccard_score
sns.mpl.pyplot.style.use('seaborn')

def generate_random_string():
    """Create a random tag for an unambiguous and unique use."""
    arr = np.random.random(2)
    s = ''.join([str(n).strip('0.') for n in arr])
    return s

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
    DataFrame of features. If no genre specified, return just features.

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
    Download a url and return the filepath of the directory containing the
    audio file.
    """
    dir = generate_random_string()
    outtmpl = f'data/{dir}/%(title)s.%(ext)s'
    ydl = youtube_dl.YoutubeDL(
        params={
            'format': 'bestaudio/best',
            'quiet': True,
            'ignoreerrors': True,
            'outtmpl': outtmpl,
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
    return f'data/{dir}'

def get_rows_from_m4a(url:str, genre:str, dir_fp:str):
    """
    Given the m4a audio extract sonic features from segments of an audio file
    to dictionaries. Remove the m4a and return a the data in json format.
    """
    m4a_fp = glob(f'{dir_fp}/*.m4a')[0]
    song_rows = listen(m4a_fp, genre)
    os.remove(m4a_fp)
    os.rmdir(dir_fp)
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
                    dir_fp = get_m4a(url, i+1)
                    song_json = get_rows_from_m4a(url, genre, dir_fp)
                    f.write(song_json)
                    f.write('\n')
                except Exception:
                    pass

def classify_rows(df):
    """
    Input sonic features of song segments and use pickled classifier model.
    Predict the genre of each row and return the average results for each
    genre. Set location of model depending on if being run from flask app.
    """
    if __name__ == 'archive.genres':
        pkl_fp = 'archive/10genre_clf.pkl'
    else:
        pkl_fp = '10genre_clf.pkl'
    with open(pkl_fp, 'rb') as f:
        model = pickle.load(f)
    
    seg_preds = model.predict_proba(df.drop(['song', 'genre'], axis=1))
    preds = 100 * seg_preds.sum(axis=0)/seg_preds.sum()
    return [(g,p) for g, p in zip(model.classes_, preds.round(3))]

def classify(url=None, m4a_fp=None):
    """
    Input the filepath to m4a audio file or a the url to a youtube video of a
    song. Extract audio features and classify its likelihood of being among the
    genres, Country, Jazz, Hip Hop, Classical, Metal or Electronic.
    """
    if m4a_fp == None:
        if url == None:
            return "Please Specify a url or filepath to an m4a"
        dir_fp = get_m4a(url, 1)
        m4a_fp = glob(f'{dir_fp}/*.m4a')[0]
    df = listen(fp=m4a_fp, genre=' ')
    genre_preds = classify_rows(df)
    if url:     # If the file was downloaded remove it.
        os.remove(m4a_fp)
        os.rmdir(dir_fp)
    return sorted(genre_preds, key=lambda x: x[1], reverse=True)
    
    # with open('genre_clf.pkl', 'rb') as f:
    #     model = pickle.load(f)
    
    # preds = model.predict_proba(df.drop(['song', 'genre'], axis=1))
    # return None

def percentify_cm(cm, metric='recall'):
    """
    For each label, depict the percent predicted as each label.
    Input : Confusion Matrix
    Output : Confusion Matrix Percent of True Labels
    """
    n = len(cm)
    if metric == 'recall':
        row_totals = cm.sum(axis=1).reshape(n,1)
        percents = 100 * (cm/row_totals)
    if metric == 'precision':
        column_totals = cm.sum(axis=0)
        percents = 100 * (cm/column_totals)
    return percents.round(1)

def eval_model(y_test, preds, pred_probas, labels):
    genre_scores = jaccard_score(
                y_test, preds,
                average=None,
                labels=labels
            ).round(5)

    print(f"""
    Log Loss:
        {log_loss(y_test, pred_probas, labels=labels)}
    Jaccard:
        {jaccard_score(y_test, preds, average='macro').round(5)}""")
    for label, score in zip(labels, genre_scores):
        print(f"\t-{label.title()}: {score})")

def get_barplot_html(genre_probs:list):
    """
    Input a list of tuples containing (genre and its probability).
    Return HTML with a horizontal barplot of the probabilities.
    """
    genres = [x[0] for x in genre_probs]
    probas = [x[1] for x in genre_probs]
    ax = sns.barplot(probas, genres, orient='h')
    buf = BytesIO()
    ax.figure.savefig(buf, format="jpg")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    sns.mpl.pyplot.close()
    return f"<img src='data:image/jpg;base64,{data}'/>"