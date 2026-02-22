import numpy as np
import pandas as pd
import librosa
import yt_dlp
import soundfile as sf
from scipy.signal import find_peaks
from itertools import groupby


#Function for finding plalists
def get_playlist_links(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,                                       #doesn't output in terminal
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)       #extracts number of songs in playlist

    links = []
    for entry in info['entries']:
        video_id = entry['id']
        links.append(f"https://youtu.be/{video_id}")                #Creates list of every  link to analyze

    return links                                                    #Saves links for videos in playlisy


def get_playlist_links(playlist_url):
    """Return a list of video URLs from a playlist."""
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    return [f"https://youtu.be/{entry['id']}" for entry in info['entries']]

def download_audio(youtube_url, output_file):
    """Download audio and return basic info."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'outtmpl': output_file + ".%(ext)s",
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
    return {
        "title": info.get('title'),
        "uploader": info.get('uploader'),
        "duration": info.get('duration')
    }


def bpmcalc(sample_rate, wav):
    print("========== BPM ANALYSIS ==========")

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)                                  #Converts to multi float (no mono tracks)

    wav = wav.astype(np.float32)

    max_val = np.max(np.abs(wav))
    if max_val > 0:
        wav = wav / max_val

    tempo, _ = librosa.beat.beat_track(y=wav, sr=sample_rate)       #BPM calculator module

    if tempo < 75:                                                  #generally bellow 75 means a song is counting half beats
        tempo *= 2

    print("BPM:", tempo)
    return tempo



def Detect_Misc(sample_rate, wav):
    print("========== GENRE ANALYSIS ==========")

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)      #Same with mono from before

    wav = wav.astype(np.float32)

    zcr = librosa.feature.zero_crossing_rate(y=wav)     #Zero crossing rate, has to do wth speechiness
    print("Zero Cross rate = ", zcr)
    rms = librosa.feature.rms(y=wav)                    #RMS has to do with loudness/energy
    print("Energy = ", rms)

    odf = librosa.onset.onset_strength(y=wav, sr=sample_rate)

    AC = librosa.autocorrelate(odf)
    AC_norm = AC / AC[0] if AC[0] != 0 else AC
    AC_score = np.max(AC_norm[1:]) if len(AC_norm) > 1 else 0       #Autocorrelation measures how much a signal repeats, like how rythmic it is.
    print("Autocorrelation = ", AC_score)

    timbre = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13) #measures type of instrument / sound
    print("Timbre = ", timbre)

    centroid = librosa.feature.spectral_centroid(y=wav, sr=sample_rate) #Center of mass of the song, as in does it build up or settle
    print("Energy = ", centroid)


    return (
        np.mean(zcr),
        np.mean(rms),
        np.std(rms),
        AC_score,
        np.mean(timbre, axis=1),
        np.std(timbre, axis=1),
        np.mean(centroid),
        np.std(centroid),
    )



def SlidingWindow(sample_rate, wav, duration, NoteFile, ChordsFile, num_segments=200):

    print("========== NOTE ANALYSIS ==========")

    if wav.ndim > 1:
        wav = wav[:, 0]

    noteVector = np.zeros(14)
    tab = []

    windows = np.linspace(0, duration, num_segments + 1)

    for win in range(len(windows) - 1):

        start = int(windows[win] * sample_rate)     #sets up 200 windows to sample in
        end = int(windows[win + 1] * sample_rate)

        segment = wav[start:end]
        if len(segment) == 0:
            continue

        spec = np.fft.rfft(segment)
        freq = np.fft.rfftfreq(len(segment), 1 / sample_rate)   #Fourier transforms the signal

        spec = np.abs(spec)
        threshold = 0.1 * np.max(spec)                          #threshold for counting a frequency

        peaks, _ = find_peaks(spec, height=threshold, distance=10)    #finds peaks in frequency domain

        if len(peaks) == 0:
            continue

        testfreq = freq[peaks]              #array of each frequency
        contributions = spec[peaks]         #array of each note contribution

        norm = np.linalg.norm(contributions) #Normalize contributions
        if norm > 0:
            contributions = contributions / norm

        note_dict = {}

        df = 5                              #Do not match two notes within 5Hz of eachother

        for i in range(len(testfreq)):      #matching frequency to notes and octave, I can't believe I wrote this, I don't understand it anymore, but it works
            for x in range(len(NoteFile.iloc[:, 0]) - 1):   
                for y in range(len(NoteFile.iloc[0, :]) - 1):
                    freq_val = NoteFile.iloc[x + 1, y + 1]
                    if freq_val - df < testfreq[i] < freq_val + df:
                        note_name = NoteFile.iloc[x + 1, 0]
                        weight = contributions[i]

                        noteVector[x + 1] += weight

                        if note_name in note_dict:
                            note_dict[note_name] += weight
                        else:
                            note_dict[note_name] = weight

        if note_dict:
            my_keys = sorted(note_dict, key=note_dict.get, reverse=True)

            for col in range(len(ChordsFile.iloc[0, :])):
                if set(ChordsFile.iloc[1:4, col].values.tolist()) == set(my_keys[:3]):      #Matches chords to notes if I want that
                    tab.append(ChordsFile.iloc[0, col])

    result = [k for k, g in groupby(tab) if len(list(g)) >= 3]

    return result, noteVector



NoteFile = pd.read_csv("notes.csv", header=None)        #notes to match
ChordsFile = pd.read_csv("chords.csv", header=None)     #Chords to match

def process_videos(urls, NoteFile, ChordsFile, output_csv="ForceList.csv"):
    """
    urls: list of YouTube URLs (single video or playlist)
    NoteFile, ChordsFile: DataFrames for note/chord mapping
    output_csv: where to append song vectors
    """
    for url in urls:
        try:
            print("\nProcessing:", url)
            info = download_audio(url, "youtubetest")
            wav, sample_rate = sf.read("youtubetest.wav")        
            wav = wav.astype(np.float32)

            tempo = bpmcalc(sample_rate, wav)
            misc_features = Detect_Misc(sample_rate, wav)
            result, noteVector = SlidingWindow(sample_rate, wav, int(info["duration"]), NoteFile, ChordsFile)
            noteVector[0] = tempo

            SongVector = noteVector.tolist()
            SongVector.extend(misc_features[0:4])
            SongVector.extend(misc_features[4])
            SongVector.extend(misc_features[5])
            SongVector.append(misc_features[6])
            SongVector.append(misc_features[7])
            SongVector.append(info["title"])
            SongVector.append(info["uploader"])

            pd.DataFrame([SongVector]).to_csv(output_csv, mode='a', index=False, header=False)
            print("Finished:", info["title"])

        except Exception as e:
            print("Error:", e)
            print("Skipping...\n")