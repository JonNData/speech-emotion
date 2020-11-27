def extract_audio_features(file_name, mfcc, chroma, mel):
    """
    mfcc represents the short term power spectrum of the sound
    chroma is the pitch
    mel is the spectrogram frequency
    """
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            fourier = np.abs(librosa.stft(X))
            
        # compile the three features into a result    
        result = np.array([])

        if mfcc:
            pwr_spec = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, pwr_spec)) # add to result
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=fourier, 
                                                        sr=sample_rate,
                                                        ).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

def vis_spec(audio_file):
    y, sr = librosa.load(path=audio_file, sr=None)
    plt.figure(figsize=(12, 8))

    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    plt.subplot(4, 2, 1)

    librosa.display.specshow(D, y_axis="linear")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.xlabel('Time')
    plt.show

def vis_harm(audio_file):
    y, sr = librosa.load(path=audio_file, sr=None)
    y_harm, y_perc = librosa.effects.hpss(y)
    plt.subplot(3, 1, 3)
    librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
    librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
    plt.title('Harmonic + Percussive')
    plt.tight_layout()