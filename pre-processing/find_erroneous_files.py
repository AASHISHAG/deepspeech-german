import pandas as pd
from audio import audiofile_to_input_vector
N_CONTEXT = 9

df = pd.read_csv('tuda-de.csv')

def aftiv_length(row):
    return audiofile_to_input_vector(row['wav_filename'], 26, N_CONTEXT).shape[0] - 2*N_CONTEXT

def trans_length(row):
    return len(row['transcript'])

df['aftiv_len'] = df.apply(aftiv_length, axis=1)
df['trans_len'] = df.apply(trans_length, axis=1)
df['good_flag'] = df.aftiv_len > df.trans_len

df.to_csv("final.csv")