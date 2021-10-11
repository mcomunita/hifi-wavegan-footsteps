import os
import numpy as np
from natsort import natsorted, ns
import essentia.standard as es
import torchopenl3
from sklearn.metrics.pairwise import manhattan_distances, cosine_distances, \
                                        euclidean_distances, haversine_distances

def list_folders_in_path(path):
   return [name for name in natsorted(os.listdir(path), alg=ns.IGNORECASE) if os.path.isdir(os.path.join(path, name))]

def filter_files_in_path(path, format='.wav'):
    return natsorted(filter(lambda x: x.endswith(format), os.listdir(path)), alg=ns.IGNORECASE)

def get_vggish_embs_subfolders(model, path, format='.wav'):
    embs = []
    for folder in list_folders_in_path(path):
        for file in filter_files_in_path(path=os.path.join(path,folder), format=format):
            audio = es.MonoLoader(filename=f'{path}/{folder}/{file}', sampleRate=16000)()
            if audio.shape[0] < 16000:
                audio = np.pad(audio, (0, 16000-audio.shape[0]))
            emb = es.TensorflowPredictVGGish(graphFilename=model, output='model/vggish/embeddings')(audio)
            embs.append([folder, file, emb])
    return embs

def get_openl3_embs_subfolders(model, path, format='.wav'):
    embs = []
    for folder in list_folders_in_path(path):
        for file in filter_files_in_path(path=os.path.join(path,folder), format=format):
            audio = es.MonoLoader(filename=f'{path}/{folder}/{file}', sampleRate=16000)()
            # if audio.shape[0] < 16000:
            #     audio = np.pad(audio, (0, 16000-audio.shape[0]))
            if audio.shape[0] > 8000:
                audio = audio[:8000]
            emb, _ = torchopenl3.get_audio_embedding(
                        audio=audio, 
                        sr=16000, 
                        center=False, 
                        model=model)
            embs.append([folder, file, np.array(emb.squeeze().cpu())])
    return embs

def mmd(x, y, distance='manhattan'):
    """
    Args:
        x, y: matrix of embeddings (n_samples * embedding_size)
        distance: distance metric used to compute mmd
    """
    assert distance in ['manhattan', 'euclidean', 'cosine']
    assert x.shape == y.shape

    n_samples = x.shape[0]

    if distance == 'manhattan':
        xy = manhattan_distances(x, y, sum_over_features=True)
        xx = manhattan_distances(x, sum_over_features=True)
        yy = manhattan_distances(y, sum_over_features=True)
    elif distance == 'euclidean':
        xy = euclidean_distances(x, y, squared=False)
        xx = euclidean_distances(x, squared=False)
        yy = euclidean_distances(y, squared=False)
    elif distance == 'cosine':
        xy = cosine_distances(x, y)
        xx = cosine_distances(x)
        yy = cosine_distances(y)

    mmd_ = (1/n_samples**2) * (2*np.sum(xy) - np.sum(xx) - np.sum(yy))
    return mmd_