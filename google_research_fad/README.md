## This repo is derived from:

[https://github.com/google-research/google-research/tree/master/frechet_audio_distance](https://github.com/google-research/google-research/tree/master/frechet_audio_distance)

## and is used as a submodule for:
[https://github.com/mcomunita/ccwavegan](https://github.com/mcomunita/ccwavegan)

<br>

---

<br>

Instruction on how to compute the Frèchet Audio Distance are available from the source repo [here](https://github.com/google-research/google-research/tree/master/frechet_audio_distance), but to make it work and to reduce the repo to the bare minimum to compute FAD, I did the following:

**clone repo**
```
$ git clone https://github.com/google-research/google-research.git
$ cd google-research
```

**remove unnecessarty stuff**
+ remove .git folder
+ remove all folders but frechet_audio_distance
  
```
+ rm .travis.yml
+ rm compile_protos.sh
```

**init new repo**
```
$ git init
$ git add .
$ git commit -m “removed all unnecessary files for fad”
```

**prepare venv**
```
$ python3 -m venv .venv_fad
$ source .venv_fad/bin/activate
$ python -m pip install --upgrade pip
$ pip install apache-beam numpy scipy tensorflow
```

**get vgg-ish architecture**
```
$ mkdir tensorflow_models
$ touch tensorflow_models/__init__.py
$ svn export https://github.com/tensorflow/models/trunk/research/audioset tensorflow_models/audioset/
$ rm -r tensorflow_models/audioset/yamnet
$ touch tensorflow_models/audioset/__init__.py
$ touch tensorflow_models/audioset/vggish/__init__.py
```

**get pre-trained vgg-ish**
```
$ mkdir -p data
$ curl -o data/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```

**modify frechet_audio_distance/audioset_model.py:**
- lines 25:27
```
from tensorflow_models.audioset.vggish import mel_features
from tensorflow_models.audioset.vggish import vggish_params
from tensorflow_models.audioset.vggish import vggish_slim
```
 - line 90
```
for i in range(0, samples - vggish_params.SAMPLE_RATE + 1,
```


**modify tensorflow_models/audioset/vggish/vggish_slim.py:**
- line 36
```
import tensorflow_models.audioset.vggish.vggish_params as params
```

**modify frechet_audio_distance/fad_utils.py:**
- line 37:
```
tf_record = tf.python_io.tf_record_iterator(filename).__next__()
```

**also**
```
pip install tf_slim
```

**Test FAD computation**
```
$ python -m frechet_audio_distance.gen_test_files --test_files "test_audio"
```
```
$ ls test_audio/background/*  > test_audio/test_files_background.csv
$ ls test_audio/test1/*  > test_audio/test_files_test1.csv
$ ls test_audio/test2/*  > test_audio/test_files_test2.csv
```

```
$ mkdir -p stats
$ python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_background.csv --stats stats/background_stats
$ python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_test1.csv --stats stats/test1_stats
$ python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_test2.csv --stats stats/test2_stats
```

```
$ python -m frechet_audio_distance.compute_fad --background_stats stats/background_stats --test_stats stats/test1_stats
$ python -m frechet_audio_distance.compute_fad --background_stats stats/background_stats --test_stats stats/test2_stats
```

**Good Luck!**