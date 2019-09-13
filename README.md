# Automatic Speech Recognition (ASR) - DeepSpeech German

_This is the project for the paper [German End-to-end Speech Recognition based on DeepSpeech]() published at [KONVENS 2019](https://2019.konvens.org/)._

This project aims to develop a working Speech to Text module using [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech), which can be used for any Audio processing pipeline. [Mozillla DeepSpeech](https://github.com/mozilla/DeepSpeech) is a state-of-the-art open source automatic speech recognition (ASR) toolkit. DeepSpeech is using a model trained by machine learning techniques based on [Baidu's Deep Speech](https://gigaom2.files.wordpress.com/2014/12/deep_speech3_12_17.pdf) research paper. Project DeepSpeech uses Google's TensorFlow to make the implementation easier.

## Important Links:

**Paper:** 

**Demo:** 

This Readme is written for [DeepSpeech v0.5.0](https://github.com/mozilla/DeepSpeech/releases/tag/v0.5.0). Refer to [Mozillla DeepSpeech](https://github.com/mozilla/DeepSpeech) for lastest updates.

## Contents

1. [Getting Started](#getting-started)
2. [Data-Preprocessing for Training](#data-preprocessing-for-training)
3. [Training](#training)
4. [Some Training Results](#some-training-results)
5. [](#)
6. [Acknowledgments](#acknowledgments)

### Install Python Requirements
```
pip install -r requirements.txt
```

### Language  model
For the language model a text corpus is used, also provided by the people of the " German speech data corpus"
(https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html).

### Speech data

* https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html (~30h)
* http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/ (~50h)
* https://nats.gitlab.io/swc/ (~150h)

### Download the data
1. Download the text corpus from http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz and store it to `text_corpus_path`.
2. Download the German Distant Speech Corpus (TUDA) from http://www.repository.voxforge1.org/downloads/de/german-speechdata-package-v2.tar.gz and store it to `tuda_corpus_path`.
3. Download the Spoken Wikipedia Corpus (SWC) from https://nats.gitlab.io/swc/ and prepare
   it according to https://audiomate.readthedocs.io/en/latest/documentation/indirect_support.html.
4. Download the Voxforge German Speech data (via pingu python library):

```python
from audiomate.corpus import io

dl = io.VoxforgeDownloader(lang='de')
dl.download(voxforge_corpus_path)
```

### Prepare audio data
```
# prepare_data.py creates the csv files defining the audio data used for training
./prepare_data.py $tuda_corpus_path $voxforge_corpus_path $exp_path/data
```

### Language Model

We used [KenLM](https://github.com/kpu/kenlm.git) toolkit to train a 3-gram language model. It is Language Model inference code by [Kenneth Heafield](https://kheafield.com/)

- **Installation**

```
$ git clone https://github.com/kpu/kenlm.git
$ cd kenlm
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j `nproc`
```

- **Corpus**

We used an open-source [German Speech Corpus](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz) released by [University of Hamburg](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html).

- Data pre-processing

```
cd deepspeech-german
./pre-processing/prepare_vocab.py $text_corpus_path $exp_path/clean_vocab.txt
```

- KenLM is used to build the LM
```
$kenlm_bin/lmplz --text $exp_path/clean_vocab.txt --arpa $exp_path/words.arpa --o 3
$kenlm_bin/build_binary -T -s $exp_path/words.arpa $exp_path/lm.binary
```

### Build trie
```
# The deepspeech tools are used to create the trie
$deepspeech/native_client/generate_trie data/alphabet.txt $exp_path/lm.binary $exp_path/clean_vocab.txt $exp_path/trie
```

### Run training
```
./run_training.sh $deepspeech $(realpath data/alphabet.txt) $exp_path
```

## Acknowledgments
* [Prof. Dr.-Ing. Torsten Zesch](https://www.ltl.uni-due.de/team/torsten-zesch) - Co-Author
* [Dipl.-Ling. Andrea Horbach](https://www.ltl.uni-due.de/team/andrea-horbach)
* [Matthias](https://github.com/ynop)
