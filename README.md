# Automatic Speech Recognition (ASR) - DeepSpeech German

_This is the project for the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech) published at [KONVENS 2019](https://2019.konvens.org/)._

This project aims to develop a working Speech to Text module using [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech), which can be used for any Audio processing pipeline. 
[Mozilla DeepSpeech Architecture](https://deepspeech.readthedocs.io/en/v0.9.3/DeepSpeech.html) is a state-of-the-art open-source automatic speech recognition (ASR) toolkit. DeepSpeech is using a model trained by machine learning techniques based on [Baidu's Deep Speech](https://gigaom2.files.wordpress.com/2014/12/deep_speech3_12_17.pdf) research paper. 
Project DeepSpeech uses Google's TensorFlow to make the implementation easier.

![DeepSpeech](./media/rnn_fig-624x598.png)

## Important Links:

**Paper:** https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech

**DeepSpeech-API:** https://github.com/AASHISHAG/DeepSpeech-API

This Readme is written for [DeepSpeech v0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3). Refer to [Mozillla DeepSpeech](https://github.com/mozilla/DeepSpeech) for latest updates.

## Contents

1. [Requirements](#requirements)
2. [Speech Corpus](#speech-corpus)
3. [Language Model](#language-model)
4. [Training](#training)
5. [Hyper-Parameter Optimization](#hyper-parameter-optimization)
6. [Results](#results)
7. [Trained Models](#trained-models)
8. [Acknowledgments](#acknowledgments)
9. [References](#references)

## Requirements

The instruction are focusing on Linux. It might be possible to use also MacOS and Windows as long as a shell environment can be provided, however some instructions and scripts might have to be adjusted then.

`DeepSpeech` and `KenLM` were added as sub modules. To fetch the sub modules execute:

~~~
git pull --recurse-submodules
~~~

### Developer Information

To update the used DeepSpeech version checkout in the `DeepSpeech` submodule the corresponding tag:

~~~
git checkout tags/v0.9.3
~~~

The same applies to KenLM.

### Installing Python bindings

The DeepSpeech tools require still TensorFlow 1.15, which is only supported up to Python 3.7. [`pyenv`](https://github.com/pyenv/pyenv.git) will be used to set up a dedicated Python version.

~~~
git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
~~~

Follow the instruction on the pyenv page, for Ubuntu Desktop
Add to your `~/.bashrc`:

~~~
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
~~~

~~~
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
~~~

Log out or run:

~~~
source ~/.bashrc
~~~

[`virtualenv`](https://virtualenv.pypa.io/en/latest/installation.html) will be used to provide a separate Python environment.

```
pyenv install 3.7.9
sudo pip3 install virtualenv
virtualenv --python=$HOME/.pyenv/versions/3.7.9/bin/python3.7 python-environments
source python-environments/bin/activate
pip3 install -r python_requirements.txt
```

_NOTE: While writing this 2 issues were fixed in [audiomate](https://github.com/ynop/audiomate) 
and a GitHub patched version was used. This can be updated once  the upstream audiomate has released a new version._ 

### Installing Linux dependencies

The necessary Linux dependencies can be found in `linux_requirements`.

```
xargs -a linux_requirements.txt sudo apt-get install
```

## Speech Corpus

* [German Distant Speech Corpus (TUDA-De)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html)
   * [Download Folder](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/)
* [Mozilla Common Voice](https://commonvoice.mozilla.org/en)
  * [Download](https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-7.0-2021-07-21/cv-corpus-7.0-2021-07-21-de.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3N7H2GD2S%2F20210915%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210915T181038Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEBsaDCl2Q%2B8HrFAwW0rAWCKSBD0BMS3JHJnT7D%2FyUq2hPfv8s%2FUpOe3LxV%2B205PHPTagrpnNr4ZOzwYx9R6R0NvyME1uSfKN5pqYUFbrxewJhx%2BduWM%2FGkJWCLJnxjWeaUD6qslPr4yJNWDsBLQgcGzLU2GpVmqyz7%2FHZ4pej5oLdceEUoSxZkYnhdI93V7pqe%2BpHT2LTJ9UJtCLtR2pc%2BWLpdyZggTnn7brMwztuijRnGp9pYeH87xryVozU0qFq1AAXddlgT%2Bn0%2BD%2Fe81Le9gI3cGyryzwkdT9uwttu6%2B8rZqWv4Ug5S9vROYuJbJMigMG4Lqi9hZMencyBEF4uSHAUsz6Wii4wjnkpOyHHlsjh7C1bNKD2MfCN4Z%2BBrrWFPyMskGfjc9XEuJN3AgjCTTYoUQRpm7ekzSH7PTNvrwgQgdKvntqZB2ARJokr%2BpxRyH%2FPLZ2gqNw2rYSDqtPtMux%2B%2Fx7E8PfddULZWUyfIKSKgJJN8niL90BKbADsPw%2BOg9K8c65vDrHmgBYpS1ivZCB6iM0pFBFNO4LpgkP1qOCaJJ9bGCmfgRt05p2naZBsCd7UG2173FH1PVnMJRA8Gj6bzdm%2FLzlkx%2BSr7820PIMwyGOVr8ne0RiemudEJM%2FxNaPckJZPg2O0AbwV2zdUXxI7OfVd0E0ijCW%2BB4ijOhoGoanHW3sZGFTkng97wRqUmf39o2v4xZja96IxwhRrU5bejV9KKLqiIoGMiqrsyElxtsPoRSfceLxQHnJO7G3l%2BTRmojuMICCn3mAVT5yRNfXGnuopKg%3D&X-Amz-Signature=314bae50580851720115d2fd8c729677dd5b2366893bbc9f16760f0fb0ffff1b&X-Amz-SignedHeaders=host)
* [Voxforge](http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german)
  * [Download Folder](http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/)
* [M-AILABS](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
  * [Download Folder](http://www.caito.de/data/Training/stt_tts/)
* [Spoken Wikipedia Corpora](http://nats.gitlab.io/swc/)
  * [Download Folder](https://corpora.uni-hamburg.de/hzsk/de/islandora/object/spoken-corpus:swc-2.0#additional-files)

### Download the corpus

For the speech corpus download the script [`download_speech_corpus.py`](./pre-processing/download_speech_corpus.py) is used and 
it will take the above URLs. 

~~~
source python-environments/bin/activate
python pre-processing/download_speech_corpus.py --tuda --cv --swc --voxforge --mailabs
~~~

_NOTE: The python module of `audiomate` which is used for the download process is using some fixed URLs. 
These URLs can be outdated depending on last release of `audiomate`. 
`download_speech_corpus.py` is patching these URLS, but in case a new speech corpus is released 
the URLs in the script must be updated again or passed as arguments `<speech_corpus>_url`, e.g. `--tuda_url <url>`._

It is also possible to leave out some corpus and specify just a subset to download. The default target directory
is the current directory, i.e. project root but could be set with `--target_path`. The path used in this documentation must 
be adjusted then accordingly.

The tool will display after a while a download progress indicator.

The downloads will be found in the folder `tuda`, `mozilla`, `voxforge`, `swc` and `mailabs`. 
If a download failed the incomplete file(s) must be removed manually.

### Prepare the Audio Data

The output of this step is a CSV format with references to the audio and corresponding text split into train, 
dev and test data for the TensorFlow network.

Convert to UTF-8:

~~~
pre-processing/run_to_utf_8.sh
~~~

The [`pre-processing/prepare_data`](pre-processing/prepare_data.py) script imports the downloaded corpora. It expects the corpus option 
followed by the corpus directory followed by the destination directory. Look into the script source to find all possible corpus options.
It is possible to train a subset (i.e. by just specifying one of the options) or all of the speech corpora.

Execute the following commands in the root directory of this project after running:

~~~shell
source python-environments/bin/activate
~~~

_NOTE: If the filename or folders of the downloaded speech corpora are changing the paths used in this section must be updated, too._

Some Examples:

**1. _Tuda-De_**

```shell
python pre-processing/prepare_data.py --tuda tuda/german-speechdata-package-v3 german-speech-corpus/data_tuda
```

**2. _Mozilla_**

```shell
python pre-processing/prepare_data.py --cv mozilla/cv-corpus-6.1-2020-12-11/de german-speech-corpus/data_mozilla
```

**3. _Voxforge_**

```shell
python pre-processing/prepare_data.py --voxforge voxforge german-speech-corpus/data_voxforge
```

**4. _Voxforge + Tuda-De_**

```shell
python pre-processing/prepare_data.py --voxforge voxforge --tuda tuda/german-speechdata-package-v2 german-speech-corpus/data_tuda+voxforge
```

**5. _Voxforge + Tuda-De + Mozilla_**

```shell
python pre-processing/prepare_data.py --cv mozilla/cv-corpus-6.1-2020-12-11/de/clips --voxforge voxforge --tuda tuda/german-speechdata-package-v3 german-speech-corpus/data_tuda+voxforge+mozilla
```

## Language Model

We used [KenLM](https://github.com/kpu/kenlm.git) toolkit to train a 3-gram language model. It is Language Model inference code by [Kenneth Heafield](https://kheafield.com/) Consult the repository for any information regarding the compilation in case of errors.

- **Installation**

```shell
cd kenlm
mkdir -p build
cd build
cmake ..
make -j `nproc`
```

`nproc` defines the number of threads to use.

### Text Corpus

We used an open-source [German Speech Corpus](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz) released by [University of Hamburg](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html).

#### Download the data:

```shell
mkdir german-text-corpus
cd german-text-corpus
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz
gzip -d German_sentences_8mil_filtered_maryfied.txt.gz
```

#### Pre-process the data

Execute the following commands in the root directory of this project.

```shell
source python-environments/bin/activate
python pre-processing/prepare_vocab.py german-text-corpus/German_sentences_8mil_filtered_maryfied.txt german-text-corpus/clean_vocab.txt
```

#### Build the Language Model

This step will create a `trie` data structure to describe the used vocabulary for the speech recognition. 
This file is termed as "scorer" in the DeepSpeech terminology. 
The script [`create_language_model.sh`](create_language_model.sh) is used for this.

The script will use the compiled KenLM binaries and download a native DeepSpeechClient.
The native DeepSpeech client is using the "cpu" architecture. For a different architecture pass as additional parameter the architecture.
See the `--arch` parameter in `python DeepSpeech/util/taskcluster.py --help` for details.

The mandatory `top_k` parameter in the script is set to 500000 which corresponds to > 98.43 % of all words in the downloaded text corpus.

The `default_alpha` and `default_beta` parameters are taken from [Scorer documentation](https://deepspeech.readthedocs.io/en/v0.9.3/Scorer.html) 
and are also the defaults in the `DeepSpeech.py`. 
DeepSpeech is offering the `lm_optimizer.py` scripts to find better values (in case there would be different ones)._

~~~shell
./create_language_model.sh
~~~

_NOTE: A different scorer might be useful later for the efficient speech recognition in case of a special language domain. 
Then an own text corpus should be used, and the script source can be used as a template for this scorer and be executed separately._

## Training

_NOTE: The training can be accelerated by using CUDA. 
Follow the [GPU Installation Instructions](https://www.tensorflow.org/install/gpu) for TensorFlow. At the time of writing 
also on Ubuntu 20.04 the toolchain for Ubuntu 18.04 has to be used._ 

If CUDA is used the GPU package of TensorFlow has to be installed:

~~~
source python-environments/bin/activate
pip uninstall tensorflow
pip install tensorflow-gpu==1.15.4
~~~

```shell
nohup ./train_model.sh <prepared_speech_corpus> &
```

Example:

~~~
nohup ./train_model.sh german-speech-corpus/data_tuda+voxforge+mozilla &
~~~

_Parameters:_

* The default scorer is `german-text-corpus/kenlm.scorer`.
* The default alphabet is `data/alphabet.txt`.
* The default DeepSpeech processing data is stored under `deepspeech_processing`.
* The model is stored under the directory `models`. 
  
It is possible to modify these parameters. If a parameter should take the default value, pass `""`.

~~~shell
./train_model.sh <prepared_speech_corpus> <alphabet> <scorer> <processing_dir> <models_dir>
~~~

Example:

~~~shell
./train_model.sh german-speech-corpus/data_tuda+voxforge+mozilla "" "" my_processing_dir my_models
~~~

_NOTE: The [augmentation parameters](https://deepspeech.readthedocs.io/en/v0.9.3/TRAINING.html#augmentation) are not used 
and might be interesting for improving the results._

_NOTE: Increase the `export_model_version` option in `train_model.sh` when a new model is trained._

_NOTE: In case a `Not enough time for target transition sequence (required: 171, available: 0)` is thrown, 
the currently only known fix is to edit the file `DeepSpeech/training/deepspeech_training/train.py` and
add `, ignore_longer_outputs_than_inputs=True` to the call to `tfv1.nn.ctc_loss`._

### TFLite Model Generation

The TensorFLow Lite model suitable for resource restricted devices (embedded, Raspberry Pi 4, ...) can be generated with:

~~~shell
./export_tflite.sh
~~~

_Parameters:_

* The default alphabet is `data/alphabet.txt`.
* The default DeepSpeech processing data is taken from `deepspeech_processing`.
* The model is stored under the directory `models`.
  
It is possible to modify these parameters. If a parameter should take the default value, pass `""`.

~~~shell
./export_tflite.sh <alphabet> <processing_dir> <models_dir>
~~~

Example:

~~~shell
./export_tflite.sh "" my_processing_dir my_models
~~~

### Fine-Tuning

Fine-tuning is used when the alphabet is staying the same, since the alphabet stays the same when trying to add more 
German text corpora it can be used.

Consult the [Fine-Tuning DeepSpeech Documentation](https://deepspeech.readthedocs.io/en/v0.9.3/TRAINING.html#fine-tuning-same-alphabet)

~~~shell
nohup ./fine_tuning_model.sh <prepared_speech_corpus> &
~~~

Example:

~~~
nohup ./fine_tuning_model.sh german-speech-corpus/my_own_corpus &
~~~

_Parameters:_

~~~shell
./fine_tuning_model.sh <prepared_speech_corpus> <alphabet> <scorer> <processing_dir> <models_dir> <cuda_options>
~~~

* The default alphabet is `data/alphabet.txt`. 
* The default DeepSpeech processing data is saved into `deepspeech_processing`.
* The scorer is taken from `german-text-corpus/kenlm.scorer`.
* The model is saved under the directory `models`.
* CUDA options:
  * `-load_cudnn` is not passed by default. If you don’t have a CUDA compatible GPU and the training was executed on a GPU (which is the default) 
pass `--load_cudnn` to allow training on the CPU. _NOTE: This requires also the Python `tensorflow-gpu` dependency, although no GPU is available._
  * `--train_cudnn` is not passed by default. It must be passed if the existing model was trained on a CUDA GPU and it should be continued on a CUDA GPU.

Example:

~~~shell
./fine_tuning_model.sh german-speech-corpus/my_own_corpus "" "" "" "" "" "" --train_cudnn
~~~

### Transfer Learning

Transfer learning must be used if the alphabet is changed, e.g. when using English and adding the German letters.
Transfer learning is using an existing neural network, dropping one or more layers of this neural network and
recalculated the weights with the new training data.

Consult the [Transfer Learning DeepSpeech Documentation](https://deepspeech.readthedocs.io/en/v0.9.3/TRAINING.html#transfer-learning-new-alphabet)

### Example English to German

* The English DeepSpeech model is used as base model. The checkpoints from the [DeepSpeech English model](https://github.com/mozilla/DeepSpeech/releases) must be us. 
E.g. `deepspeech-0.9.3-checkpoint.tar.gz` must be downloaded and extracted to `/deepspeech-0.9.3-checkpoint`.
* For the German language the language model and scorer must be build (this was done already [above](#build-the-language-model)).
* Call [`transfer_model.sh`](transfer_model.sh):

```shell
nohup ./transfer_model.sh german-speech-corpus/data_tuda+voxforge+mozilla+swc+mailabs deepspeech-0.9.3-checkpoint &
```

_NOTE: The checkpoints should be from the same DeepSpeech version to perform transfer learning._

_Parameters:_

* The default alphabet is `data/alphabet.txt`. 
* The default DeepSpeech processing data is saved into `deepspeech_transfer_processing`.
* The scorer is taken from `german-text-corpus/kenlm.scorer`.
* The model is saved under the directory `transfer_models`.
* The drop layer parameter is 1. 
* CUDA options:
  * `-load_cudnn` is not passed by default. If you don’t have a CUDA compatible GPU and the training was executed on a GPU (which is the default) 
pass `--load_cudnn` to allow training on the CPU. _NOTE: This requires also the Python `tensorflow-gpu` dependency, although no GPU is available._
  * `--train_cudnn` is not passed by default. It must be passed if the existing model was trained on a CUDA GPU and it should be continued on a CUDA GPU.

It is possible to modify these parameters. If a parameter should take the default value, pass `""`.

```shell
nohup ./transfer_model.sh <prepared_speech_corpus> <load_checkpoint_directory> <alphabet> <scorer> <processing_dir> <models_dir> <drop_layers> <cuda_options> &
```

Example:

```shell
nohup ./transfer_model.sh german-speech-corpus/data_tuda+voxforge+mozilla+swc+mailabs deepspeech-0.9.3-checkpoint "" "" "" my_models 2 --train_cudnn &
```

### Hyper-Parameter Optimization

The learning rate can be tested with the script [hyperparameter_optimization.sh](hyperparameter_optimization.sh).

_NOTE: The drop out rate parameter is not tested in this script._

Execute it with:

```shell
nohup ./hyperparameter_optimization.sh <prepared_speech_corpus> &
```

## Results

Some WER (word error rate) results from our findings.

- Mozilla 79.7%
- Voxforge 72.1%
- Tuda-De 26.8%
- Tuda-De+Mozilla 57.3%
- Tuda-De+Voxforge 15.1%
- Tuda-De+Voxforge+Mozilla 21.5%

_NOTE: Refer our paper for more information._

## Trained Models

The DeepSpeech model can be directly re-trained on new datasets. The required dependencies are available at:

**1. _v0.5.0_**

This model is trained on DeepSpeech v0.5.0 with _**Mozilla_v3+Voxforge+Tuda-De**_ (please refer the paper for more details)
https://drive.google.com/drive/folders/1nG6xii2FP6PPqmcp4KtNVvUADXxEeakk?usp=sharing

https://drive.google.com/file/d/1VN1xPH0JQNKK6DiSVgyQ4STFyDY_rle3/view

**2. _v0.6.0_**

This model is trained on DeepSpeech v0.6.0 with _**Mozilla_v4+Voxforge+Tuda-De+MAILABS(454+57+184+233h=928h)**_

https://drive.google.com/drive/folders/1BKblYaSLnwwkvVOQTQ5roOeN0SuQm8qr?usp=sharing

**3. _v0.7.4_**

This model is trained on DeepSpeech v0.7.4 using pre-trained English model released by Mozilla _**English+Mozilla_v5+MAILABS+Tuda-De+Voxforge (1700+750+233+184+57h=2924h)**_

https://drive.google.com/drive/folders/1PFSIdmi4Ge8EB75cYh2nfYOXlCIgiMEL?usp=sharing

**4. _v0.9.0_**

This model is trained on DeepSpeech v0.9.0 using pre-trained English model released by Mozilla _**English+Mozilla_v5+SWC+MAILABS+Tuda-De+Voxforge (1700+750+248+233+184+57h=3172h)**_

Thanks to [Karsten Ohme](https://github.com/kaoh) for providing the TFLite model.

Link: https://drive.google.com/drive/folders/1L7ILB-TMmzL8IDYi_GW8YixAoYWjDMn1?usp=sharing

 _**Why being SHY to STAR the repository, if you use the resources? :D**_

## Acknowledgments
* [Prof. Dr.-Ing. Torsten Zesch](https://www.ltl.uni-due.de/team/torsten-zesch) - Co-Author
* [Dipl.-Ling. Andrea Horbach](https://www.ltl.uni-due.de/team/andrea-horbach)
* [Matthias](https://github.com/ynop/audiomate)


## References
If you use our findings/scripts in your academic work, please cite:
```
@inproceedings{agarwal-zesch-2019-german,
    author = "Aashish Agarwal and Torsten Zesch",
    title = "German End-to-end Speech Recognition based on DeepSpeech",
    booktitle = "Preliminary proceedings of the 15th Conference on Natural Language Processing (KONVENS 2019): Long Papers",
    year = "2019",
    address = "Erlangen, Germany",
    publisher = "German Society for Computational Linguistics \& Language Technology",
    pages = "111--119"
}
```
<!--  An open-access Arxiv preprint is available here: -->
