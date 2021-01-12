#!/usr/bin/env python3
import argparse
import logging
import multiprocessing

import audiomate.corpus.io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download data for training.')
    parser.add_argument('target_path', type=str, default='.')
    parser.add_argument('--tuda', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--tuda_url', type=str,
                        default='http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/'
                                'german-speechdata-package-v3.tar.gz')
    parser.add_argument('--voxforge', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--voxforge_url', type=str,
                        default='http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/')
    parser.add_argument('--swc', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--swc_url', type=str,
                        default='https://corpora.uni-hamburg.de/hzsk/de/islandora/object/'
                                'file:swc-2.0_de-with-audio/datastream/TAR/de-with-audio.tar')
    parser.add_argument('--mailabs', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--mailabs_url', type=str,
                        default='http://www.caito.de/data/Training/stt_tts/de_DE.tgz')
    parser.add_argument('--cv', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--cv_url', type=str,
                        default='https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/'
                                'cv-corpus-6.1-2020-12-11/de.tar.gz')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # monkey patching
    audiomate.corpus.io.common_voice.DOWNLOAD_URLS = {'de': args.cv_url}
    audiomate.corpus.io.mailabs.DOWNLOAD_URLS = {'de_DE': args.mailabs_url}
    audiomate.corpus.io.swc.URLS = {'de': args.swc_url}
    from audiomate.corpus.io import TudaDownloader, VoxforgeDownloader, CommonVoiceDownloader, SWCDownloader, \
        MailabsDownloader

    if args.tuda:
        dl = TudaDownloader(url=args.tuda_url, num_threads=multiprocessing.cpu_count())
        dl.download(args.target_path + '/tuda')
    if args.voxforge:
        dl = VoxforgeDownloader(lang='de', url=args.voxforge_url, num_workers=multiprocessing.cpu_count())
        dl.download(args.target_path + '/voxforge')
    if args.cv:
        dl = CommonVoiceDownloader(lang='de', num_threads=multiprocessing.cpu_count())
        dl.download(args.target_path + '/mozilla')
    if args.mailabs:
        dl = MailabsDownloader(tags=['de_DE'], num_threads=multiprocessing.cpu_count())
        dl.download(args.target_path + '/mailabs')
    if args.swc:
        dl = SWCDownloader(lang='de')
        dl.download(args.target_path + '/swc')
