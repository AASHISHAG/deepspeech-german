#!/bin/bash
# file name: to_utf8

FILES="/home/LTLab.lan/agarwal/german-speech-corpus/voxforge/*/etc/prompts-original"
#FILES="/home/LTLab.lan/agarwal/german-speech-corpus/voxforge/*/etc/PROMPTS"

for f in $FILES
do
  encoding=$(file -i "$f" | sed "s/.*charset=\(.*\)$/\1/")
  if [  "${encoding}" = "iso-8859-1" ]
  then
    iconv -f ISO-8859-1 -t UTF-8 $f > "${f}_utf8.tex"
    mv "${f}_utf8.tex" $f
    echo ": CONVERTED TO UTF-8."
  else
    echo ": UTF-8 ALREADY."
  fi
done
