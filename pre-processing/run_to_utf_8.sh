#!/bin/bash
# file name: to_utf8
# MAINTAINER: Aashish Agarwal

FILES="voxforge/*/etc/prompts-original"
#FILES="voxforge/*/etc/PROMPTS"

for f in $FILES
do
  #filename="${f%.*}"
  #echo -n "$f"
  #file -I $f
  encoding=$(file -i "$f" | sed "s/.*charset=\(.*\)$/\1/")
  #if file -I $f | grep -wq "iso-8859-1"
  if [  "${encoding}" = "iso-8859-1" ]
  then
    #mkdir -p converted
    #cp $f ./converted
    iconv -f ISO-8859-1 -t UTF-8 $f > "${f}_utf8.tex"
    mv "${f}_utf8.tex" $f
    echo ": CONVERTED TO UTF-8."
  else
    echo ": UTF-8 ALREADY."
  fi
done
