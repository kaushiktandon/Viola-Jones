#!/bin/bash

if [ -f face.info ]; then
  rm face.info
fi

for i in {1..2429}
do
  j=${i}
  printf -v filepath "$(pwd)/train/face/face%05d.png 1 0 0 24 24\n" ${j}
  echo ${filepath} >> face.info
done