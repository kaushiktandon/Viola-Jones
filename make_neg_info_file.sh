#!/bin/bash

if [ -f nonface.txt ]; then
  rm nonface.txt
fi

for filename in $(pwd)/train/nonface/*
do
  echo ${filename} >> nonface.txt
done