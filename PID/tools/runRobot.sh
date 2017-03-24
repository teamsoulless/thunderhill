#!/bin/sh
./startTrainingThunderhill.sh
sleep 1
../build/pid $1 $2 $3 > robotout.txt
sleep 1
./endTraining.sh
tail -1 robotout.txt | awk '{print $2}'
