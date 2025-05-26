#!/bin/bash

for seed in 0
do
    SEED=${seed}
    export SEED
    for scene in 0
    do
        SCENE_NUM=${scene}
        export SCENE_NUM
        echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
        python3 -u scripts/splatamv1.3.2-time.py configs/tum/tum_eval.py | tee outputtgum3-time${SCENE_NUM}-${SEED}.txt
    done
done