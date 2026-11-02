#!/bin/bash
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg45_corridor/src
export SCENE_RUNNER=/home/ryosei_t/runs/alg45_corridor/run_trials_scene.py
TTTCARV_WORKERS=8 bash /home/ryosei_t/scripts/a45_arms.sh /home/ryosei_t/runs/alg45_corridor/${A45_OUT:-r1} ${A45_N:-100} 20260824 /home/ryosei_t/runs/alg45_corridor/${A45_ARMS:-arms.txt}
