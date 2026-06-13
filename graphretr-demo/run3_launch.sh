#!/bin/bash
# run3_launch.sh -- fully detached run-3 pipeline. Survives SSH/laptop close
# (launch with: setsid nohup bash run3_launch.sh > runs/run3_launch.log 2>&1 &)
#   1. wait for any in-flight reembed_openai.py to exit
#   2. if it did not reach DONE, resume it (idempotent) up to 3 times
#   3. switch configs/campaign.yaml to the openai_small embedder
#   4. optimize: 100 steps OR until claude CLI limits stop it gracefully
#   5. final: seed + best on the locked test split
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
PY=.venv/bin/python

echo "[launch] $(date) waiting for running reembed (if any) to exit"
while pgrep -f reembed_openai.py > /dev/null; do sleep 15; done

for i in 1 2 3; do
    grep -q "\[reembed\] DONE" runs/reembed.log && break
    echo "[launch] $(date) reembed incomplete -- resume attempt $i"
    "$PY" -u reembed_openai.py >> runs/reembed.log 2>&1
done
if ! grep -q "\[reembed\] DONE" runs/reembed.log; then
    echo "[launch] $(date) reembed still incomplete after 3 attempts -- ABORT"
    exit 1
fi
echo "[launch] $(date) reembed DONE"

sed -i 's/^embedder: minilm.*/embedder: openai_small/' configs/campaign.yaml
grep "^embedder:" configs/campaign.yaml

echo "[launch] $(date) starting run3 optimize (extract_first, 100 steps max)"
"$PY" -u -m graphretr_opt.cli optimize --steps 100 --campaign-name run3 \
    --strategy extract_first > runs/run3.log 2>&1
echo "[launch] $(date) optimize exited rc=$? -- running final test"

"$PY" -u -m graphretr_opt.cli final --campaign-name run3 \
    --strategy extract_first > runs/run3_final.log 2>&1
echo "[launch] $(date) final exited rc=$? -- pipeline complete"
