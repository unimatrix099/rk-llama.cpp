#!/usr/bin/env bash
# Repeat the decode check; on a crash, keep the core and print a backtrace.
O=~/bench-logs/2026-10-05-stress-$1; mkdir -p $O; cd ~/rk-llama.cpp-rebase
for i in $(seq 1 $2); do
  rm -f core core.*
  ( ulimit -c unlimited; bash ~/rk-llama.cpp-rebase/docs/backend/rknpu2-autoresearch/decode-check.sh ) > $O/run-$i.txt 2>&1
  rc=$?
  echo "run $i rc=$rc $(tail -1 $O/run-$i.txt)" >> $O/summary.txt
  c=$(ls core* 2>/dev/null | head -1)
  if [ -n "$c" ]; then
    mv $c $O/core-$i
    gdb -batch -ex "thread apply all bt 12" build/bin/llama-server $O/core-$i > $O/bt-$i.txt 2>&1
  fi
done
echo DONE >> $O/summary.txt
