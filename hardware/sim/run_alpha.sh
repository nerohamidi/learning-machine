#!/bin/bash
# +alpha regression: every testbench x every stimulus x zero gating off/on. Logs in alpha_logs/.
# Usage (from hardware/sim, conda env hdl on PATH):  ./run_alpha.sh [quick]
set -e
export PATH=~/miniconda3/envs/hdl/bin:$PATH
mkdir -p alpha_logs
iverilog -g2005 -o compiled       -c filelist
iverilog -g2005 -o compiled_tiled -c filelist_tiled
iverilog -g2005 -o compiled_p2    -c filelist_p2
iverilog -g2005 -o compiled_os    -c filelist_os
iverilog -g2005 -o compiled_attn  -c filelist_attn
run() { local name=$1; shift; "$@" > alpha_logs/$name.log 2>&1; printf "%-28s %s\n" "$name" "$(grep -h 'RESULT\|No error detected' alpha_logs/$name.log | head -1)"; }
run part1_core_tb vvp compiled
for z in 0 1; do
  for L in 27 24 29; do
    run ws_f${L}_fast_zg$z vvp compiled_tiled +dir=stim_f$L +fast=1 +zg=$z
    run os_f${L}_zg$z      vvp compiled_os +dir=stim_os_f$L +zg=$z
  done
  run os_rand_zg$z vvp compiled_os +dir=stim_os_rand +zg=$z
  for d in stim_attn_*; do run ${d#stim_}_zg$z vvp compiled_attn +dir=$d +zg=$z; done
done
if [ "$1" != "quick" ]; then
  for L in 27 24 29; do run ws_f${L}_template vvp compiled_tiled +dir=stim_f$L +fast=0; done
  for z in 0 1; do run p2_diff_zg$z vvp compiled_p2 +mode=diff +zg=$z; done &
  run p2_conv vvp compiled_p2 +mode=conv
  wait
fi
