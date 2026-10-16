# BlackHole baseline screen

Copies of the two laptop scripts the BlackHole threads use to screen candidate bases, so a
machine without the project's shared folder can fetch them:

- `muggebench.sh`: the eval thread's MuggeBench runner for one GGUF (score with 95% interval,
  plus the one-agent footprint at a 16K context). `CTX16=1` scores in the 16K mode.
- `footprint.sh`: the serving thread's footprint measurement (one slot, 16,384 context,
  q8_0 KV, flash attention, after a ~12K-token prompt; at the default `-ub` and at 128).

The shared-folder copies (`blackhole/eval/muggebench/`, `blackhole/serving/`) are the
originals; these were copied on 2026-10-16.
