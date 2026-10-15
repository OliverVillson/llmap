"""The one sampling default for every model call the train harness makes outside the engine's
own bench runs (the teacher's answers to repair requests here; scripts/bh_run.py's recording
proxy reads the same values): Qwen's card for thinking off, with top_k always finite (run 1's
top_k 0 was one of the three bugs that scored its agent 0%) and min_p 0 (llama-server's
default 0.05 makes MiniCPM5-2B repeat; vLLM's default is already 0)."""

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0

DEFAULTS = {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "min_p": MIN_P}
