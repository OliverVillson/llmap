"""The Centipede train harness (BlackHole's BH<version> worker): scripts/bh_run.py runs it.

Data path (data.py and the modules it uses): kits from bhdata become MuggeBench-format
projects, the teacher works them through the engine's own loop, and harvest.py turns the
passing calls into training rows that are byte-identical to what the runtime harness sends,
because every row is rendered by the engine itself (render.py → mugge/bench/render.ts).
"""
