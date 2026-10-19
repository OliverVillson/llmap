# centipede-server: the demo on two gaming laptops

This runs the same demo as `deploy/MAC.md`, but on two laptops with NVIDIA GPUs. You reach it from your Mac with `ssh centipede-server`.

- **The Ubuntu laptop is the head (`centipede-server`).** It runs its own llama-server, the engine, the demo and the live view.
- **The Windows laptop is a helper.** Its llama-server only adds worker slots.

Each GPU serves its own copy of Qwen3.5-9B, and the engine spreads the agents across both. Splitting one model across the two GPUs over the USB-C cable would be slower. The cable carries only the HTTP calls, so a plain network connection works too.

`deploy/box.sh` is `deploy/mac.sh`. On Linux with an NVIDIA GPU it uses llama.cpp's CUDA release build and sizes the crew to the GPU's free memory.

## 1. The Ubuntu laptop

Run this on its console, or over ssh:

```sh
sudo apt-get update && sudo apt-get install -y git curl
git clone --depth 1 -b claude/mac-demo https://github.com/OliverVillson/llmap.git ~/llmap-demo || git -C ~/llmap-demo pull --ff-only
cd ~/llmap-demo/mugge && deploy/box.sh setup && deploy/box.sh download && deploy/box.sh start && deploy/box.sh wait && deploy/box.sh check && deploy/box.sh ssh-config
```

The last command prints a few lines starting `Host centipede-server`. Paste them into `~/.ssh/config` on your Mac.

If `setup` stops at the NVIDIA driver, run `sudo ubuntu-drivers install && sudo reboot`, then run the block again.

## 2. The Windows laptop

Run this in PowerShell:

```powershell
irm https://raw.githubusercontent.com/OliverVillson/llmap/claude/mac-demo/mugge/deploy/helper.ps1 -OutFile $HOME\helper.ps1; powershell -ExecutionPolicy Bypass -File $HOME\helper.ps1
```

- The first run downloads llama.cpp and the model (5.7 GB). After that it serves straight away.
- It prints this PC's addresses.
- If Windows asks whether llama-server may use the network, allow it.
- Keep the window open.

## 3. Connect them, then run

From your Mac:

```sh
ssh centipede-server
cd ~/llmap-demo/mugge && deploy/box.sh helper add http://WINDOWS-ADDRESS:8080
set -a && . deploy/mac.env && set +a && nohup bun demo/run.ts > demo.log 2>&1 &
sleep 3; tail -f demo.log
```

Replace `WINDOWS-ADDRESS` with the Wi-Fi address the helper window printed, such as `192.168.1.23` (`ipconfig` shows it too). Start the demo only after `helper add` has reported the helper's slots; a demo started before that runs on the head alone.

While that ssh session is open, the live view is at http://127.0.0.1:7341/ on your Mac. The `LocalForward` line in the ssh config carries it.

`deploy/box.sh status` shows the GPU, the calls running, and whether each helper answers. If a helper stops answering, the next demo run goes ahead without it.

## The crew on two 8 GB cards

These figures come from `deploy/box.sh crew`. It runs the same arithmetic as `mac_crew.py`, with q4_0 KV and 12,288 tokens a slot by default.

| Model | Head (Ubuntu) | Helper (Windows, ~0.5 GB used by the display) | Agents |
|---|---|---|---|
| Qwen3.5-9B Q4_K_M | 4 workers + 2 leads + 1 orchestrator | 4 workers | 10 |
| Qwen3.5-4B Q6_K (`MODEL=4b`, and `-Model 4b` on Windows) | 16 workers + 2 leads + 1 orchestrator | 16 workers | 40 |

The per-slot memory is 158 MiB for the 9B: 12,288 tokens × 9 KiB of q4_0 KV, plus 50 MiB of recurrent state. The fixed costs are:
- the weights
- 1.3 GB for the CUDA context and compute (an estimate)
- a 0.25 GB margin

**Speed** is an estimate, not yet measured. For the 9B, expect about 50 tok/s for one call on a laptop RTX 3070 and about 35 on a 4060, rising to roughly 3× that across a few calls. Either GPU is several times the M5 Air.

## If something fails

| Symptom | Fix |
|---|---|
| `start` dies with out of memory | `deploy/box.sh stop; CREW_WORKERS=2 deploy/box.sh start && deploy/box.sh wait` (the compute size is an estimate) |
| `helper add` says it does not answer | Check the address. Is the helper window still open? Does Windows Firewall allow llama-server? |
| calls cut (`finish_reason: length`) often | `KV=q8_0 CTX=16384` gives longer slots but fewer of them |

Paste failures in the project chat.
