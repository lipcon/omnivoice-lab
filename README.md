# OmniVoice Lab

A small web UI around [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) for
generating speech and inspecting what came out, packaged for a single NVIDIA GPU.

It exists to do two things the bundled `omnivoice-demo` does not:

- **measure every clip as it is generated** — peak level, speech level, and the
  longest *internal* pause, so generation artefacts are caught before they end
  up baked into a training set;
- **run a whole prompt list in one go**, taking the same `{id: text}` JSON that
  drives the NeMo dataset builder and writing one 24 kHz wav per id.

## Requirements

An NVIDIA GPU with the driver installed plus
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image installs the `cu128` PyTorch wheels, which bundle the CUDA runtime, so
no CUDA base image or local toolkit is needed — just a driver new enough for
CUDA 12.8. Check with `nvidia-smi`.

VRAM is not a constraint: the model is 0.6B parameters and upstream test it on a
4 GB card. A 24 GB board has room to spare.

## Run

```bash
cp .env.example .env     # optional
docker compose up --build
```

Then open <http://localhost:8000>.

The first start downloads the model from Hugging Face into `./data/hf`, which
takes a while; the page shows `loading model…` and polls until it is ready. That
directory is a volume, so rebuilding the image never re-downloads the weights.

Everything else also lands under `./data`: uploaded references in
`data/references/`, generated runs in `data/outputs/<run-id>/` alongside a
`manifest.json` holding the per-clip measurements.

## Using it

Upload a 3–10 s reference clip, optionally with its transcript (Whisper fills it
in otherwise), then generate. **Keep one reference for a whole dataset** —
changing it mid-run changes the speaker.

Two settings matter more than they look:

**Deterministic.** OmniVoice has no seed argument. `position_temperature`
defaults to 5.0, so repeated runs of the same text differ, and across hundreds
of clips the voice drifts. Zeroing both temperatures is the only reproducibility
control available, and for building a single-speaker dataset it should stay on.

**Peak-normalise.** Raw output sits well below the level of a typical recorded
corpus. Since NeMo's FastPitch and HiFi-GAN preprocessors both use
`normalize: null`, the absolute waveform level goes straight into the mel, and a
mismatch shows up as a constant offset across every bin. Normalising to
−0.1 dBFS matches the convention used by the recorded datasets.

**Redoing one clip.** A batch run is rarely uniformly good: one line in fifty
comes out with the wrong timbre or an odd pause. The ↻ beside a row generates
that row again, with the generation settings as they stand, so a single line can
be redone slower or with more guidance than the run it belongs to. It re-rolls
with the temperatures on — a deterministic retry would hand back the identical
clip — and the wav keeps its name, so the run's zip always carries the latest
take. The take it replaces goes to `_history/` inside the run, which the zip
leaves out. A run can be reopened later from the dropdown without generating
anything again.

## API

| | |
|---|---|
| `GET /api/health` | model status, device, VRAM |
| `GET POST /api/references` | list / upload reference clips |
| `DELETE /api/references/{name}` | remove one |
| `POST /api/generate` | one clip; wav body, measurements in the `X-Stats` header |
| `POST /api/batch` | a whole prompt list; writes a run directory |
| `GET /api/outputs` | the runs on disk |
| `GET /api/outputs/{run}/manifest` | reopen a finished run |
| `POST /api/outputs/{run}/{id}/regenerate` | redo one clip of it, in place |
| `GET /api/outputs/{run}/download` | the run as a zip |

```bash
curl -s -X POST localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Prova di sintesi vocale.","deterministic":true,"normalize_peak":true}' \
  -D- -o out.wav | grep -i x-stats
```

## Configuration

Set in `.env` or the environment:

| variable | default | |
|---|---|---|
| `PORT` | `8000` | |
| `GPU_ID` | `0` | which physical card to pass in, as `nvidia-smi` numbers it |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | |
| `OMNIVOICE_DEVICE` | `cuda:0` | see below |
| `OMNIVOICE_DTYPE` | `float16` | |
| `OMNIVOICE_LANGUAGE` | *(auto)* | language of the reference clips |
| `OMNIVOICE_ASR_DEVICE` | *(model device)* | e.g. `cpu`, to keep the GPU free |
| `HF_ENDPOINT` | `https://huggingface.co` | |

### Picking a GPU

`GPU_ID` selects the card, not `OMNIVOICE_DEVICE`. Only the selected card is
passed into the container, so inside it the card is always `cuda:0` — setting
`OMNIVOICE_DEVICE=cuda:5` fails there with no such device. To run on the sixth
card:

```
GPU_ID=5
OMNIVOICE_DEVICE=cuda:0
```

`GPU_ID=all` exposes every card, and only then does `OMNIVOICE_DEVICE=cuda:5`
address one.

### The reference transcript

Type it. A reference uploaded without one is transcribed by Whisper, and the
text it produces is what the model aligns the voice against — wrong text, worse
clone, nothing in the logs. `OMNIVOICE_LANGUAGE` at least stops auto-detection
guessing the language of a short clip.

The transcript is saved next to the clip the first time, so Whisper runs once
per reference. Without that cache OmniVoice re-transcribes inside every
`generate()` call: the same five-second file, once per clip, for a whole run.

## Licence

Apache-2.0, the same licence as [OmniVoice](https://github.com/k2-fsa/OmniVoice)
itself, so there is no friction in using both. The model weights carry their
own terms. Cloning a real person's voice needs that person's consent.
