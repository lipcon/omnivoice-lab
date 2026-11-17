"""OmniVoice playground: a small FastAPI wrapper around k2-fsa/OmniVoice.

Two things this adds over the bundled `omnivoice-demo`:

  * every generation is measured (peak, speech level, longest internal pause)
    so artefacts are caught as they are produced rather than after a dataset
    has already been built;
  * a batch mode that takes the same {id: text} JSON used to drive the NeMo
    dataset builder, and writes one wav per id.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

SAMPLE_RATE = 24_000

DATA_DIR = Path(os.environ.get("OMNIVOICE_DATA", "/data"))
REF_DIR = DATA_DIR / "references"
OUT_DIR = DATA_DIR / "outputs"
STATIC_DIR = Path(__file__).parent / "static"

MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
DTYPE = os.environ.get("OMNIVOICE_DTYPE", "float16")

# Whisper transcribes a reference clip that was uploaded without its text.
# Left empty it auto-detects, which on a short or atypical clip can pick the
# wrong language and hand the model a nonsense ref_text -- with nothing in the
# logs to say so. Setting it is the cheapest way to remove that failure mode.
LANGUAGE = os.environ.get("OMNIVOICE_LANGUAGE", "").strip()
# The ASR model can sit on a different GPU, or on the CPU, to keep the
# generation device free. Empty means "same device as the model".
ASR_DEVICE = os.environ.get("OMNIVOICE_ASR_DEVICE", "").strip()

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

app = FastAPI(title="OmniVoice Lab")

# The model is loaded once in the background: from_pretrained downloads weights
# on first run, which must not block the page from rendering.
_model: Any = None
_state: dict[str, Any] = {"status": "loading", "error": None, "started": time.time()}
_lock = threading.Lock()


def _load_model() -> None:
    global _model
    try:
        import torch
        from omnivoice import OmniVoice

        dtype = getattr(torch, DTYPE)
        kwargs: dict[str, Any] = {"device_map": DEVICE, "dtype": dtype}
        if ASR_DEVICE:
            kwargs["asr_device"] = ASR_DEVICE
        model = OmniVoice.from_pretrained(MODEL_ID, **kwargs)
        with _lock:
            _model = model
        _state["status"] = "ready"
    except Exception as exc:  # surfaced in the UI instead of killing the server
        _state["status"] = "error"
        _state["error"] = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def _startup() -> None:
    for d in (REF_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_load_model, daemon=True).start()


def transcribe_reference(model: Any, path: Path, language: str = "") -> str:
    """Transcribe a reference clip once, with an explicit language if given.

    OmniVoice.transcribe() takes no language argument, so the HF pipeline is
    driven directly to pass one. If that private attribute ever goes away the
    public method still works, just with auto-detection.
    """
    lang = (language or LANGUAGE).strip()
    if getattr(model, "_asr_pipe", None) is None:
        model.load_asr_model()
    pipe = getattr(model, "_asr_pipe", None)
    if pipe is not None and lang:
        out = pipe(str(path), generate_kwargs={"language": lang, "task": "transcribe"})
        return (out["text"] or "").strip()
    return model.transcribe(str(path))


def require_model() -> Any:
    if _state["status"] == "error":
        raise HTTPException(503, f"Model failed to load: {_state['error']}")
    if _model is None:
        raise HTTPException(503, "Model is still loading, try again in a moment.")
    return _model


# ---------------------------------------------------------------- measurement

def analyse(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, Any]:
    """Level and pause statistics for one clip.

    `speech_rms_db` ignores silent frames, so it reflects the voice itself
    rather than the ratio of speech to pauses. `longest_pause_s` is the longest
    INTERNAL silence -- the failure mode where a generator drops a second of
    nothing into the middle of a sentence.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    out: dict[str, Any] = {
        "duration_s": round(x.size / sr, 3),
        "peak_dbfs": None,
        "speech_rms_db": None,
        "lead_silence_s": 0.0,
        "tail_silence_s": 0.0,
        "longest_pause_s": 0.0,
        "longest_pause_at_s": None,
        "warnings": [],
    }
    if x.size == 0:
        out["warnings"].append("empty audio")
        return out

    peak = float(np.max(np.abs(x)))
    out["peak_dbfs"] = round(20 * np.log10(max(peak, 1e-9)), 2)

    hop = max(int(0.02 * sr), 1)
    n = x.size // hop
    if n == 0:
        return out
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(np.maximum(rms, 1e-9))

    # Relative threshold, like librosa.effects.trim: level-independent. The
    # absolute floor stops an all-silent clip from reading as all speech, since
    # a relative threshold alone has nothing to be relative to.
    voiced = (db > (db.max() - 45.0)) & (db > -70.0)
    if not voiced.any():
        out["warnings"].append("no speech detected")
        return out

    out["speech_rms_db"] = round(float(20 * np.log10(max(rms[voiced].mean(), 1e-9))), 2)

    idx = np.flatnonzero(voiced)
    first, last = int(idx[0]), int(idx[-1])
    out["lead_silence_s"] = round(first * hop / sr, 3)
    out["tail_silence_s"] = round((n - 1 - last) * hop / sr, 3)

    best_len, best_at, run = 0, None, 0
    for i in range(first, last + 1):
        if voiced[i]:
            if run > best_len:
                best_len, best_at = run, i - run
            run = 0
        else:
            run += 1
    if run > best_len:
        best_len, best_at = run, last + 1 - run

    out["longest_pause_s"] = round(best_len * hop / sr, 3)
    out["longest_pause_at_s"] = round(best_at * hop / sr, 3) if best_at is not None else None

    if out["longest_pause_s"] >= 0.7:
        out["warnings"].append(
            f"{out['longest_pause_s']:.2f}s pause inside the sentence "
            f"at {out['longest_pause_at_s']:.2f}s"
        )
    if peak >= 0.999:
        out["warnings"].append("clipped")
    if out["peak_dbfs"] is not None and out["peak_dbfs"] < -20:
        out["warnings"].append("very quiet")
    return out


def peak_normalise(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x
    return (x * (10 ** (target_dbfs / 20.0) / peak)).astype(np.float32)


def generation_kwargs(p: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "num_step": int(p.get("num_step", 32)),
        "guidance_scale": float(p.get("guidance_scale", 2.0)),
        "preprocess_prompt": bool(p.get("preprocess_prompt", True)),
        "postprocess_output": bool(p.get("postprocess_output", True)),
    }
    if p.get("deterministic"):
        # position_temperature defaults to 5.0, which makes every run differ.
        # Zeroing both temperatures is the only reproducibility control the
        # model exposes -- there is no seed argument.
        kw["position_temperature"] = 0.0
        kw["class_temperature"] = 0.0
    if p.get("speed"):
        kw["speed"] = float(p["speed"])
    if p.get("duration"):
        kw["duration"] = float(p["duration"])
    return kw


# A VoiceClonePrompt is the encoded reference. Building it once and reusing it
# is what the maintainers recommend for keeping one voice across a run (issue
# #44); it also drops the per-call reference encoding. Keyed by the mtimes of
# the wav and its transcript, so correcting either rebuilds it.
_prompt_cache: dict[str, tuple[tuple[float, float], Any]] = {}


def clone_prompt(model: Any, ref_path: Path, ref_text: str) -> Any:
    meta = ref_path.with_suffix(".txt")
    key = (ref_path.stat().st_mtime, meta.stat().st_mtime if meta.exists() else 0.0)
    hit = _prompt_cache.get(ref_path.name)
    if hit and hit[0] == key:
        return hit[1]
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_path), ref_text=ref_text or None
    )
    _prompt_cache[ref_path.name] = (key, prompt)
    return prompt


def synthesise(text: str, p: dict[str, Any]) -> np.ndarray:
    model = require_model()
    kw = generation_kwargs(p)

    # generate() takes a language, though the parameter docs do not list it.
    # Leaving it out makes a 646-language model infer the language as well as
    # the speaker, which is one degree of freedom more than it needs.
    lang = (p.get("language") or LANGUAGE).strip()
    if lang:
        kw["language"] = lang
    # instruct alongside a reference: consistent with it, the maintainers say
    # it stabilises the described attributes; conflicting, the reference wins.
    if (p.get("instruct") or "").strip():
        kw["instruct"] = p["instruct"].strip()

    ref_name = (p.get("reference") or "").strip()
    if ref_name:
        ref_path = REF_DIR / SAFE_NAME.sub("_", ref_name)
        if not ref_path.exists():
            raise HTTPException(404, f"Reference not found: {ref_name}")
        meta = ref_path.with_suffix(".txt")
        ref_text = (p.get("ref_text") or "").strip()
        if not ref_text and meta.exists():
            ref_text = meta.read_text(encoding="utf-8").strip()
        try:
            kw["voice_clone_prompt"] = clone_prompt(model, ref_path, ref_text)
        except Exception:
            # older builds without create_voice_clone_prompt
            kw["ref_audio"] = str(ref_path)
            if ref_text:
                kw["ref_text"] = ref_text

    audio = model.generate(text=text, **kw)
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    x = np.asarray(audio, dtype=np.float32).reshape(-1)

    if p.get("normalize_peak"):
        x = peak_normalise(x, float(p.get("target_dbfs", -0.1)))
    return x


# ----------------------------------------------------------------- endpoints

@app.get("/api/health")
def health() -> JSONResponse:
    info: dict[str, Any] = {
        "status": _state["status"],
        "error": _state["error"],
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": DTYPE,
        "sample_rate": SAMPLE_RATE,
        "language": LANGUAGE or "auto",
        "asr_device": ASR_DEVICE or DEVICE,
        "asr_error": _state.get("asr_error"),
        "elapsed_s": round(time.time() - _state["started"], 1),
    }
    try:
        import torch

        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(i)
            info["gpu"] = torch.cuda.get_device_name(i)
            info["vram_free_gb"] = round(free / 2**30, 1)
            info["vram_total_gb"] = round(total / 2**30, 1)
    except Exception:
        pass
    return JSONResponse(info)


@app.get("/api/references")
def list_references() -> JSONResponse:
    items = []
    for f in sorted(REF_DIR.glob("*.wav")):
        meta = f.with_suffix(".txt")
        try:
            info = sf.info(str(f))
            duration, sr = round(info.duration, 2), info.samplerate
        except Exception:
            duration, sr = None, None
        items.append({
            "name": f.name,
            "duration_s": duration,
            "sample_rate": sr,
            "ref_text": meta.read_text(encoding="utf-8").strip() if meta.exists() else "",
        })
    return JSONResponse({"references": items})


@app.post("/api/references")
async def upload_reference(
    file: UploadFile = File(...),
    ref_text: str = Form(""),
    language: str = Form(""),
) -> JSONResponse:
    name = SAFE_NAME.sub("_", Path(file.filename or "reference").stem) + ".wav"
    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(400, f"Could not read audio: {exc}")

    audio = audio.mean(axis=1)  # the model takes a single channel
    dest = REF_DIR / name
    sf.write(str(dest), audio, sr, subtype="PCM_16")
    if ref_text.strip():
        dest.with_suffix(".txt").write_text(ref_text.strip(), encoding="utf-8")

    # Without a cached transcript, OmniVoice re-runs Whisper on this same clip
    # inside every single generate() call (models/omnivoice.py: "if ref_text is
    # None"). Transcribing once here and saving it turns 561 transcriptions of
    # one five-second file into one, and lets the text be corrected by hand.
    transcribed = False
    if not ref_text.strip():
        try:
            auto = transcribe_reference(require_model(), dest, language)
        except HTTPException:
            auto = ""  # model still loading; generate() will transcribe instead
        except Exception as exc:
            auto = ""
            _state["asr_error"] = f"{type(exc).__name__}: {exc}"
        if auto:
            dest.with_suffix(".txt").write_text(auto, encoding="utf-8")
            ref_text, transcribed = auto, True

    stats = analyse(audio, sr)
    if not 3.0 <= stats["duration_s"] <= 10.0:
        stats["warnings"].append(
            f"{stats['duration_s']:.1f}s reference; the model is tuned for 3-10s"
        )
    if transcribed:
        stats["warnings"].append(
            "Transcript written by Whisper -- check it before generating a set"
        )
    return JSONResponse({
        "name": name,
        "sample_rate": sr,
        "stats": stats,
        "ref_text": ref_text.strip(),
        "transcribed": transcribed,
        "language": (language or LANGUAGE) or "auto",
    })


@app.put("/api/references/{name}/text")
def set_reference_text(name: str, payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """Correct a transcript -- in particular one Whisper got wrong."""
    path = REF_DIR / SAFE_NAME.sub("_", name)
    if not path.exists():
        raise HTTPException(404, f"Reference not found: {name}")
    text = (payload.get("ref_text") or "").strip()
    meta = path.with_suffix(".txt")
    if text:
        meta.write_text(text, encoding="utf-8")
    elif meta.exists():
        meta.unlink()
    _prompt_cache.pop(path.name, None)
    return JSONResponse({"name": path.name, "ref_text": text})


@app.delete("/api/references/{name}")
def delete_reference(name: str) -> JSONResponse:
    path = REF_DIR / SAFE_NAME.sub("_", name)
    if path.exists():
        path.unlink()
    meta = path.with_suffix(".txt")
    if meta.exists():
        meta.unlink()
    return JSONResponse({"deleted": path.name})


@app.post("/api/generate")
def generate(payload: dict[str, Any] = Body(...)) -> Response:
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")

    started = time.time()
    audio = synthesise(text, payload)
    elapsed = time.time() - started

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")

    stats = analyse(audio)
    stats["generation_s"] = round(elapsed, 2)
    stats["rtf"] = round(elapsed / stats["duration_s"], 3) if stats["duration_s"] else None

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Stats": json.dumps(stats)},
    )


@app.post("/api/batch")
def batch(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """Generate many clips from {id: text} or [{id, text}, ...].

    Accepts the same JSON that drives the NeMo dataset builder, so a run can be
    pointed straight at an existing prompt list.
    """
    items_in = payload.get("items")
    if isinstance(items_in, dict):
        items = [{"id": k, "text": v} for k, v in items_in.items()]
    elif isinstance(items_in, list):
        items = [{"id": str(i.get("id")), "text": i.get("text")} for i in items_in]
    else:
        raise HTTPException(400, "items must be an object or a list")

    items = [i for i in items if (i.get("text") or "").strip()]
    if not items:
        raise HTTPException(400, "no usable items")

    limit = int(payload.get("limit") or 0)
    if limit:
        items = items[:limit]

    run_id = payload.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_DIR / SAFE_NAME.sub("_", str(run_id))
    run_dir.mkdir(parents=True, exist_ok=True)

    results, failures = [], 0
    for item in items:
        key = SAFE_NAME.sub("_", str(item["id"]))
        try:
            started = time.time()
            audio = synthesise(item["text"].strip(), payload)
            elapsed = time.time() - started
            path = run_dir / f"{key}.wav"
            sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")

            stats = analyse(audio)
            stats["generation_s"] = round(elapsed, 2)
            results.append({
                "id": key, "text": item["text"], "file": path.name,
                "url": f"/api/outputs/{run_dir.name}/{path.name}", **stats,
            })
        except Exception as exc:
            failures += 1
            results.append({
                "id": key, "text": item["text"],
                "error": f"{type(exc).__name__}: {exc}", "warnings": ["failed"],
            })

    (run_dir / "manifest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flagged = sum(1 for r in results if r.get("warnings"))
    return JSONResponse({
        "run_id": run_dir.name, "count": len(results),
        "failures": failures, "flagged": flagged, "results": results,
    })


@app.get("/api/outputs")
def list_runs() -> JSONResponse:
    runs = []
    for d in sorted(OUT_DIR.iterdir(), reverse=True) if OUT_DIR.exists() else []:
        if d.is_dir():
            runs.append({"run_id": d.name, "count": len(list(d.glob("*.wav")))})
    return JSONResponse({"runs": runs})


# Declared BEFORE /{name}: routes match in declaration order, so the
# catch-all would swallow "download" as a filename and 404.
@app.get("/api/outputs/{run_id}/download")
def download_run(run_id: str) -> Response:
    run_dir = OUT_DIR / SAFE_NAME.sub("_", run_id)
    if not run_dir.is_dir():
        raise HTTPException(404, "run not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run_dir.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_dir.name}.zip"'},
    )


@app.get("/api/outputs/{run_id}/{name}")
def get_output(run_id: str, name: str) -> FileResponse:
    path = OUT_DIR / SAFE_NAME.sub("_", run_id) / SAFE_NAME.sub("_", name)
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(path), media_type="audio/wav")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
