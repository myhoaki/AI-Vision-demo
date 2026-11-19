"""Lazy, process-wide PaddleOCR inference adapter."""

from __future__ import annotations

import os
import math
import re
import threading
import time
from typing import Any

_lock = threading.Lock()
_engine: Any | None = None
_engine_config: tuple[str, str] | None = None
_engine_error: str | None = None


def _get_engine(language: str, model_name: str):
    global _engine, _engine_config, _engine_error
    config = (language, model_name)
    with _lock:
        if _engine is not None and _engine_config == config:
            return _engine
        if _engine_error and _engine_config == config:
            raise RuntimeError(_engine_error)
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR 2.x API, CPU-only. Detection and recognition models are
            # initialized once per process/configuration and cached by PaddleOCR.
            kwargs = {
                "lang": language,
                "use_angle_cls": False,
                "use_gpu": False,
                "show_log": False,
                "ocr_version": model_name,
            }
            if model_name:
                kwargs["det_model_dir"] = os.getenv("PADDLEOCR_DET_MODEL_DIR") or None
                kwargs["rec_model_dir"] = os.getenv("PADDLEOCR_REC_MODEL_DIR") or None
            _engine = PaddleOCR(**{key: value for key, value in kwargs.items() if value is not None})
            _engine_config = config
            _engine_error = None
            return _engine
        except Exception as error:
            _engine = None
            _engine_config = config
            _engine_error = str(error)[:500]
            raise RuntimeError(_engine_error) from error


def engine_status() -> dict:
    return {"available": _engine is not None, "error": _engine_error}


def recognize(image, *, language: str = "en", model_name: str = "PP-OCRv4") -> dict:
    """Run OCR on a PIL image and return text regions in image-local coordinates."""
    started = time.perf_counter()
    try:
        import numpy as np

        engine = _get_engine(language, model_name)
        rgb = image.convert("RGB")
        result = engine.ocr(np.asarray(rgb), cls=False)
        lines = result[0] if result and result[0] else []
        regions = []
        for row in lines:
            try:
                polygon, (text, score) = row
                regions.append({
                    "box": [[float(point[0]), float(point[1])] for point in polygon],
                    "text": str(text),
                    "confidence": float(score),
                })
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        return {"regions": regions, "elapsedMs": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as error:
        return {"regions": [], "elapsedMs": round((time.perf_counter() - started) * 1000, 1), "error": str(error)[:500]}


def _prepare(image, options):
    """Apply only explicitly enabled, conservative preprocessing."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    result = image
    if options.get("grayscale"):
        result = ImageOps.grayscale(result).convert("RGB")
    if options.get("contrast"):
        result = ImageEnhance.Contrast(result).enhance(1.35)
    if options.get("sharpen"):
        result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
    scale = min(2.0, max(1.0, float(options.get("upscale", 1))))
    if scale > 1 and min(result.size) < 700:
        result = result.resize((round(result.width * scale), round(result.height * scale)), Image.Resampling.LANCZOS)
    return result


def recognize_container(source_image, box, *, language="en", model_name="PP-OCRv4", confidence=0.25,
                        padding=0.04, roi_positions=(0.0, 0.5, 1.0), roi_width=0.15,
                        roi_height=0.75, roi_padding=0.02, rotations=(0, 90, 180, 270),
                        max_attempts=8, full_fallback=True, preprocessing=None):
    """Run three longitudinally positioned ROIs, then optionally the full crop."""
    from .container_number_validator import candidates
    preprocessing = preprocessing or {}
    W, H = source_image.size
    pad_x = round((box["x2"] - box["x1"]) * max(0, float(padding)))
    pad_y = round((box["y2"] - box["y1"]) * max(0, float(padding)))
    x1=max(0,math.floor(box["x1"])-pad_x); y1=max(0,math.floor(box["y1"])-pad_y)
    x2=min(W,math.ceil(box["x2"])+pad_x); y2=min(H,math.ceil(box["y2"])+pad_y)
    crop=source_image.crop((x1,y1,max(x1+1,x2),max(y1+1,y2)))
    horizontal=crop.width >= crop.height
    attempts=[]; started=time.perf_counter(); selected=None
    attempt_limit=max(3,min(32,int(max_attempts)))
    longitudinal = crop.width if horizontal else crop.height
    transverse = crop.height if horizontal else crop.width
    roi_width = min(0.75, max(0.05, float(roi_width)))
    roi_height = min(1.0, max(0.1, float(roi_height)))
    roi_padding = min(0.2, max(0.0, float(roi_padding)))
    labels=("start", "middle", "end")
    for label, center in zip(labels, roi_positions):
        center=min(1.0,max(0.0,float(center)))
        lo=max(0, round((center-roi_width/2-roi_padding)*longitudinal))
        hi=min(longitudinal, round((center+roi_width/2+roi_padding)*longitudinal))
        cross_lo=max(0, round((0.5-roi_height/2-roi_padding)*transverse))
        cross_hi=min(transverse, round((0.5+roi_height/2+roi_padding)*transverse))
        rect=(lo,cross_lo,hi,cross_hi) if horizontal else (cross_lo,lo,cross_hi,hi)
        roi=crop.crop(rect)
        # Normalize vertical container text to horizontal before OCR. Each of
        # the three primary regions is still evaluated independently.
        angle=0 if horizontal else 90
        if rotations and angle not in rotations:
            angle=int(rotations[0])%360
        oriented=roi.rotate(-angle,expand=True) if angle else roi
        if len(attempts)>=attempt_limit: break
        prepared=_prepare(oriented,preprocessing)
        result=recognize(prepared,language=language,model_name=model_name)
        # PaddleOCR reports coordinates in the image it received. Undo optional
        # preprocessing upscaling before mapping points back into the source crop.
        scale_x=oriented.width/prepared.width
        scale_y=oriented.height/prepared.height
        def map_point(point):
            px, py = point[0]*scale_x, point[1]*scale_y
            if angle == 90: px, py = py, roi.height - px
            elif angle == 180: px, py = roi.width - px, roi.height - py
            elif angle == 270: px, py = roi.width - py, px
            return [px + rect[0], py + rect[1]]
        regions=[dict(r, box=[map_point(p) for p in r["box"]]) for r in result["regions"] if r["confidence"]>=confidence]
        raw=" ".join(r["text"] for r in result["regions"])
        accepted=" ".join(r["text"] for r in result["regions"] if r["confidence"]>=confidence)
        numbers=candidates(accepted)
        attempt={"kind":"roi","label":label,"position":center,"width":roi_width,"height":roi_height,"rotation":angle,"rect":{"x1":rect[0] if horizontal else rect[1],"y1":rect[1] if horizontal else rect[0],"x2":rect[2] if horizontal else rect[3],"y2":rect[3] if horizontal else rect[2]},"regions":regions,"rawText":raw,"acceptedText":accepted,"candidates":numbers,"elapsedMs":result["elapsedMs"],"status":"recognized" if any(n["validCheckDigit"] for n in numbers) else ("verify" if numbers else "unrecognized"),"error":result.get("error")}
        attempts.append(attempt)
        if any(n["validCheckDigit"] for n in numbers): selected=selected or next(n["candidate"] for n in numbers if n["validCheckDigit"])
    if full_fallback and not selected:
        prepared=_prepare(crop,preprocessing)
        result=recognize(prepared,language=language,model_name=model_name)
        scale_x=crop.width/prepared.width; scale_y=crop.height/prepared.height
        regions=[dict(r,box=[[p[0]*scale_x,p[1]*scale_y] for p in r["box"]]) for r in result["regions"] if r["confidence"]>=confidence]
        raw=" ".join(r["text"] for r in result["regions"]); accepted=" ".join(r["text"] for r in result["regions"] if r["confidence"]>=confidence); numbers=candidates(accepted)
        attempts.append({"kind":"full","label":"full-container fallback","percentage":1,"rotation":0,"rect":{"x1":0,"y1":0,"x2":crop.width,"y2":crop.height},"regions":regions,"rawText":raw,"acceptedText":accepted,"candidates":numbers,"elapsedMs":result["elapsedMs"],"error":result.get("error")})
    all_candidates=[]
    for a in attempts:
        for c in a["candidates"]:
            entry={**c,"source":a["label"],"confidence":max((r["confidence"] for r in a["regions"]),default=0)}
            if entry["candidate"] not in [v["candidate"] for v in all_candidates]: all_candidates.append(entry)
    valid=[c for c in all_candidates if c["validCheckDigit"]]
    final=valid[0]["candidate"] if len({c["candidate"] for c in valid})==1 else None
    status="recognized" if final else ("verify" if all_candidates else "unrecognized")
    regions=[r for a in attempts for r in a["regions"]]
    raw_text=" | ".join(a["rawText"] for a in attempts if a["rawText"])
    return {"cropRect":{"x1":x1,"y1":y1,"x2":x2,"y2":y2},"cropSize":{"width":crop.width,"height":crop.height},"attempts":attempts,"candidates":all_candidates,"finalNumber":final,"status":status,"regions":regions,"rawText":raw_text,"recognizedText":final or "","elapsedMs":round((time.perf_counter()-started)*1000,1)}
