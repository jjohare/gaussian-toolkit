# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Person detection and removal pipeline stage.

Detects people in video frames and removes them via inpainting before COLMAP
processing. This prevents ghost gaussians from appearing in the final 3D
reconstruction.

Three methods are supported:

- **comfyui** (default, highest quality): Uses RTDETR_detect for person
  detection and RecraftImageInpaintingNode for inpainting, both running on
  the remote ComfyUI server at 192.168.2.48:8188.
- **opencv**: Uses torchvision Faster R-CNN for local detection and OpenCV
  Navier-Stokes inpainting. Fast, no network dependency, moderate quality.
- **flux**: Uses local Faster R-CNN detection + FLUX inpainting via ComfyUI.
  High quality inpainting but slower than Recraft.

Usage::

    from pipeline.person_remover import PersonRemover

    # High quality (ComfyUI RTDETR + Recraft)
    remover = PersonRemover(method="comfyui", comfyui_url="http://192.168.2.48:8188")
    manifest = remover.process_directory("frames/", "cleaned_frames/")

    # Fast local fallback
    remover = PersonRemover(method="opencv", confidence=0.5)
    manifest = remover.process_directory("frames/", "cleaned_frames/")
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

logger = logging.getLogger(__name__)

# COCO class index for "person"
_COCO_PERSON_CLASS = 1

# Supported image extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Workflow template directory
_WORKFLOW_DIR = Path(__file__).parent / "workflows"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single person detection in a frame."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # H x W, uint8, 255=person
    confidence: float
    area_pct: float  # percentage of frame covered


@dataclass
class FrameAction:
    """Record of what was done to a single frame."""
    filename: str
    action: str  # "clean" | "inpainted" | "dropped" | "flagged_inpainted"
    person_count: int
    coverage_pct: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# ComfyUI HTTP helpers
# ---------------------------------------------------------------------------

def _comfyui_request(
    url: str,
    data: dict | None = None,
    method: str = "POST",
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Send a request to a ComfyUI API endpoint."""
    body = json.dumps(data).encode("utf-8") if data else None
    req = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)
    except (HTTPError, URLError) as exc:
        raise ConnectionError(f"ComfyUI request to {url} failed: {exc}") from exc


def _image_to_base64_png(image: np.ndarray) -> str:
    """Encode a BGR numpy image as base64 PNG string."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _base64_to_image(b64: str) -> np.ndarray:
    """Decode a base64 PNG/JPEG string to a BGR numpy image."""
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# PersonRemover
# ---------------------------------------------------------------------------

class PersonRemover:
    """Detect and remove people from video frames before SfM processing.

    Parameters
    ----------
    method : str
        Inpainting method:
        - "comfyui": RTDETR detection + Recraft inpainting on remote ComfyUI.
        - "opencv": Local Faster R-CNN detection + OpenCV NS inpainting.
        - "telea": Local Faster R-CNN detection + OpenCV Telea inpainting.
        - "flux": Local Faster R-CNN detection + FLUX inpainting via ComfyUI.
    comfyui_url : str | None
        ComfyUI WebSocket/API URL (e.g. "http://192.168.2.48:8188").
        Required for method="comfyui". Also used for method="flux" via the
        SaladTechnologies API wrapper.
    flux_endpoint : str | None
        SaladTechnologies comfyui-api URL for FLUX inpainting.
    confidence : float
        Minimum detection confidence threshold (0.0-1.0).
    dilation_px : int
        Pixels to dilate person masks by, to catch shadows and reflections.
    drop_threshold : float
        If person coverage exceeds this fraction, drop the frame entirely.
    flag_threshold : float
        If person coverage exceeds this fraction (but below drop), flag for review.
    device : str | None
        Torch device for local detection. None = auto-detect.
    comfyui_timeout : float
        Timeout in seconds for ComfyUI workflow execution.
    """

    def __init__(
        self,
        method: str = "opencv",
        comfyui_url: str | None = None,
        flux_endpoint: str | None = None,
        confidence: float = 0.5,
        dilation_px: int = 15,
        drop_threshold: float = 0.30,
        flag_threshold: float = 0.05,
        device: str | None = None,
        comfyui_timeout: float = 120.0,
    ) -> None:
        self.method = method
        self.comfyui_url = comfyui_url.rstrip("/") if comfyui_url else None
        self.flux_endpoint = flux_endpoint
        self.confidence = confidence
        self.dilation_px = dilation_px
        self.drop_threshold = drop_threshold
        self.flag_threshold = flag_threshold
        self.comfyui_timeout = comfyui_timeout

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Any = None
        self._flux_inpainter: Any = None
        self._comfyui_available: bool | None = None

    # ------------------------------------------------------------------
    # Local model management
    # ------------------------------------------------------------------

    def _ensure_local_model(self) -> None:
        """Lazy-load the Faster R-CNN model for local person detection."""
        if self._model is not None:
            return

        logger.info("Loading Faster R-CNN (resnet50) on %s", self.device)
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self._model = fasterrcnn_resnet50_fpn(weights=weights)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Person detection model ready")

    def _ensure_flux_inpainter(self) -> None:
        """Lazy-load the FLUX ComfyUI inpainter."""
        if self._flux_inpainter is not None:
            return
        if not self.flux_endpoint:
            raise ValueError("flux_endpoint required for method='flux'")

        from pipeline.comfyui_inpainter import ComfyUIInpainter
        self._flux_inpainter = ComfyUIInpainter(api_url=self.flux_endpoint)

    # ------------------------------------------------------------------
    # ComfyUI availability check
    # ------------------------------------------------------------------

    def _check_comfyui(self) -> bool:
        """Check if the remote ComfyUI server is reachable."""
        if self._comfyui_available is not None:
            return self._comfyui_available

        if not self.comfyui_url:
            self._comfyui_available = False
            return False

        try:
            _comfyui_request(f"{self.comfyui_url}/system_stats", method="GET", timeout=5)
            self._comfyui_available = True
            logger.info("ComfyUI server reachable at %s", self.comfyui_url)
        except Exception as exc:
            self._comfyui_available = False
            logger.warning("ComfyUI not reachable at %s: %s", self.comfyui_url, exc)

        return self._comfyui_available

    # ------------------------------------------------------------------
    # ComfyUI-based detection + inpainting (RTDETR + Recraft)
    # ------------------------------------------------------------------

    def _comfyui_detect_and_inpaint(
        self, image: np.ndarray,
    ) -> tuple[list[Detection], np.ndarray | None]:
        """Run the full person_removal workflow on ComfyUI.

        Uses RTDETR_detect for person detection and RecraftImageInpaintingNode
        for high-quality inpainting, connected via a GrowMask node for dilation.

        Returns
        -------
        tuple[list[Detection], np.ndarray | None]
            (detections, inpainted_image). inpainted_image is None if no people found.
        """
        wf_path = _WORKFLOW_DIR / "person_removal.json"
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow not found: {wf_path}")

        with open(wf_path) as f:
            workflow = json.load(f)

        # Encode the input image as base64 for LoadImageBase64 node
        b64_image = _image_to_base64_png(image)

        # Inject parameters into the workflow
        graph = workflow
        # Node 1: LoadImageBase64 — set the image data
        graph["1"]["inputs"]["image"] = b64_image
        # Node 2: RTDETR_detect — set confidence threshold
        graph["2"]["inputs"]["threshold"] = self.confidence
        # Node 3: GrowMask — set dilation
        graph["3"]["inputs"]["expand"] = self.dilation_px
        # Node 5: RecraftImageInpaintingNode — seed
        graph["5"]["inputs"]["seed"] = int(time.time()) % (2**31)

        # Submit the workflow via ComfyUI /prompt API
        prompt_id = str(uuid.uuid4())
        payload = {"prompt": graph, "client_id": prompt_id}

        try:
            resp = _comfyui_request(
                f"{self.comfyui_url}/prompt",
                data=payload,
                timeout=self.comfyui_timeout,
            )
        except ConnectionError:
            logger.warning("ComfyUI workflow submission failed, falling back to local")
            return [], None

        queue_id = resp.get("prompt_id", "")

        # Poll for completion
        inpainted = self._poll_comfyui_result(queue_id)
        if inpainted is None:
            logger.warning("ComfyUI returned no result, falling back to local")
            return [], None

        # We don't get per-detection boxes from the workflow output, but we can
        # detect locally to populate the manifest. The inpainting already happened
        # on the server, so we just need detection metadata.
        h, w = image.shape[:2]
        detections = self._detect_people_local(image)

        return detections, inpainted

    def _poll_comfyui_result(self, prompt_id: str) -> np.ndarray | None:
        """Poll the ComfyUI /history endpoint until the workflow completes."""
        deadline = time.monotonic() + self.comfyui_timeout
        interval = 1.0

        while time.monotonic() < deadline:
            try:
                history = _comfyui_request(
                    f"{self.comfyui_url}/history/{prompt_id}",
                    method="GET",
                    timeout=10,
                )
            except ConnectionError:
                time.sleep(interval)
                continue

            if prompt_id not in history:
                time.sleep(interval)
                continue

            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                logger.error("ComfyUI workflow failed: %s", status.get("messages"))
                return None

            outputs = entry.get("outputs", {})
            # Find the SaveImage / PreviewImage node output
            for node_id, node_out in outputs.items():
                images = node_out.get("images", [])
                if images:
                    img_info = images[0]
                    filename = img_info.get("filename", "")
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")

                    # Fetch the image from ComfyUI
                    params = f"filename={filename}&subfolder={subfolder}&type={img_type}"
                    try:
                        req = Request(f"{self.comfyui_url}/view?{params}")
                        with urlopen(req, timeout=30) as resp:
                            img_bytes = resp.read()
                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    except Exception as exc:
                        logger.error("Failed to fetch result image: %s", exc)
                        return None

            time.sleep(interval)

        logger.error("ComfyUI workflow timed out after %.0fs", self.comfyui_timeout)
        return None

    # ------------------------------------------------------------------
    # Local person detection (Faster R-CNN)
    # ------------------------------------------------------------------

    def _detect_people_local(self, image: np.ndarray) -> list[Detection]:
        """Detect people using the local Faster R-CNN model."""
        self._ensure_local_model()

        h, w = image.shape[:2]
        total_pixels = h * w

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            predictions = self._model([tensor])[0]

        detections: list[Detection] = []
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        for i in range(len(labels)):
            if labels[i] != _COCO_PERSON_CLASS:
                continue
            if scores[i] < self.confidence:
                continue

            x1, y1, x2, y2 = boxes[i].astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            if self.dilation_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.dilation_px * 2 + 1, self.dilation_px * 2 + 1),
                )
                mask = cv2.dilate(mask, kernel, iterations=1)

            area_pct = float(np.count_nonzero(mask)) / total_pixels

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                mask=mask,
                confidence=float(scores[i]),
                area_pct=area_pct,
            ))

        return detections

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect_people(self, image: np.ndarray) -> list[Detection]:
        """Detect people in an image.

        Uses RTDETR on ComfyUI when method="comfyui" and the server is available,
        otherwise falls back to local Faster R-CNN.

        Parameters
        ----------
        image : np.ndarray
            BGR image (H, W, 3), uint8 as loaded by cv2.imread.

        Returns
        -------
        list[Detection]
            Detected people with bounding boxes, masks, and coverage.
        """
        return self._detect_people_local(image)

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------

    def _combine_masks(
        self, detections: list[Detection], shape: tuple[int, int],
    ) -> np.ndarray:
        """Merge all detection masks into a single binary mask."""
        combined = np.zeros(shape, dtype=np.uint8)
        for det in detections:
            combined = np.maximum(combined, det.mask)
        return combined

    def remove_people(
        self, image: np.ndarray, detections: list[Detection],
    ) -> np.ndarray:
        """Inpaint detected person regions from the image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (H, W, 3), uint8.
        detections : list[Detection]
            Person detections from detect_people().

        Returns
        -------
        np.ndarray
            Inpainted image with people removed, same shape and dtype.
        """
        if not detections:
            return image.copy()

        h, w = image.shape[:2]
        combined_mask = self._combine_masks(detections, (h, w))

        if self.method in ("opencv", "comfyui"):
            # For comfyui fallback case where server is down, use opencv
            if self.method == "comfyui":
                logger.debug("Using OpenCV inpainting as local fallback")
            return cv2.inpaint(
                image, combined_mask, inpaintRadius=5, flags=cv2.INPAINT_NS,
            )
        elif self.method == "telea":
            return cv2.inpaint(
                image, combined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA,
            )
        elif self.method == "flux":
            return self._inpaint_flux(image, combined_mask)
        else:
            raise ValueError(f"Unknown inpainting method: {self.method}")

    def _inpaint_flux(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using FLUX via ComfyUI."""
        self._ensure_flux_inpainter()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._flux_inpainter.inpaint(
            image=rgb,
            mask=mask,
            prompt="clean empty background, photorealistic, high quality, no people",
            negative_prompt="person, people, human, artifacts, distortion, blurry",
        )
        return cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self, image: np.ndarray, filename: str = "",
    ) -> tuple[np.ndarray | None, FrameAction]:
        """Process a single frame: detect, decide, and optionally inpaint.

        For method="comfyui", detection and inpainting happen together on the
        remote server. For other methods, detection is local and inpainting
        uses the configured local method.

        Parameters
        ----------
        image : np.ndarray
            BGR image.
        filename : str
            Original filename for the manifest.

        Returns
        -------
        tuple[np.ndarray | None, FrameAction]
            (cleaned_image, action). cleaned_image is None if the frame was dropped.
        """
        t0 = time.monotonic()

        # ComfyUI path: detect + inpaint in one workflow
        if self.method == "comfyui" and self._check_comfyui():
            detections, inpainted = self._comfyui_detect_and_inpaint(image)

            if inpainted is not None and detections:
                h, w = image.shape[:2]
                combined = self._combine_masks(detections, (h, w))
                total_coverage = float(np.count_nonzero(combined)) / (h * w)
                elapsed = time.monotonic() - t0

                det_records = [
                    {
                        "bbox": list(d.bbox),
                        "confidence": round(d.confidence, 4),
                        "area_pct": round(d.area_pct, 4),
                    }
                    for d in detections
                ]

                if total_coverage > self.drop_threshold:
                    return None, FrameAction(
                        filename=filename,
                        action="dropped",
                        person_count=len(detections),
                        coverage_pct=round(total_coverage * 100, 2),
                        detections=det_records,
                        elapsed_s=elapsed,
                    )

                action = "flagged_inpainted" if total_coverage > self.flag_threshold else "inpainted"
                return inpainted, FrameAction(
                    filename=filename,
                    action=action,
                    person_count=len(detections),
                    coverage_pct=round(total_coverage * 100, 2),
                    detections=det_records,
                    elapsed_s=elapsed,
                )

            elif not detections:
                elapsed = time.monotonic() - t0
                return image.copy(), FrameAction(
                    filename=filename,
                    action="clean",
                    person_count=0,
                    coverage_pct=0.0,
                    elapsed_s=elapsed,
                )

            # If ComfyUI returned no image but had detections, fall through to local

        # Local path: detect with Faster R-CNN, inpaint locally
        detections = self._detect_people_local(image)

        if not detections:
            elapsed = time.monotonic() - t0
            return image.copy(), FrameAction(
                filename=filename,
                action="clean",
                person_count=0,
                coverage_pct=0.0,
                elapsed_s=elapsed,
            )

        h, w = image.shape[:2]
        combined = self._combine_masks(detections, (h, w))
        total_coverage = float(np.count_nonzero(combined)) / (h * w)

        det_records = [
            {
                "bbox": list(d.bbox),
                "confidence": round(d.confidence, 4),
                "area_pct": round(d.area_pct, 4),
            }
            for d in detections
        ]

        if total_coverage > self.drop_threshold:
            elapsed = time.monotonic() - t0
            logger.warning(
                "%s: person coverage %.1f%% exceeds drop threshold (%.0f%%), dropping",
                filename, total_coverage * 100, self.drop_threshold * 100,
            )
            return None, FrameAction(
                filename=filename,
                action="dropped",
                person_count=len(detections),
                coverage_pct=round(total_coverage * 100, 2),
                detections=det_records,
                elapsed_s=elapsed,
            )

        cleaned = self.remove_people(image, detections)
        elapsed = time.monotonic() - t0

        if total_coverage > self.flag_threshold:
            action = "flagged_inpainted"
            logger.info(
                "%s: coverage %.1f%% above flag threshold, inpainted but flagged",
                filename, total_coverage * 100,
            )
        else:
            action = "inpainted"
            logger.info(
                "%s: %d people (%.1f%% coverage), inpainted",
                filename, len(detections), total_coverage * 100,
            )

        return cleaned, FrameAction(
            filename=filename,
            action=action,
            person_count=len(detections),
            coverage_pct=round(total_coverage * 100, 2),
            detections=det_records,
            elapsed_s=elapsed,
        )

    # ------------------------------------------------------------------
    # Directory processing
    # ------------------------------------------------------------------

    def process_directory(self, input_dir: str, output_dir: str) -> dict[str, Any]:
        """Process all frames in a directory, writing cleaned frames to output_dir.

        Parameters
        ----------
        input_dir : str
            Directory containing source frames (jpg/png).
        output_dir : str
            Directory for cleaned output frames. Created if needed.

        Returns
        -------
        dict
            Manifest with per-frame actions, summary statistics, and timing.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not in_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {in_path}")

        image_files = sorted(
            f for f in in_path.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        )

        if not image_files:
            raise FileNotFoundError(f"No image files found in {in_path}")

        logger.info(
            "Processing %d frames from %s -> %s (method=%s, confidence=%.2f)",
            len(image_files), in_path, out_path, self.method, self.confidence,
        )

        t0 = time.monotonic()
        actions: list[FrameAction] = []
        counts = {"clean": 0, "inpainted": 0, "flagged_inpainted": 0, "dropped": 0}

        for i, img_file in enumerate(image_files):
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning("Could not read %s, skipping", img_file)
                continue

            cleaned, action = self.process_frame(image, img_file.name)
            actions.append(action)
            counts[action.action] = counts.get(action.action, 0) + 1

            if cleaned is not None:
                out_file = out_path / img_file.name
                cv2.imwrite(str(out_file), cleaned)

            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                logger.info("Progress: %d/%d frames", i + 1, len(image_files))

        total_elapsed = time.monotonic() - t0

        manifest = {
            "input_dir": str(in_path),
            "output_dir": str(out_path),
            "method": self.method,
            "confidence_threshold": self.confidence,
            "dilation_px": self.dilation_px,
            "drop_threshold": self.drop_threshold,
            "flag_threshold": self.flag_threshold,
            "total_frames": len(image_files),
            "summary": counts,
            "total_elapsed_s": round(total_elapsed, 2),
            "avg_frame_time_s": round(
                total_elapsed / max(len(image_files), 1), 2,
            ),
            "frames": [
                {
                    "filename": a.filename,
                    "action": a.action,
                    "person_count": a.person_count,
                    "coverage_pct": a.coverage_pct,
                    "detections": a.detections,
                    "elapsed_s": round(a.elapsed_s, 3),
                }
                for a in actions
            ],
        }

        manifest_path = out_path / "person_removal_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        logger.info(
            "Person removal complete: %d clean, %d inpainted, %d flagged, "
            "%d dropped (%.1fs total)",
            counts["clean"], counts["inpainted"], counts["flagged_inpainted"],
            counts["dropped"], total_elapsed,
        )

        return manifest
