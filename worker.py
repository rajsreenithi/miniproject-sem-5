"""AuraVision capture/inference/database worker; no Flask or rules engine."""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import threading
import time
from datetime import datetime
from typing import Any
from db.connection import get_connection
from rules.engine import RuleEngine
from rules.incidents import IncidentManager

import cv2
import numpy as np

from db.queries import (
    insert_crossings,
    insert_observations,
    insert_visits,
    load_camera_config,
)
from engine.annotate import FrameAnnotator
from engine.detector import PersonDetector
from engine.heatmap import HeatmapAccumulator
from engine.linecount import LineCounter
from engine.zones import ZoneMonitor


PROJECT_ROOT = Path(__file__).resolve().parent
LIVE_DIR = PROJECT_ROOT / "data" / "live"
HEATMAP_DIR = PROJECT_ROOT / "data" / "heatmaps"
LOG_DIR = PROJECT_ROOT / "logs"
OBSERVATION_INTERVAL_SEC = 10.0
LIVE_FRAME_INTERVAL_SEC = 0.5
HEATMAP_INTERVAL_SEC = 60.0
DB_BATCH_INTERVAL_SEC = 1.0
QUEUE_MAXSIZE = 10000

def rules_connection():
    """Create a database connection for rules and incident processing."""
    return get_connection(attempts=3, delay_sec=1.0)
def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    file_handler = RotatingFileHandler(
        LOG_DIR / "worker.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)


def parse_source(camera: dict[str, Any]) -> int | str:
    source_type = camera["source_type"]
    uri = str(camera["source_uri"])
    if source_type == "webcam":
        try:
            return int(uri)
        except ValueError:
            return uri
    return uri


def capture_loop(
    source: int | str,
    source_type: str,
    latest: dict[str, Any],
    lock: threading.Lock,
    stop_event: threading.Event,
    loop_file: bool,
) -> None:
    """Continuously capture, retaining only the newest frame."""
    backoff = 1.0
    while not stop_event.is_set():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            if source_type == "rtsp":
                logging.warning("RTSP open failed; retrying in %.1fs", backoff)
                stop_event.wait(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue
            logging.error("Could not open video source: %s", source)
            stop_event.set()
            return

        backoff = 1.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1.0 or fps > 120.0:
            fps = 25.0
        frame_delay = 1.0 / fps
        logging.info("Capture opened source=%s fps=%.2f", source, fps)

        while not stop_event.is_set():
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if source_type == "file" and loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cap.release()
                if source_type == "rtsp":
                    logging.warning("RTSP stream ended; reconnecting in %.1fs", backoff)
                    stop_event.wait(backoff)
                    backoff = min(backoff * 2.0, 30.0)
                    break
                logging.info("Video reached end of file.")
                stop_event.set()
                return
            with lock:
                latest["frame"] = frame
                latest["captured_at"] = datetime.now()
                latest["sequence"] += 1
            remaining = frame_delay - (time.monotonic() - started)
            if remaining > 0:
                stop_event.wait(remaining)
        cap.release()


def _convert_visits(
    finished: list[Any], camera_id: int
) -> list[dict[str, Any]]:
    rows = []
    for visit in finished:
        rows.append({
            "camera_id": camera_id,
            "zone_id": int(visit.zone_id),
            "track_id": int(visit.track_id),
            "entered_at": visit.entered_at,
            "left_at": visit.left_at,
            "dwell_sec": int(round(visit.dwell_sec)),
        })
    return rows


def _crossing_rows(crossings: list[Any], camera_id: int) -> list[dict[str, Any]]:
    rows = []
    for crossing in crossings:
        rows.append({
            "camera_id": camera_id,
            "line_id": int(crossing.line_id),
            "track_id": int(crossing.track_id),
            "ts": crossing.ts,
            "direction": crossing.direction,
        })
    return rows


def inference_loop(
    camera_id: int,
    config: dict[str, Any],
    latest: dict[str, Any],
    lock: threading.Lock,
    stop_event: threading.Event,
    db_queue: queue.Queue,
    loop_file: bool,
) -> None:
    camera = config["camera"]
    polygons = {
        int(zone["zone_id"]): zone["polygon"]
        for zone in config["zones"]
        if len(zone["polygon"]) >= 3
    }
    line_definitions = {
        int(line["line_id"]): (
            (int(line["x1"]), int(line["y1"])),
            (int(line["x2"]), int(line["y2"])),
        )
        for line in config["count_lines"]
    }
    stride = max(1, int(camera["frame_stride"] or 1))
    detector = PersonDetector(conf=0.25, imgsz=960)
    zone_monitor = ZoneMonitor(polygons, grace_sec=2.0)
    site_id = int(camera["site_id"])

    rule_engine = RuleEngine(
        site_id=site_id,
        connection_factory=rules_connection,
        clock=datetime.now,
    )

    incident_manager = IncidentManager(
        camera_id=camera_id,
        site_id=site_id,
        connection_factory=rules_connection,
        clock=datetime.now,
    )
    # Engine API expects each line ID to map to its two endpoint coordinates.
    line_counter = LineCounter(line_definitions)
    heatmap = None
    annotator = FrameAnnotator()
    last_sequence = -1
    processed_frames = 0
    last_observation = 0.0
    last_live_write = 0.0
    last_heatmap_write = 0.0
    last_annotated = None
    logging.info(
        "Inference started camera_id=%s stride=%s zones=%s lines=%s",
        camera_id, stride, len(polygons), len(line_definitions)
    )

    while not stop_event.is_set():
        with lock:
            frame = None if latest["frame"] is None else latest["frame"].copy()
            sequence = latest["sequence"]
        if frame is None or sequence == last_sequence:
            stop_event.wait(0.01)
            continue
        last_sequence = sequence
        processed_frames += 1
        if (processed_frames - 1) % stride != 0:
            continue

        now = datetime.now()
        try:
            tracks = detector.process(frame)
            if heatmap is None:
                height, width = frame.shape[:2]
                heatmap = HeatmapAccumulator(width, height)
            heatmap.update(tracks)

            counts, finished_visits = zone_monitor.update(tracks, now)
            dwell_by_track = zone_monitor.current_dwell(now)

            incident_manager.evaluate(
                dwell_by_track=dwell_by_track,
                rule_engine=rule_engine,
                now=now,
            )
            crossings = line_counter.update(tracks, now)

            
# The annotator accepts one line: ((x1, y1), (x2, y2)).
# Use the first configured counting line, if one exists.

            annotation_line = None

            if line_definitions:
                if isinstance(line_definitions, dict):
                    first_line = next(iter(line_definitions.values()), None)
                else:
                    first_line = line_definitions[0]

                if first_line is not None:
                    if isinstance(first_line, dict):
                        start = first_line.get("start")
                        end = first_line.get("end")
                        if start is not None and end is not None:
                            annotation_line = (tuple(start), tuple(end))
                    elif len(first_line) == 2:
                        annotation_line = (
                            tuple(first_line[0]),
                            tuple(first_line[1]),
                        )

            annotated = annotator.annotate(
                frame,
                tracks,
                zones=polygons,
                counts=counts,
                line=annotation_line,
            )
            last_annotated = annotated if annotated is not None else frame


            if finished_visits:
                rows = _convert_visits(finished_visits, camera_id)
                db_queue.put(("visits", rows))

            if crossings:
                rows = _crossing_rows(crossings, camera_id)
                db_queue.put(("crossings", rows))

            monotonic_now = time.monotonic()
            if monotonic_now - last_observation >= OBSERVATION_INTERVAL_SEC:
                rows = []
                for zone_id in polygons:
                    rows.append({
                        "camera_id": camera_id,
                        "zone_id": zone_id,
                        "ts": now.replace(microsecond=0),
                        "people_count": int(counts.get(zone_id, 0)),
                    })
                if rows:
                    db_queue.put(("observations", rows))
                last_observation = monotonic_now

            if monotonic_now - last_live_write >= LIVE_FRAME_INTERVAL_SEC:
                LIVE_DIR.mkdir(parents=True, exist_ok=True)
                live_path = LIVE_DIR / f"camera_{camera_id}.jpg"
                temp_path = LIVE_DIR / f"camera_{camera_id}.tmp.jpg"
                if cv2.imwrite(str(temp_path), last_annotated):
                    os.replace(temp_path, live_path)
                last_live_write = monotonic_now

            if heatmap is not None and monotonic_now - last_heatmap_write >= HEATMAP_INTERVAL_SEC:
                HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
                heatmap_path = HEATMAP_DIR / f"camera_{camera_id}.png"
                heatmap.save_png(str(heatmap_path))
                last_heatmap_write = monotonic_now

        except Exception:
            logging.exception("Inference failed for camera_id=%s", camera_id)
            stop_event.wait(0.2)

    # Flush latest heatmap on shutdown.
    if heatmap is not None:
        try:
            HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
            heatmap.save_png(str(HEATMAP_DIR / f"camera_{camera_id}.png"))
        except Exception:
            logging.exception("Could not save final heatmap")


def writer_loop(db_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Persist queued batches; retain failed batches and retry without losing process."""
    pending: list[tuple[str, list[dict[str, Any]]]] = []
    last_retry_log = 0.0
    while not stop_event.is_set() or not db_queue.empty() or pending:
        try:
            item = db_queue.get(timeout=0.25)
            pending.append(item)
            db_queue.task_done()
        except queue.Empty:
            pass

        if not pending:
            continue
        if len(pending) < 10 and time.monotonic() - last_retry_log < DB_BATCH_INTERVAL_SEC:
            continue

        # Group queued records by table before each transaction attempt.
        grouped: dict[str, list[dict[str, Any]]] = {
            "observations": [], "visits": [], "crossings": []
        }
        for kind, rows in pending:
            grouped[kind].extend(rows)

        try:
            if grouped["observations"]:
                insert_observations(grouped["observations"])
            if grouped["visits"]:
                insert_visits(grouped["visits"])
            if grouped["crossings"]:
                insert_crossings(grouped["crossings"])
            pending.clear()
            last_retry_log = time.monotonic()
        except Exception:
            if time.monotonic() - last_retry_log >= 5.0:
                logging.exception("Database write failed; retaining %s batches for retry", len(pending))
                last_retry_log = time.monotonic()
            stop_event.wait(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, required=True)
    parser.add_argument("--loop", action="store_true", help="Loop file sources at EOF")
    args = parser.parse_args()

    configure_logging()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        config = load_camera_config(args.camera_id)
    except Exception:
        logging.exception("Could not load camera configuration")
        raise SystemExit(1)

    camera = config["camera"]
    if not camera["is_active"]:
        logging.error("Camera %s is inactive in the database", args.camera_id)
        raise SystemExit(1)

    source = parse_source(camera)
    latest = {"frame": None, "captured_at": None, "sequence": 0}
    lock = threading.Lock()
    stop_event = threading.Event()
    db_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    capture_thread = threading.Thread(
        target=capture_loop,
        name="capture",
        args=(source, camera["source_type"], latest, lock, stop_event, args.loop),
        daemon=True,
    )
    inference_thread = threading.Thread(
        target=inference_loop,
        name="inference",
        args=(args.camera_id, config, latest, lock, stop_event, db_queue, args.loop),
        daemon=True,
    )
    writer_thread = threading.Thread(
        target=writer_loop, name="db-writer", args=(db_queue, stop_event), daemon=True
    )

    logging.info("Starting AuraVision worker camera_id=%s source_type=%s", args.camera_id, camera["source_type"])
    writer_thread.start()
    capture_thread.start()
    inference_thread.start()

    try:
        while not stop_event.wait(0.5):
            if not capture_thread.is_alive():
                stop_event.set()
                break
    except KeyboardInterrupt:
        logging.info("Ctrl+C received; shutting down")
        stop_event.set()
    finally:
        stop_event.set()
        capture_thread.join(timeout=5)
        inference_thread.join(timeout=10)
        writer_thread.join(timeout=15)
        logging.info("AuraVision worker stopped")


if __name__ == "__main__":
    main()
