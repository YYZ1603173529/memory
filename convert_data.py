import argparse
import json
import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

MAX_DURATION = 320  # Maximum frame extraction duration in seconds; override with --max_duration.
FFMPEG_THREADS = 2

# Process-level duration cache to avoid repeated ffprobe calls.
_duration_cache = {}
_duration_lock = threading.Lock()
_duration_in_flight = {}

# Frame output directory locks to prevent concurrent extraction/deletion for the same out_dir.
# value: (Lock, refcount)
_dir_locks = {}
_dir_locks_guard = threading.Lock()


def _get_ffmpeg_exe():
    """Return path to ffmpeg executable (from imageio-ffmpeg or system PATH)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def get_video_duration(video_path):
    """Get the actual video duration in seconds, using a thread-safe process cache."""
    with _duration_lock:
        if video_path in _duration_cache:
            return _duration_cache[video_path]
        if video_path in _duration_in_flight:
            event = _duration_in_flight[video_path]
        else:
            _duration_in_flight[video_path] = threading.Event()
            event = None
    if event is not None:
        event.wait()
        return _duration_cache[video_path]

    ffmpeg = _get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True,
    )
    dur = 0.0
    # Parse duration from ffmpeg stderr: "Duration: 00:01:23.45"
    import re as _re
    match = _re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        dur = int(h) * 3600 + int(m) * 60 + float(s)
    with _duration_lock:
        _duration_cache[video_path] = dur
        _duration_in_flight.pop(video_path).set()
    return dur


def _list_frame_paths(output_dir, expected_count=None):
    if expected_count is not None:
        return [os.path.join(output_dir, f"frame_{i:06d}.jpg") for i in range(expected_count)]
    return sorted(
        os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")
    )


def extract_frames(video_path, output_dir, fps=1.0, rewrite=False,
                   max_duration=None):
    """Extract frames from a video at the given fps.

    Returns (frame_paths, effective_duration, truncated).
    Videos longer than max_duration only use the first max_duration seconds.
    """
    if max_duration is None:
        max_duration = MAX_DURATION
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    if duration <= 0:
        raise ValueError(f"ffprobe returned duration={duration} for {video_path}")

    # max_duration <= 0 means no limit.
    if max_duration and max_duration > 0:
        truncated = duration > max_duration
        effective_duration = min(duration, max_duration)
    else:
        truncated = False
        effective_duration = duration

    # Get the directory-specific lock to prevent concurrent extraction/deletion in the same directory.
    with _dir_locks_guard:
        if output_dir not in _dir_locks:
            _dir_locks[output_dir] = [threading.Lock(), 0]
        entry = _dir_locks[output_dir]
        entry[1] += 1
        dir_lock = entry[0]

    try:
        with dir_lock:
            # If frame files already exist and rewrite is not requested, return them directly.
            existing = _list_frame_paths(output_dir)
            if existing and not rewrite:
                return existing, effective_duration, truncated

            # Delete old frames in rewrite mode.
            if rewrite and existing:
                for p in existing:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
            ffmpeg = _get_ffmpeg_exe()
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
            ]
            cmd += [
                "-threads",
                str(FFMPEG_THREADS),
                "-fflags",
                "+discardcorrupt",
                "-err_detect",
                "ignore_err",
            ]
            cmd += [
                "-i",
                video_path,
            ]
            cmd += [
                "-vf",
                f"fps={fps}",
                "-q:v",
                "5",
                "-start_number",
                "0",
                "-y",
                output_pattern,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            expected_count = max(int(duration * fps), 1)
            paths = _list_frame_paths(output_dir, expected_count)
            if not paths or not os.path.exists(paths[0]):
                paths = _list_frame_paths(output_dir)
            if not paths:
                stderr = (result.stderr or "").strip()
                raise ValueError(
                    f"ffmpeg extracted 0 frames from {video_path}"
                    + (f": {stderr}" if stderr else "")
                )
            return paths, effective_duration, truncated
    finally:
        with _dir_locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                del _dir_locks[output_dir]


def parse_times(time_str):
    """Parse a time field such as '8' or '5,6,7' and return a list of ints."""
    if not time_str:
        return []
    return [int(float(t.strip())) for t in str(time_str).split(",") if t.strip()]


def convert_sample(sample, frame_dir, rewrite=False,
                   max_duration=None):
    """Convert one raw sample to inference format.

    Returns (result_dict, truncated_bool, warning_msg) or (None, False, warning_msg).
    """
    video_path = sample["video_path"]
    task_type = sample.get("task_type", "")
    source = sample.get("source", "")
    video_stem = os.path.splitext(sample["video_name"])[0]

    # Dynamically adjust fps based on video duration.
    duration = get_video_duration(video_path)
    if duration >= 160:
        fps = 1.0
    elif duration >= 64:
        fps = 2.0
    else:
        fps = 4.0

    # Extract frames, automatically truncating videos longer than max_duration.
    frame_source = str(source or "").strip("/")
    out_dir = os.path.join(frame_dir, task_type, frame_source, video_stem)
    frame_paths, effective_duration, truncated = extract_frames(
        video_path, out_dir, fps, rewrite=rewrite, max_duration=max_duration)

    # Frame extraction may cover the full video, but JSON output only uses frames for effective_duration.
    effective_n = max(int(effective_duration * fps), 1)
    used_paths = frame_paths[:effective_n]

    # Group by second: each second contains fps frames, and messages are organized by second.
    frames_per_sec = max(int(fps), 1)
    n_seconds = max(int(effective_duration), 1)

    # question_map: second_idx -> text
    question_map = {}
    for q in sample.get("question", []):
        for t in parse_times(q.get("time")):
            if t > effective_duration:
                if truncated:
                    continue
                return None, False, (
                    f"question time {t}s > video duration {effective_duration:.2f}s, "
                    f"skipping: {sample.get('video_name', video_path)}")
            si = min(t, n_seconds - 1)
            question_map[si] = q["content"]

    # response_map: second_idx -> text
    # support both flat list [{"content":..,"time":..}, ...] and
    # nested list [[{"content":..,"time":..}, ...], ...] formats
    response_map = {}
    raw_responses = sample.get("response", [])
    flat_responses = []
    for item in raw_responses:
        if isinstance(item, list):
            flat_responses.extend(item)
        else:
            flat_responses.append(item)
    for r in flat_responses:
        for t in parse_times(r.get("time")):
            if t > effective_duration:
                if truncated:
                    continue
                return None, False, (
                    f"response time {t}s > video duration {effective_duration:.2f}s, "
                    f"skipping: {sample.get('video_name', video_path)}")
            si = min(t, n_seconds - 1)
            response_map[si] = r["content"]

    # Build messages. The system prompt is added automatically by qwen3vl.py.
    # Each second has one user message containing frames_per_sec <image> tags.
    messages = []
    for sec in range(n_seconds):
        # user
        parts = []
        if sec in question_map:
            parts.append(question_map[sec])
        parts.append(f"<{sec:.1f} seconds>")
        for _ in range(frames_per_sec):
            parts.append("<image>")
        messages.append({"role": "user", "content": "\n".join(parts)})

        # assistant (ground truth)
        if sec in response_map:
            messages.append({"role": "assistant", "content": f"</response> {response_map[sec]}"})
        else:
            messages.append({"role": "assistant", "content": "</silence>"})

    # Keep only the frames actually used by the per-second grouping.
    actual_used_paths = used_paths[:n_seconds * frames_per_sec]

    return {
        "messages": messages,
        "images": actual_used_paths,
        "video_name": sample["video_name"],
        "video_path": video_path,
        "task_type": task_type,
        "source": source,
    }, truncated, None


def collect_json_files(path):
    """Collect all JSON files under the input path."""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in sorted(names):
            if name.endswith(".json") and name != "example.json":
                files.append(os.path.join(root, name))
    return files


def _process_sample(sample_and_args):
    """Wrapper for concurrent workers. Returns (result, truncated, warning)."""
    (
        sample,
        frame_dir,
        rewrite,
        max_duration,
    ) = sample_and_args
    try:
        return convert_sample(sample, frame_dir, rewrite=rewrite,
                              max_duration=max_duration)
    except Exception as exc:
        return None, False, (
            f"{type(exc).__name__}: {exc}, "
            f"skipping: {sample.get('video_name', sample.get('video_path', '<unknown>'))}"
        )


def _log_sample_result(result, was_truncated, warn_msg, logger, truncated_logger,
                       max_duration):
    """Log one sample result and return (filtered_count_increment, truncated_count_increment)."""
    filtered_inc = 0
    truncated_inc = 0

    if warn_msg:
        filtered_inc = 1
        logger.warning(f"  Filtered sample: {warn_msg}")

    if result and was_truncated:
        truncated_inc = 1
        orig_dur = _duration_cache[result["video_path"]]
        truncated_logger.info(
            f"TRUNCATED: {result['video_name']} | "
            f"path={result['video_path']} | "
            f"original_duration={orig_dur:.2f}s | "
            f"used_duration={max_duration}s | "
            f"frames={len(result['images'])}")
        logger.info(
            f"  Truncated video: {result['video_name']} "
            f"({orig_dur:.2f}s -> {max_duration}s)")

    return filtered_inc, truncated_inc


def setup_logging(log_dir):
    """Set up timestamped logs and return (general_logger, truncated_logger)."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # General log.
    general_logger = logging.getLogger("convert_eval")
    general_logger.handlers.clear()
    general_logger.setLevel(logging.INFO)
    general_log_path = os.path.join(log_dir, f"convert_eval_{timestamp}.log")
    gh = logging.FileHandler(general_log_path, encoding="utf-8")
    gh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    general_logger.addHandler(gh)
    # Also output to console.
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    general_logger.addHandler(sh)

    # Truncation log for videos longer than max_duration.
    truncated_logger = logging.getLogger("truncated")
    truncated_logger.handlers.clear()
    truncated_logger.setLevel(logging.INFO)
    truncated_log_path = os.path.join(log_dir, f"truncated_videos_{timestamp}.log")
    th = logging.FileHandler(truncated_log_path, encoding="utf-8")
    th.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    truncated_logger.addHandler(th)

    general_logger.info(f"General log: {general_log_path}")
    general_logger.info(f"Truncation log: {truncated_log_path}")

    return general_logger, truncated_logger


def main():
    parser = argparse.ArgumentParser(description="Convert raw evaluation data to inference format")
    parser.add_argument("--input", "-i", help="Input JSON file or raw_data/ directory")
    parser.add_argument("--output", "-o", help="Output directory; preserves the original directory structure")
    parser.add_argument("--frame_dir", help="Root directory for saved frame images")
    parser.add_argument("--max_samples", type=int, default=0, help="Only process the first N samples; 0 means process all samples")
    parser.add_argument("--rewrite", action="store_true", help="Force rewrite; equivalent to specifying both --rewrite_frames and --rewrite_json")
    parser.add_argument("--rewrite_frames", action="store_true", help="Force frame re-extraction and overwrite existing frame files")
    parser.add_argument("--rewrite_json", action="store_true", help="Force output JSON regeneration and overwrite existing files")
    parser.add_argument("--workers", "-w", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--log_dir", default="logs", help="Directory for saved logs")
    parser.add_argument("--max_duration", type=int, default=MAX_DURATION,
                        help=f"Maximum frame extraction duration in seconds; longer videos are truncated (default {MAX_DURATION})")

    # Memory generation mode
    parser.add_argument("--generate_memory", help="Generate memory from a single video file (path to video)")
    parser.add_argument("--memory_output", default="memory_output", help="Output directory for memory results (default: memory_output)")
    parser.add_argument("--audio_dir", default="", help="Directory for extracted audio segments (enables ASR)")
    parser.add_argument("--chunk_size", type=int, default=200, help="Frames per chunk (default: 200)")
    parser.add_argument("--compress_every", type=int, default=5, help="Compress to long-term every N chunks (default: 5)")

    args = parser.parse_args()

    # Memory generation mode
    if args.generate_memory:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        frame_dir = args.frame_dir or "memory_frames"
        generate_memory_from_video(
            video_path=args.generate_memory,
            output_dir=args.memory_output,
            frame_dir=frame_dir,
            audio_dir=args.audio_dir,
            chunk_size=args.chunk_size,
            compress_every=args.compress_every,
            max_duration=args.max_duration,
        )
        return

    if args.rewrite:
        args.rewrite_frames = True
        args.rewrite_json = True

    logger, truncated_logger = setup_logging(args.log_dir)

    input_base = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    input_files = collect_json_files(args.input)
    logger.info(f"Found {len(input_files)} JSON files, using {args.workers} workers, "
                f"max_duration={args.max_duration}s")

    global_count = 0
    truncated_count = 0
    filtered_count = 0
    for fpath in tqdm(input_files, desc="Converting"):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        if args.max_samples > 0:
            remaining = args.max_samples - global_count
            if remaining <= 0:
                break
            data = data[:remaining]

        rel = os.path.relpath(fpath, input_base)
        out_path = os.path.join(args.output, rel)
        if not args.rewrite_json and os.path.exists(out_path):
            logger.info(f"  {rel}: output already exists, skipping (use --rewrite_json or --rewrite to overwrite)")
            continue

        # Prefetch all unique video durations in parallel in the main process.
        unique_videos = list({s["video_path"] for s in data} - set(_duration_cache))
        if unique_videos:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(get_video_duration, unique_videos))

        # Process samples concurrently.
        task_args = [
            (
                sample,
                args.frame_dir,
                args.rewrite_frames,
                args.max_duration,
            )
            for sample in data
        ]
        results = []
        if args.workers > 1 and len(data) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_sample, ta): i for i, ta in enumerate(task_args)}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"  {os.path.basename(fpath)}", leave=False):
                    r, was_truncated, warn_msg = future.result()
                    filtered_inc, truncated_inc = _log_sample_result(
                        r, was_truncated, warn_msg, logger, truncated_logger,
                        args.max_duration)
                    filtered_count += filtered_inc
                    truncated_count += truncated_inc
                    if r:
                        results.append((futures[future], r))
            # Preserve original order.
            results.sort(key=lambda x: x[0])
            results = [r for _, r in results]
        else:
            for ta in tqdm(task_args, desc=f"  {os.path.basename(fpath)}", leave=False):
                r, was_truncated, warn_msg = _process_sample(ta)
                filtered_inc, truncated_inc = _log_sample_result(
                    r, was_truncated, warn_msg, logger, truncated_logger,
                    args.max_duration)
                filtered_count += filtered_inc
                truncated_count += truncated_inc
                if r:
                    results.append(r)

        if not results:
            continue

        # Preserve directory structure: raw_data/qa/.../1.json -> data/qa/.../1.json.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        global_count += len(results)
        logger.info(f"  {rel}: {len(results)} samples -> {out_path}")

        if args.max_samples > 0 and global_count >= args.max_samples:
            break

    logger.info(f"Done. Processed {global_count} samples, "
                f"{truncated_count} videos were truncated (>{args.max_duration}s), "
                f"{filtered_count} samples were filtered")


# ============================================================
# Memory Generation Mode
# ============================================================

def extract_audio_for_chunk(video_path: str, audio_dir: str, start_sec: float,
                             duration_sec: float) -> str:
    """Extract one audio segment for a chunk and return its WAV path.

    Returns empty string if no audio stream or extraction fails.
    """
    os.makedirs(audio_dir, exist_ok=True)
    wav_path = os.path.join(audio_dir, f"audio_{int(start_sec):06d}_{int(duration_sec)}s.wav")
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path
    ffmpeg = _get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-y", wav_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path
    return ""


def generate_memory_from_video(
    video_path: str,
    output_dir: str,
    frame_dir: str,
    audio_dir: str = "",
    chunk_size: int = 200,
    compress_every: int = 5,
    max_duration: int = None,
    summarizer_model: str = "qwen3-omni-flash",
    summarizer_api_base: str = "",
    api_key: str = "",
    asr_model: str = "qwen3-asr-flash",
):
    """Process a video through the three-tier memory pipeline and save results.

    Steps:
      1. Extract frames at 1 FPS
      2. Extract audio (optional, if audio_dir is set)
      3. Group into chunks, generate mid-term summaries
      4. Compress into long-term memory every N chunks
      5. Save memory trace to output_dir
    """
    from memory_summarizer import SummarizerModel

    summarizer_api_base = summarizer_api_base or "http://localhost:8000/v1"
    api_key = api_key or "EMPTY"

    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("memory_gen")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(sh)

    # 1. Extract frames
    logger.info(f"Extracting frames from {video_path}...")
    frame_paths, effective_duration, truncated = extract_frames(
        video_path, frame_dir, fps=1.0, rewrite=False, max_duration=max_duration)
    logger.info(f"  {len(frame_paths)} frames, duration={effective_duration:.1f}s")

    # 3. Initialize summarizer
    summarizer = SummarizerModel(
        model_name=summarizer_model,
        api_base=summarizer_api_base,
        api_key=api_key,
        asr_model_name=asr_model,
        key_frames_per_chunk=8,
        max_pixels=262144,
        prompt_phase_seconds=10.0,
        mid_term_max_tokens=4000,
        long_term_max_tokens=2000,
    )

    # 4. Process chunks
    frame_time_ranges = [f"{i:.1f} seconds" for i in range(len(frame_paths))]
    n_frames = len(frame_paths)
    chunk_count = (n_frames + chunk_size - 1) // chunk_size

    mid_term_summaries = []       # [{chunk_index, frame_range, summary_text, ...}]
    mid_term_history = []
    long_term_history = []
    long_term_memory = ""
    compression_index = 1
    enable_audio = bool(audio_dir)

    for chunk_idx in range(chunk_count):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_frames = frame_paths[chunk_start:chunk_end]
        chunk_time_ranges = frame_time_ranges[chunk_start:chunk_end]

        if not chunk_frames:
            continue

        frame_range = f"{chunk_time_ranges[0]}-{chunk_time_ranges[-1]}"
        logger.info(f"Chunk {chunk_idx + 1}/{chunk_count}: {frame_range} ({len(chunk_frames)} frames)")

        # Select key frames
        key_frames = summarizer.select_key_frames(
            chunk_frames, chunk_time_ranges, [])
        logger.info(f"  Key frames: {len(key_frames)}")

        # Build audio transcript for this chunk: extract one continuous segment + ASR once
        audio_transcripts = []
        if enable_audio:
            chunk_duration = chunk_end - chunk_start
            wav_path = extract_audio_for_chunk(video_path, audio_dir, float(chunk_start),
                                                float(chunk_duration))
            if wav_path:
                import base64 as _b64
                try:
                    with open(wav_path, "rb") as f:
                        audio_b64 = _b64.b64encode(f.read()).decode("ascii")
                    text = summarizer.transcribe_audio(audio_b64)
                    if text:
                        audio_transcripts.append({
                            "time_range": f"{chunk_start:.1f}s-{chunk_end:.1f}s",
                            "text": text,
                        })
                        logger.info(f"  Audio: {len(text)} chars transcribed")
                except Exception as e:
                    logger.warning(f"  ASR failed for chunk {chunk_idx + 1}: {e}")

        # Generate mid-term summary
        summary_text, _debug = summarizer.generate_detailed_summary(
            chunk_idx + 1,
            frame_range,
            key_frames,
            len(chunk_frames),
            audio_transcripts=audio_transcripts,
        )
        logger.info(f"  Summary: {summary_text[:120]}...")

        mid_entry = {
            "chunk_index": chunk_idx + 1,
            "frame_range": frame_range,
            "summary_text": summary_text,
            "frame_count": len(chunk_frames),
            "key_frame_count": len(key_frames),
            "audio_transcript_count": len(audio_transcripts),
            "audio_transcripts": audio_transcripts,
            "compressed_to_long_term": False,
        }
        mid_term_summaries.append(mid_entry)
        mid_term_history.append(mid_entry)

        # Compress to long-term every N chunks
        if len(mid_term_summaries) >= compress_every:
            logger.info(f"  Compressing {len(mid_term_summaries)} mid-term summaries to long-term...")
            merged, token_count, compressed_text, _debug = summarizer.batch_compress_to_longterm(
                long_term_memory, mid_term_summaries)
            long_term_memory = merged
            for entry in mid_term_summaries:
                entry["compressed_to_long_term"] = True
                entry["compressed_batch_index"] = compression_index
            long_term_history.append({
                "batch_index": compression_index,
                "source_chunk_indices": [e["chunk_index"] for e in mid_term_summaries],
                "compressed_text": compressed_text,
                "token_count": token_count,
            })
            compression_index += 1
            mid_term_summaries.clear()
            logger.info(f"  Long-term memory tokens: {token_count}")

    # Final compression for remaining mid-term summaries
    if mid_term_summaries:
        logger.info(f"  Final compression: {len(mid_term_summaries)} remaining summaries")
        merged, token_count, compressed_text, _debug = summarizer.batch_compress_to_longterm(
            long_term_memory, mid_term_summaries)
        long_term_memory = merged
        for entry in mid_term_summaries:
            entry["compressed_to_long_term"] = True
        long_term_history.append({
            "batch_index": compression_index,
            "source_chunk_indices": [e["chunk_index"] for e in mid_term_summaries],
            "compressed_text": compressed_text,
            "token_count": token_count,
        })

    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    result = {
        "video_path": video_path,
        "video_name": video_stem,
        "duration": effective_duration,
        "total_frames": n_frames,
        "chunk_size": chunk_size,
        "compress_every": compress_every,
        "mid_term_history": mid_term_history,
        "long_term_history": long_term_history,
        "long_term_memory": long_term_memory,
    }

    result_path = os.path.join(output_dir, f"memory_{video_stem}_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Memory saved to: {result_path}")

    # Also save a plain-text version of the long-term memory
    text_path = os.path.join(output_dir, f"memory_{video_stem}_{timestamp}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Duration: {effective_duration:.1f}s\n")
        f.write(f"Chunks: {chunk_count}, Frames: {n_frames}\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== MID-TERM SUMMARIES ===\n\n")
        for entry in mid_term_history:
            f.write(f"[Chunk {entry['chunk_index']}] {entry['frame_range']}\n")
            f.write(f"{entry['summary_text']}\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== LONG-TERM MEMORY ===\n\n")
        f.write(long_term_memory)
    logger.info(f"Text memory saved to: {text_path}")

    return result


if __name__ == "__main__":
    main()
