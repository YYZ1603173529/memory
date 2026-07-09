# -*- coding: utf-8 -*-
"""Test memory retrieval: ask questions about a video using generated memory as context."""

import argparse
import json
import os
from openai import OpenAI


def load_memory(memory_json_path: str) -> dict:
    with open(memory_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(data: dict, short_term_count: int = 2) -> str:
    """Build a three-tier memory context matching the paper's hierarchy.

    - Short-term: the most recent N mid-term summaries (most detailed, "what just happened")
    - Mid-term: older mid-term summaries (still detailed, per-chunk)
    - Long-term: compressed blocks covering the full video span
    """
    parts = []
    mid_term = data.get("mid_term_history", [])

    # Short-term: last N summaries (most recent, highest detail)
    if mid_term:
        short = mid_term[-short_term_count:] if len(mid_term) >= short_term_count else mid_term
        short_parts = []
        for entry in short:
            chunk_text = f"<{entry['frame_range']}>\n{entry['summary_text']}"
            # Include raw ASR transcripts for short-term memory
            audio_transcripts = entry.get("audio_transcripts", [])
            if audio_transcripts:
                asr_lines = ["\n[ASR Transcripts]"]
                for at in audio_transcripts:
                    asr_lines.append(f"  {at['time_range']}: {at['text']}")
                chunk_text += "\n".join(asr_lines)
            short_parts.append(chunk_text)
        parts.append("[Short-term Memory — recent detailed context]\n" + "\n\n".join(short_parts))

    # Mid-term: remaining older summaries (detailed backup)
    if len(mid_term) > short_term_count:
        older = mid_term[:-short_term_count]
        older_parts = []
        for entry in older:
            older_parts.append(f"<{entry['frame_range']}>\n{entry['summary_text']}")
        parts.append("[Mid-term Memory — older chunk summaries]\n" + "\n\n".join(older_parts))

    # Long-term: compressed blocks (full history, highly compressed)
    long_term = data.get("long_term_memory", "")
    if long_term:
        parts.append("[Long-term Memory — compressed full history]\n" + long_term)

    return "\n\n" + "=" * 50 + "\n\n".join(parts)


def ask_question(
    question: str,
    context: str,
    client: OpenAI,
    model: str = "qwen3-omni-flash",
) -> str:
    prompt = f"""You are answering questions about a video you watched earlier.
Below is your memory of what happened in the video, organized by time ranges.
Answer the user's question based ONLY on the memory content below.
If the memory doesn't contain enough information, say so honestly.

{context}

[Question]
{question}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Test memory retrieval from generated memory JSON")
    parser.add_argument("--memory", "-m", help="Path to memory JSON file (e.g. my_output/memory_gym_01_*.json)")
    parser.add_argument("--question", "-q", help="Single question to ask (non-interactive mode)")
    parser.add_argument("--show_context", action="store_true", help="Print the full context before answering")
    args = parser.parse_args()

    # Find latest memory file if not specified
    if not args.memory:
        import glob
        files = sorted(glob.glob("my_output/memory_gym_01_*.json"))
        if not files:
            print("No memory files found. Specify --memory path.")
            return
        args.memory = files[-1]
        print(f"Using memory: {args.memory}")

    data = load_memory(args.memory)
    context = build_context(data)

    video_name = data.get("video_name", "unknown")
    duration = data.get("duration", 0)

    api_key = os.environ.get(
        "MODEL_API_KEY",
        "EMPTY",
    )
    api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)

    if args.question:
        # Single question mode
        print(f"\nQ: {args.question}\n")
        if args.show_context:
            print("=" * 50)
            print(context[:2000] + ("..." if len(context) > 2000 else ""))
            print("=" * 50 + "\n")
        answer = ask_question(args.question, context, client)
        print(f"A: {answer}")
    else:
        # Interactive mode
        print(f"Video: {video_name} ({duration:.0f}s)")
        print(f"Memory: {len(data.get('mid_term_history', []))} mid-term chunks, "
              f"{len(data.get('long_term_history', []))} long-term blocks")
        print(f"Context size: {len(context)} chars")
        print("\nEnter questions (Ctrl+C to exit):\n")

        while True:
            try:
                q = input("Q: ").strip()
                if not q:
                    continue
                if q == "/context":
                    print(context[:3000] + ("..." if len(context) > 3000 else ""))
                    continue
                answer = ask_question(q, context, client)
                print(f"A: {answer}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nDone.")
                break


if __name__ == "__main__":
    main()
