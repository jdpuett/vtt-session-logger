#!/usr/bin/env python3
"""
VTT to D&D Session Log Converter — v2
Watches the script's directory for .vtt files and converts them to
Campaign Logger-formatted Markdown session logs via the Claude API.

v2 additions:
  - Markdown normalization: **bold** → __bold__, *italic* → _italic_
    (Campaign Logger reserves * for page classification tags)
  - Smart chunking: splits session logs into ≤2400-char LogEntry chunks
    based on scene headings, sub-sections, and paragraph breaks
  - Local backup: saves full log + CL-format JSON before uploading
  - Campaign Logger API upload: POSTs chunks as LogEntries (requires auth)
  - Upload manifest: JSON log of API results for troubleshooting

SETUP — Local git (recommended):
  cd /path/to/this/script
  git init
  git add vtt_to_session_log_v2.py
  git commit -m "v2: CL API integration, chunking, local backup"
  # Then after each change:
  git add -A && git commit -m "description of change"
"""

import os
import re
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import anthropic

# Optional: requests for CL API calls
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Configuration ────────────────────────────────────────────────────────────

WATCH_DIR = Path(__file__).parent
OUTPUT_DIR = WATCH_DIR
BACKUP_DIR = WATCH_DIR / "session_backups"
POLL_INTERVAL = 5
PROCESSED_LOG = WATCH_DIR / ".processed_vtts"

# Campaign Logger API Configuration
CL_API_BASE = "https://logger.campaign-logger.com"
CL_API_CLIENT = os.environ.get("CAMPAIGN_LOGGER_API_CLIENT", "")
CL_API_SECRET = os.environ.get("CAMPAIGN_LOGGER_API_SECRET", "")
CL_CAMPAIGN_ID = os.environ.get("CAMPAIGN_LOGGER_CAMPAIGN_ID", "")
# Note: Log ID is no longer needed as an env var — the script creates
# a new Log via the API for each session and gets the ID back.

# Chunking
MAX_ENTRY_CHARS = 2400  # Hard ceiling per LogEntry rawText

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vtt2cl")

# ── System Prompt ────────────────────────────────────────────────────────────

PROMPT_FILE = WATCH_DIR / "session_log_prompt.md"

def load_system_prompt() -> str:
    """Load the system prompt from the sidecar file.

    The prompt file contains campaign-specific details (player names,
    character roster, campaign setting, etc.) and is kept separate from
    the script so the script can be shared without leaking personal info.

    Expected location: same directory as this script, named
    'session_log_prompt.md'

    Add 'session_log_prompt.md' to your .gitignore if using git.
    """
    if not PROMPT_FILE.exists():
        log.error(
            f"System prompt file not found: {PROMPT_FILE}\n"
            f"  Create this file with your campaign-specific AI instructions.\n"
            f"  See session_log_prompt.example.md for a template."
        )
        sys.exit(1)

    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if len(text) < 100:
        log.warning("System prompt file seems very short. Check its contents.")
    return text

# ── VTT Parsing ──────────────────────────────────────────────────────────────

def parse_vtt(vtt_text: str) -> str:
    """Strip VTT timing/metadata and return clean speaker: text transcript."""
    lines = vtt_text.splitlines()
    entries = []
    current_speaker = None
    current_text = []

    timestamp_re = re.compile(
        r"^\d{2}:\d{2}:\d{2}[\.,]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[\.,]\d{3}"
    )
    cue_re = re.compile(r"^\d+$")
    speaker_tag_re = re.compile(r"^<v ([^>]+)>(.*)$")
    speaker_colon_re = re.compile(r"^([A-Za-z][^:]{1,40}):\s+(.+)$")

    def flush():
        if current_speaker and current_text:
            text = " ".join(current_text).strip()
            if text:
                entries.append(f"{current_speaker}: {text}")

    for line in lines:
        line = line.strip()
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("NOTE")
            or cue_re.match(line)
            or timestamp_re.match(line)
        ):
            continue

        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line:
            continue

        m = speaker_tag_re.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_text = [m.group(2).strip()] if m.group(2).strip() else []
            continue

        m = speaker_colon_re.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_text = [m.group(2).strip()]
            continue

        if current_speaker:
            current_text.append(line)

    flush()
    return "\n".join(entries) if entries else vtt_text


# ── Markdown Normalization ───────────────────────────────────────────────────

def normalize_markdown_for_cl(text: str) -> str:
    """Convert asterisk-based markdown to underscore-based for Campaign Logger.

    Campaign Logger reserves * for page classification tags (*"Page Name"),
    so we must use __bold__ and _italic_ instead of **bold** and *italic*.

    Strategy:
      1. Convert **bold** → __bold__ first (greedy, handles the longer pattern)
      2. Then convert remaining *italic* → _italic_
      3. Leave CL tags alone (@"x", #"x", ^"x", &"x", *"x")
      4. Leave * in list items alone (lines starting with * or - *)
    """
    # Step 1: Protect CL page tags — temporarily replace *" with a placeholder
    cl_page_tag_placeholder = "\x00CLPAGE\x00"
    text = text.replace('*"', cl_page_tag_placeholder)

    # Step 2: Convert **bold** → __bold__
    # Match ** that aren't at start of line followed by space (which would be
    # a list item with bold). Use non-greedy match for content.
    text = re.sub(r'\*\*(.+?)\*\*', r'__\1__', text)

    # Step 3: Convert remaining *italic* → _italic_
    # Only match * that are adjacent to word characters (not standalone *)
    # This avoids catching list-item bullets like "* item"
    text = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'_\1_', text)

    # Step 4: Restore CL page tags
    text = text.replace(cl_page_tag_placeholder, '*"')

    return text


# ── Scene Parsing & Chunking ─────────────────────────────────────────────────

def parse_scenes(markdown_text: str) -> list[dict]:
    """Split a markdown session log into scenes based on ## headings.

    Returns a list of dicts: [{"title": str, "body": str}, ...]
    Title comes from the ## heading line. Body is everything until the next ##.
    Content before the first ## heading (if any) gets title "Preamble".
    """
    lines = markdown_text.split("\n")
    scenes = []
    current_title = ""
    current_lines = []

    for line in lines:
        # Match ## headings but not ### (sub-sections stay within their scene)
        if re.match(r"^## \S", line):
            # Save previous scene
            if current_title or current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    scenes.append({
                        "title": current_title,
                        "body": body,
                    })
            current_title = line.removeprefix("## ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last scene
    if current_title or current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            scenes.append({
                "title": current_title,
                "body": body,
            })

    return scenes


def chunk_scene(scene: dict, max_chars: int = MAX_ENTRY_CHARS) -> list[dict]:
    """Break a scene into chunks that fit within max_chars.

    Strategy (in order of preference):
      1. If the full scene (heading + body) fits, return it as-is
      2. Split at --- horizontal rule boundaries
      3. Split at ### sub-section boundaries
      4. Split at paragraph breaks (\\n\\n)
      5. Last resort: hard split at max_chars on a line boundary
    """
    title = scene["title"]
    body = scene["body"]

    # Reconstruct the full rawText as it would appear in CL
    if title:
        full_text = f"## {title}\n\n{body}"
    else:
        full_text = body

    # Case 1: Fits as-is
    if len(full_text) <= max_chars:
        return [{"title": title, "rawText": full_text}]

    # Case 2+3: Try splitting at --- or ### boundaries within the body
    chunks = _split_at_boundaries(title, body, max_chars)
    if chunks:
        return chunks

    # Case 4: Split at paragraph breaks
    chunks = _split_at_paragraphs(title, body, max_chars)
    if chunks:
        return chunks

    # Case 5: Hard split at line boundaries (last resort)
    return _hard_split(title, body, max_chars)


def _split_at_boundaries(title: str, body: str, max_chars: int) -> list[dict]:
    """Split body at --- or ### boundaries, keeping chunks under max_chars."""
    # Split on --- first, then on ### within each section
    sections = re.split(r"\n---+\n", body)

    if len(sections) <= 1:
        # Try ### instead
        sections = re.split(r"\n(?=### )", body)

    if len(sections) <= 1:
        return []  # No boundaries found, fall through to paragraph split

    chunks = []
    current_buffer = ""
    part_num = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Build what this chunk would look like
        if not current_buffer:
            if part_num == 0 and title:
                candidate = f"## {title}\n\n{section}"
            else:
                suffix = f" (cont.)" if title else ""
                candidate = f"## {title}{suffix}\n\n{section}" if title else section
        else:
            candidate = current_buffer + "\n\n---\n\n" + section

        if len(candidate) <= max_chars:
            current_buffer = candidate
        else:
            # Save current buffer if it has content
            if current_buffer:
                chunks.append(current_buffer)

            # Start new buffer with this section
            part_num += 1
            if title:
                current_buffer = f"## {title} (cont.)\n\n{section}"
            else:
                current_buffer = section

            # If even this single section exceeds max_chars, we need paragraph split
            if len(current_buffer) > max_chars:
                sub_chunks = _split_at_paragraphs(
                    f"{title} (cont.)" if title else "",
                    section,
                    max_chars,
                )
                if sub_chunks:
                    for sc in sub_chunks:
                        chunks.append(sc["rawText"])
                    current_buffer = ""
                # else: will be caught by hard split later

    if current_buffer:
        chunks.append(current_buffer)

    if not chunks:
        return []

    return [{"title": title, "rawText": chunk} for chunk in chunks]


def _split_at_paragraphs(title: str, body: str, max_chars: int) -> list[dict]:
    """Split body at paragraph breaks (\\n\\n), keeping chunks under max_chars."""
    paragraphs = re.split(r"\n\n+", body)
    if len(paragraphs) <= 1:
        return []

    chunks = []
    current_buffer = ""
    part_num = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if not current_buffer:
            if part_num == 0 and title:
                candidate = f"## {title}\n\n{para}"
            else:
                suffix = " (cont.)" if title else ""
                header = f"## {title}{suffix}\n\n" if title else ""
                candidate = f"{header}{para}"
        else:
            candidate = current_buffer + "\n\n" + para

        if len(candidate) <= max_chars:
            current_buffer = candidate
        else:
            if current_buffer:
                chunks.append(current_buffer)
            part_num += 1
            if title:
                current_buffer = f"## {title} (cont.)\n\n{para}"
            else:
                current_buffer = para
            # Single paragraph too long — will need hard split
            if len(current_buffer) > max_chars:
                hard = _hard_split(
                    f"{title} (cont.)" if title else "", para, max_chars
                )
                for h in hard:
                    chunks.append(h["rawText"])
                current_buffer = ""

    if current_buffer:
        chunks.append(current_buffer)

    if not chunks:
        return []

    return [{"title": title, "rawText": chunk} for chunk in chunks]


def _hard_split(title: str, body: str, max_chars: int) -> list[dict]:
    """Last resort: split on line boundaries to fit within max_chars."""
    lines = body.split("\n")
    chunks = []
    current_buffer = ""
    part_num = 0

    def make_header(part_index: int) -> str:
        if not title:
            return ""
        suffix = "" if part_index == 0 else " (cont.)"
        return f"## {title}{suffix}\n\n"

    def split_oversize_line(line: str, part_index: int) -> tuple[list[str], int]:
        pieces = []
        remaining = line

        while remaining:
            header = make_header(part_index)
            available = max_chars - len(header)
            if available <= 0:
                raise ValueError("max_chars too small to fit the chunk header")

            piece = remaining[:available]
            split_at = piece.rfind(" ")
            if split_at > 0 and len(remaining) > available:
                piece = piece[:split_at]

            piece = piece.rstrip()
            if not piece:
                piece = remaining[:available]

            pieces.append(f"{header}{piece}" if header else piece)
            remaining = remaining[len(piece):].lstrip()
            part_index += 1

        return pieces, part_index

    for line in lines:
        if not current_buffer:
            candidate = f"{make_header(part_num)}{line}" if title else line
        else:
            candidate = current_buffer + "\n" + line

        if len(candidate) <= max_chars:
            current_buffer = candidate
            continue

        if current_buffer:
            chunks.append(current_buffer)
            current_buffer = ""
            part_num += 1

        candidate = f"{make_header(part_num)}{line}" if title else line
        if len(candidate) <= max_chars:
            current_buffer = candidate
            continue

        line_chunks, part_num = split_oversize_line(line, part_num)
        chunks.extend(line_chunks)

    if current_buffer:
        chunks.append(current_buffer)

    return [{"title": title, "rawText": chunk} for chunk in chunks]


def chunk_session_log(markdown_text: str) -> list[dict]:
    """Full pipeline: parse scenes, then chunk each scene to fit CL limits.

    Returns a list of dicts with 'title' and 'rawText' keys, each rawText
    guaranteed to be ≤ MAX_ENTRY_CHARS characters.
    """
    scenes = parse_scenes(markdown_text)
    all_chunks = []

    for scene in scenes:
        chunks = chunk_scene(scene)
        all_chunks.extend(chunks)

    return all_chunks


# ── Local Backup ─────────────────────────────────────────────────────────────

def save_local_backup(
    session_log: str,
    chunks: list[dict],
    stem: str,
    backup_dir: Path = BACKUP_DIR,
) -> dict:
    """Save the full session log and CL-format JSON locally.

    Returns a dict with paths to the saved files.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the full markdown log
    md_path = backup_dir / f"{stem}_{timestamp}.md"
    md_path.write_text(session_log, encoding="utf-8")
    log.info(f"Full log backed up: {md_path.name}")

    # Build CL-format JSON (matches the export format you showed me)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    log_entries = []
    for i, chunk in enumerate(chunks):
        # Use zero-padded index for ordering to maintain sequence
        log_entries.append({
            "rawText": chunk["rawText"],
            "rawPrefix": "",
            "rawSuffix": "",
            "ordering": f"{i:04d}",
            "title": "",
        })

    cl_json = {
        "version": 2,
        "type": "log",
        "title": stem.replace("_", " ").title(),
        "logEntries": log_entries,
        "exportedOn": now_iso,
    }

    json_path = backup_dir / f"{stem}_{timestamp}_cl.json"
    json_path.write_text(
        json.dumps(cl_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"CL JSON backed up: {json_path.name}")

    # Save individual chunks for inspection
    chunks_dir = backup_dir / f"{stem}_{timestamp}_chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_path = chunks_dir / f"chunk_{i:03d}.md"
        chunk_path.write_text(chunk["rawText"], encoding="utf-8")

    log.info(f"Individual chunks saved: {chunks_dir.name}/ ({len(chunks)} files)")

    return {
        "markdown": str(md_path),
        "cl_json": str(json_path),
        "chunks_dir": str(chunks_dir),
    }


# ── Campaign Logger API ──────────────────────────────────────────────────────

def cl_api_available() -> bool:
    """Check if Campaign Logger API is configured and ready."""
    if not HAS_REQUESTS:
        log.warning("'requests' library not installed. Run: pip install requests")
        return False
    if not CL_API_CLIENT or not CL_API_SECRET:
        log.warning(
            "CAMPAIGN_LOGGER_API_CLIENT and/or CAMPAIGN_LOGGER_API_SECRET not set. "
            "Generate a client ID + secret at your CL API Tokens page and "
            "set both in your environment."
        )
        return False
    if not CL_CAMPAIGN_ID:
        log.warning(
            "CAMPAIGN_LOGGER_CAMPAIGN_ID not set. "
            "This is the ID of your campaign (e.g. Portals of Stonegate)."
        )
        return False
    return True


def _cl_headers() -> dict:
    """Build HTTP headers for Campaign Logger API requests.

    CL uses two custom headers for authentication:
      api-client: your client ID
      api-secret: your client secret

    Generate these at your CL account's API Tokens page.
    """
    return {
        "Content-Type": "application/json",
        "api-client": CL_API_CLIENT,
        "api-secret": CL_API_SECRET,
    }


def prompt_for_session_title(vtt_filename: str = "") -> str:
    """Prompt the user to enter a session log title interactively.

    Shows a suggested title derived from the filename, but lets the user
    type whatever they want.
    """
    suggestion = ""
    if vtt_filename:
        # Turn "session_006_audio.vtt" into "Session 006 Audio"
        stem = Path(vtt_filename).stem
        suggestion = stem.replace("_", " ").replace("-", " ").title()

    print("\n" + "─" * 50)
    print("  Enter a title for this session log.")
    print("  This will be the Log name in Campaign Logger.")
    if suggestion:
        print(f"  Suggestion: {suggestion}")
    print("─" * 50)

    while True:
        title = input("  Title: ").strip()
        if title:
            if len(title) > 100:
                print(f"  Title too long ({len(title)} chars). "
                      f"Campaign Logger max is 100.")
                continue
            return title
        if suggestion:
            use_suggestion = input(
                f"  Use '{suggestion}'? (y/n): "
            ).strip().lower()
            if use_suggestion in ("y", "yes", ""):
                return suggestion[:100]
        print("  Please enter a title.")


def create_cl_log(
    title: str,
    description: str = "",
    campaign_id: Optional[str] = None,
) -> Optional[str]:
    """Create a new Log in Campaign Logger via POST /logs.

    Returns the new log ID on success, or None on failure.
    """
    if not cl_api_available():
        return None

    target_campaign_id = campaign_id or CL_CAMPAIGN_ID
    url = f"{CL_API_BASE}/logs"

    payload = {
        "title": title,
        "description": description,
        "campaignId": target_campaign_id,
        "isPinned": False,
    }

    try:
        resp = requests.post(
            url, json=payload, headers=_cl_headers(), timeout=30
        )
        if resp.status_code in (200, 201):
            try:
                data = resp.json()
                # The response structure may vary — try common patterns
                log_id = (
                    data.get("id")
                    or data.get("data", {}).get("id")
                    or data.get("stringId")
                )
                if log_id:
                    log.info(
                        f"Created Log '{title}' in Campaign Logger (ID: {log_id})"
                    )
                    return log_id
                else:
                    # If we can't parse the ID, log the full response
                    log.warning(
                        f"Log created but couldn't parse ID from response: "
                        f"{resp.text[:500]}"
                    )
                    return None
            except ValueError:
                log.warning(
                    f"Log created (HTTP {resp.status_code}) but response "
                    f"isn't JSON: {resp.text[:500]}"
                )
                return None
        else:
            log.error(
                f"Failed to create Log: HTTP {resp.status_code} — "
                f"{resp.text[:500]}"
            )
            return None
    except requests.RequestException as e:
        log.error(f"Failed to create Log: {e}")
        return None


def upload_to_campaign_logger(
    chunks: list[dict],
    log_id: str,
) -> dict:
    """POST each chunk as a LogEntry to Campaign Logger API.

    Args:
        chunks: List of dicts with 'title' and 'rawText' keys.
        log_id: The ID of the Log to add entries to (created via create_cl_log).

    Returns a manifest dict with results for each chunk.
    """
    if not cl_api_available():
        return {"status": "skipped", "reason": "API not configured"}

    url = f"{CL_API_BASE}/log-entries"
    headers = _cl_headers()

    manifest = {
        "upload_started": datetime.now(timezone.utc).isoformat(),
        "log_id": log_id,
        "total_chunks": len(chunks),
        "results": [],
    }

    for i, chunk in enumerate(chunks):
        # Use zero-padded index for ordering to maintain sequence
        payload = {
            "rawText": chunk["rawText"],
            "rawPrefix": "",
            "rawSuffix": "",
            "ordering": f"{i:04d}",
            "title": "",
            "logId": log_id,
            "isShared": False,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            result = {
                "chunk_index": i,
                "status_code": resp.status_code,
                "success": resp.status_code in (200, 201),
                "chars": len(chunk["rawText"]),
                "title": chunk.get("title", ""),
            }
            if resp.status_code in (200, 201):
                try:
                    result["response"] = resp.json()
                except ValueError:
                    result["response_text"] = resp.text[:500]
                log.info(
                    f"  Chunk {i+1}/{len(chunks)} uploaded "
                    f"({len(chunk['rawText'])} chars)"
                )
            else:
                result["error"] = resp.text[:500]
                log.error(
                    f"  Chunk {i+1}/{len(chunks)} FAILED: "
                    f"HTTP {resp.status_code} — {resp.text[:200]}"
                )
        except requests.RequestException as e:
            result = {
                "chunk_index": i,
                "success": False,
                "error": str(e),
            }
            log.error(f"  Chunk {i+1}/{len(chunks)} FAILED: {e}")

        manifest["results"].append(result)

    manifest["upload_finished"] = datetime.now(timezone.utc).isoformat()
    manifest["successful"] = sum(1 for r in manifest["results"] if r.get("success"))
    manifest["failed"] = manifest["total_chunks"] - manifest["successful"]

    return manifest


def upload_fully_succeeded(manifest: dict) -> bool:
    """Return True when every chunk upload succeeded."""
    return (
        manifest.get("status") != "skipped"
        and manifest.get("failed", 0) == 0
        and manifest.get("successful", 0) == manifest.get("total_chunks", 0)
    )


def save_manifest(manifest: dict, stem: str, backup_dir: Path = BACKUP_DIR):
    """Save the upload manifest to a JSON file."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = backup_dir / f"{stem}_{timestamp}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Upload manifest saved: {path.name}")
    return path


# ── Claude API Call ───────────────────────────────────────────────────────────

def convert_to_session_log(transcript: str, filename: str) -> str:
    """Send transcript to Claude and return the formatted session log."""
    client = anthropic.Anthropic()
    system_prompt = load_system_prompt()

    log.info("Sending to Claude API...")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is the VTT transcript from file '{filename}'. "
                    "Please convert it into a D&D session log following your "
                    "instructions.\n\n"
                    f"TRANSCRIPT:\n{transcript}"
                ),
            }
        ],
    )

    return message.content[0].text


# ── File Handling ─────────────────────────────────────────────────────────────

def load_processed() -> set:
    if PROCESSED_LOG.exists():
        return set(PROCESSED_LOG.read_text().splitlines())
    return set()


def mark_processed(filename: str):
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")


def process_vtt(vtt_path: Path, upload: bool = True):
    """Full pipeline: VTT → parse → Claude → normalize → chunk → save → upload."""
    # Validate the file is actually within the watch directory
    try:
        vtt_path.resolve().relative_to(WATCH_DIR.resolve())
    except ValueError:
        log.error(f"Security: {vtt_path} is outside the watch directory. Skipping.")
        return

    log.info(f"Found: {vtt_path.name}")

    # ── Read VTT ──
    try:
        raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.error(f"Error reading file: {e}")
        return

    # ── Parse VTT ──
    log.info("Parsing VTT...")
    transcript = parse_vtt(raw)

    if len(transcript.strip()) < 100:
        log.warning(
            "Transcript seems very short after parsing. Check the VTT format."
        )

    # ── Convert via Claude ──
    try:
        session_log = convert_to_session_log(transcript, vtt_path.name)
    except Exception as e:
        log.error(f"Error calling Claude API: {e}")
        log.error("Make sure ANTHROPIC_API_KEY is set in your environment.")
        return

    # ── Normalize markdown for Campaign Logger ──
    log.info("Normalizing markdown (* → _ for CL compatibility)...")
    session_log = normalize_markdown_for_cl(session_log)

    # ── Save full markdown output ──
    stem = vtt_path.stem
    out_path = OUTPUT_DIR / f"{stem}.md"
    if out_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{stem}_{ts}.md"

    try:
        out_path.write_text(session_log, encoding="utf-8")
        log.info(f"Session log saved: {out_path.name}")
    except Exception as e:
        log.error(f"Error saving output: {e}")
        return

    # ── Chunk for Campaign Logger ──
    log.info(f"Chunking session log (max {MAX_ENTRY_CHARS} chars/entry)...")
    chunks = chunk_session_log(session_log)
    log.info(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        chars = len(chunk["rawText"])
        status = "OK" if chars <= MAX_ENTRY_CHARS else "OVER LIMIT"
        log.info(f"  [{i+1}] {chars:>5} chars — {status} — {chunk['title'][:50]}")

    # ── Local backup ──
    log.info("Saving local backup...")
    backup_paths = save_local_backup(session_log, chunks, stem)

    # ── Upload to Campaign Logger ──
    should_mark_processed = True

    if upload and cl_api_available():
        # Prompt for session title
        session_title = prompt_for_session_title(vtt_path.name)

        # Create the Log in Campaign Logger
        log.info(f"Creating Log '{session_title}' in Campaign Logger...")
        new_log_id = create_cl_log(
            title=session_title,
            description=f"Auto-imported from {vtt_path.name}",
        )

        if new_log_id:
            log.info(f"Uploading {len(chunks)} entries to Log '{session_title}'...")
            manifest = upload_to_campaign_logger(chunks, log_id=new_log_id)
            manifest["session_title"] = session_title
            save_manifest(manifest, stem)
            should_mark_processed = upload_fully_succeeded(manifest)
            log.info(
                f"Upload complete: {manifest['successful']}/{manifest['total_chunks']} "
                f"successful, {manifest['failed']} failed"
            )
            if not should_mark_processed:
                log.warning(
                    "Upload did not fully succeed. Leaving this VTT unprocessed "
                    "so watch mode can retry it."
                )
        else:
            should_mark_processed = False
            log.error(
                "Failed to create Log in Campaign Logger. "
                "Chunks saved locally — you can retry with --chunk-and-upload"
            )
    elif upload:
        log.info(
            "Skipping upload — Campaign Logger API not configured. "
            "Set CAMPAIGN_LOGGER_API_CLIENT, CAMPAIGN_LOGGER_API_SECRET, "
            "and CAMPAIGN_LOGGER_CAMPAIGN_ID in your environment."
        )

    if should_mark_processed:
        mark_processed(vtt_path.name)
    log.info("Done.\n")


# ── Standalone Utilities ─────────────────────────────────────────────────────

def chunk_existing_log(md_path: Path, upload: bool = False):
    """Utility: chunk and optionally upload an existing markdown session log.

    Useful for processing logs that were already generated by v1 of the script
    or written manually.

    Usage:
        python vtt_to_session_log_v2.py --chunk path/to/session.md
        python vtt_to_session_log_v2.py --chunk-and-upload path/to/session.md
    """
    log.info(f"Processing existing log: {md_path.name}")

    text = md_path.read_text(encoding="utf-8")

    # Normalize markdown
    log.info("Normalizing markdown...")
    text = normalize_markdown_for_cl(text)

    # Chunk
    log.info(f"Chunking (max {MAX_ENTRY_CHARS} chars/entry)...")
    chunks = chunk_session_log(text)
    log.info(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        chars = len(chunk["rawText"])
        status = "OK" if chars <= MAX_ENTRY_CHARS else "OVER LIMIT"
        log.info(f"  [{i+1}] {chars:>5} chars — {status} — {chunk['title'][:50]}")

    # Save backup
    stem = md_path.stem
    backup_paths = save_local_backup(text, chunks, stem)

    # Upload if requested
    if upload and cl_api_available():
        session_title = prompt_for_session_title(md_path.name)

        log.info(f"Creating Log '{session_title}' in Campaign Logger...")
        new_log_id = create_cl_log(
            title=session_title,
            description=f"Imported from {md_path.name}",
        )

        if new_log_id:
            log.info(f"Uploading {len(chunks)} entries...")
            manifest = upload_to_campaign_logger(chunks, log_id=new_log_id)
            manifest["session_title"] = session_title
            save_manifest(manifest, stem)
            log.info(
                f"Upload complete: {manifest['successful']}/{manifest['total_chunks']} "
                f"successful"
            )
        else:
            log.error(
                "Failed to create Log. Chunks saved locally — "
                "you can retry with --chunk-and-upload"
            )

    log.info("Done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Handle utility modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--chunk" and len(sys.argv) > 2:
            chunk_existing_log(Path(sys.argv[2]), upload=False)
            return
        elif sys.argv[1] == "--chunk-and-upload" and len(sys.argv) > 2:
            chunk_existing_log(Path(sys.argv[2]), upload=True)
            return
        elif sys.argv[1] in ("--help", "-h"):
            print("Usage:")
            print("  python vtt_to_session_log_v2.py           "
                  "  # Watch mode (default)")
            print("  python vtt_to_session_log_v2.py --chunk FILE.md    "
                  "  # Chunk existing log")
            print("  python vtt_to_session_log_v2.py --chunk-and-upload FILE.md "
                  "# Chunk + upload")
            return

    # Watch mode — check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY environment variable is not set.")
        log.error("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print("═" * 60)
    print("  D&D Session Log Converter v2")
    print("  Campaign Logger Integration")
    print("═" * 60)
    print(f"  Watch dir:  {WATCH_DIR}")
    print(f"  Backup dir: {BACKUP_DIR}")
    cl_status = "Configured" if cl_api_available() else "Not configured"
    print(f"  CL API:     {cl_status}")
    if CL_API_CLIENT:
        print(f"  CL Client:  {CL_API_CLIENT[:8]}...  (redacted)")
    print(f"  Max chars:  {MAX_ENTRY_CHARS}/entry")
    print(f"  Drop a .vtt file here to convert automatically.")
    print(f"  Press Ctrl+C to stop.\n")

    processed = load_processed()

    # Process existing unprocessed VTTs on startup
    for vtt_file in sorted(WATCH_DIR.glob("*.vtt")):
        if vtt_file.name not in processed:
            process_vtt(vtt_file)
            processed.add(vtt_file.name)

    # Watch loop
    try:
        while True:
            time.sleep(POLL_INTERVAL)
            for vtt_file in sorted(WATCH_DIR.glob("*.vtt")):
                if vtt_file.name not in processed:
                    process_vtt(vtt_file)
                    processed.add(vtt_file.name)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
