#!/usr/bin/env python3
"""
VTT to D&D Session Log Converter
Watches the script's directory for .vtt files and converts them to
Campaign Logger-formatted Markdown session logs via the Claude API.
"""

import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import anthropic

# ── Configuration ────────────────────────────────────────────────────────────

WATCH_DIR = Path(__file__).parent          # Same folder as this script
OUTPUT_DIR = WATCH_DIR                     # Markdown files land here too
POLL_INTERVAL = 5                          # Seconds between folder scans
PROCESSED_LOG = WATCH_DIR / ".processed_vtts"  # Tracks already-converted files

SYSTEM_PROMPT = """You are a scribe for a tabletop RPG campaign. Your job is to convert raw Zoom meeting transcripts (in VTT format) into polished, readable session logs for a D&D 5e campaign.

CAMPAIGN DETAILS:
- Campaign name: Stonegate (tag as ^"Stonegate" on first mention only)
- System: D&D 5e (gestalt variant — all characters have two classes leveled simultaneously)
- Setting: A homebrew world created by the DM. The primary city where most adventures take place is #"Stonegate", which also lends the campaign its name


TONE & STYLE:
- Casual but immersive — write like a knowledgeable fan recapping a great session, not a formal historian
- Third person, past tense
- Focus exclusively on actual gameplay: combat, exploration, roleplay, decisions, discoveries, and story beats
- Cut all out-of-character chatter, technical issues, crosstalk, side conversations, and anything not relevant to what happened in the game world
- Keep the narrative flowing — don't just list events, connect them with cause and effect

STRUCTURE:
- Use markdown headers to break the session into natural narrative sections (## for major beats)
- No TL;DR or summary section at the top — just dive into the session
- End with a brief "Where We Left Off" section noting the party's current situation

CAMPAIGN LOGGER TAGS:
Apply these tags the FIRST time each entity is mentioned in the log. After the first mention, use the name naturally without the tag.
- Characters & NPCs: @"Full Name"  (e.g. @"Aldric Vane")
- Groups & Organizations (including the party): ^"Group Name"  (e.g. ^"The Hollow Blades", ^"party")
- Locations & Places: #"Place Name"  (e.g. #"Inn of Cerulean Storms", #"Thornwall")
- Monsters & Creatures: &"Creature Type"  (e.g. &"Troll", &"Brigand")

PLAYER & CHARACTER ROSTER:
This is a gestalt campaign — every character has two classes leveled simultaneously. Always use the character name (never the player name) in the session log. Apply the Campaign Logger @"Name" tag on first mention only. Use correct pronouns for each character.

- John Puett is the DM (he/him) — never appears as a character; attribute his narration/rulings as story context only, refer to him as "the Dm"
- Zanery plays @"Kaelen Thray" (he/him) — Wood Elf, Criminal background, Ranger/Monk L2
- Josh plays @"Louis Chasseur du Prime" (he/him) — Fallen Aasimar, Urban Bounty Hunter background, Hexblade Warlock/Fighter L2
- Curtis Jaffe plays @"Courtad the Wolf" (he/him) — Shifter (Wildhunt), Criminal background, Paladin/Barbarian L2
- Hannah plays @"Seraphina Valecrest" (she/her) — Human (Variant), Noble background, Storm Sorcery Sorcerer/Tempest Cleric L2
- Vince plays @"Billy Bob" (he/him) — Halfling, Noble background, Paladin/Wizard L2

Use character class/race details to inform how you describe their actions — a Wildhunt Shifter Paladin/Barbarian fights very differently from a Storm Sorcerer/Tempest Cleric, and that flavor should come through in the narration. If a character was absent from a session, simply omit them from the log naturally.

IMPORTANT:
- Always use character names, not player names, in the log
- Use the correct pronouns for each character as listed above
- If a location, NPC, or monster name is unclear from context, use your best judgment or a reasonable placeholder
- Preserve important dice roll outcomes, critical hits, and dramatic moments — these are highlights
- Do not invent events that didn't happen; stick to what's in the transcript
"""

# ── VTT Parsing ──────────────────────────────────────────────────────────────

def parse_vtt(vtt_text: str) -> str:
    """Strip VTT timing/metadata and return clean speaker: text transcript."""
    lines = vtt_text.splitlines()
    entries = []
    current_speaker = None
    current_text = []

    # Regex to detect VTT timestamp lines like: 00:01:23.456 --> 00:01:25.789
    timestamp_re = re.compile(r"^\d{2}:\d{2}:\d{2}[\.,]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[\.,]\d{3}")
    # Regex to detect cue numbers (lines that are just digits)
    cue_re = re.compile(r"^\d+$")
    # Regex to detect speaker labels like: <v John Smith>text or John Smith: text
    speaker_tag_re = re.compile(r"^<v ([^>]+)>(.*)$")
    speaker_colon_re = re.compile(r"^([A-Za-z][^:]{1,40}):\s+(.+)$")

    def flush():
        if current_speaker and current_text:
            text = " ".join(current_text).strip()
            if text:
                entries.append(f"{current_speaker}: {text}")

    for line in lines:
        line = line.strip()

        # Skip WEBVTT header, NOTE blocks, blank lines, cue numbers, timestamps
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE") or \
           cue_re.match(line) or timestamp_re.match(line):
            continue

        # Remove any inline VTT tags like <c>, <i>, etc.
        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line:
            continue

        # Check for <v Speaker> format
        m = speaker_tag_re.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_text = [m.group(2).strip()] if m.group(2).strip() else []
            continue

        # Check for "Speaker: text" format
        m = speaker_colon_re.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_text = [m.group(2).strip()]
            continue

        # Continuation of current speaker's text
        if current_speaker:
            current_text.append(line)

    flush()

    return "\n".join(entries) if entries else vtt_text  # fallback to raw if parsing fails

# ── Claude API Call ───────────────────────────────────────────────────────────

def convert_to_session_log(transcript: str, filename: str) -> str:
    """Send transcript to Claude and return the formatted session log."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    print(f"  Sending to Claude API...")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is the VTT transcript from file '{filename}'. "
                    "Please convert it into a D&D session log following your instructions.\n\n"
                    f"TRANSCRIPT:\n{transcript}"
                )
            }
        ]
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

def process_vtt(vtt_path: Path):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found: {vtt_path.name}")

    try:
        raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return

    print("  Parsing VTT...")
    transcript = parse_vtt(raw)

    if len(transcript.strip()) < 100:
        print("  WARNING: Transcript seems very short after parsing. Check the VTT format.")

    try:
        session_log = convert_to_session_log(transcript, vtt_path.name)
    except Exception as e:
        print(f"  ERROR calling Claude API: {e}")
        print("  Make sure ANTHROPIC_API_KEY is set in your environment.")
        return

    # Build output filename: same stem, .md extension, with timestamp if collision
    stem = vtt_path.stem
    out_path = OUTPUT_DIR / f"{stem}.md"
    if out_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{stem}_{timestamp}.md"

    try:
        out_path.write_text(session_log, encoding="utf-8")
        print(f"  ✓ Session log saved: {out_path.name}")
    except Exception as e:
        print(f"  ERROR saving output: {e}")
        return

    mark_processed(vtt_path.name)

# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    # Check for API key early
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print(f"D&D Session Log Converter")
    print(f"Watching: {WATCH_DIR}")
    print(f"Drop a .vtt file here and it will be converted automatically.")
    print(f"Press Ctrl+C to stop.\n")

    processed = load_processed()

    # Process any existing unprocessed VTTs on startup
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
