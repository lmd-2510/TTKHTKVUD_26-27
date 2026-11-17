"""Build the English data acquisition section of the Oddball report."""
import argparse
import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from download_data import DATASET, REPO, SUBJECTS, selected_paths

INK = colors.HexColor("#172A3D")
TEAL = colors.HexColor("#12645C")
MUTED = colors.HexColor("#42576B")
RULE = colors.HexColor("#D5DFE5")
LIGHT = colors.HexColor("#F0F5F6")
STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=30,
                            textColor=INK, spaceAfter=11),
    "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13, leading=16,
                              textColor=TEAL, spaceBefore=13, spaceAfter=6, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15,
                           textColor=INK, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9.5, leading=13,
                           alignment=TA_LEFT, textColor=INK),
    "thead": ParagraphStyle("thead", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
                            textColor=colors.white),
}

TASKS = [
    ("Oddball", "In scope. Provides frequent standards, rare targets and rare distractors."),
    ("MSIT+", "Out of scope. Studies interference and cognitive control."),
    ("SRT", "Out of scope. Measures simple reaction time in a separate task."),
    ("Resting state", "Out of scope. Records spontaneous activity without the Oddball labels."),
]

FILES = [
    ("*_eeg.eeg", "Continuous binary EEG samples. Preserves the full recorded channel set."),
    ("*_eeg.vhdr", "BrainVision header. Entry point for MNE. Specifies how to decode samples and "
                    "links the signal and marker files."),
    ("*_eeg.vmrk", "BrainVision markers. Stores event annotations at sample positions."),
    ("*_events.tsv", "BIDS event table. Provides onset in seconds, duration and event_type, "
                     "the field used to map stimulus labels."),
    ("*_channels.tsv", "Channel inventory. Describes channel names, types and other metadata "
                       "used to check the input channels."),
    ("task-oddball_eeg.json", "Shared task protocol and acquisition settings, including sampling frequency."),
    ("task-oddball_events.json", "Shared data dictionary for the event table and its codes."),
    ("dataset_description.json", "Dataset identification and provenance."),
]

EVENTS = [
    ("S 5", "Standard / frequent stimulus", "Label 0"),
    ("S 6", "Rare target stimulus", "Label 1"),
    ("S 7", "Rare distractor / deviant stimulus", "Exclude from binary samples"),
    ("S 1", "Recorded button response", "Keep as metadata"),
    ("S 10", "Break marker", "Keep as metadata"),
    ("S 12", "Beginning-of-run marker", "Keep as metadata"),
]

SOURCES = [
    ("Dzianok et al. (2022), GigaScience 11, giac015", "https://doi.org/10.1093/gigascience/giac015"),
    ("OpenNeuro ds004621 dataset", "https://openneuro.org/datasets/ds004621"),
    ("Source file tree and Oddball JSON metadata", "https://github.com/OpenNeuroDatasets/ds004621"),
]


def paragraph(text, style="body"):
    return Paragraph(text, STYLES[style])


def report_table(headers, rows, widths):
    table = Table(
        [[paragraph(escape(label), "thead") for label in headers]]
        + [[paragraph(escape(value), "cell") for value in row] for row in rows],
        colWidths=widths, repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def audit_paragraphs(manifest_path):
    if manifest_path is None:
        return [paragraph("Download inventory pending. File totals, channel counts and event counts "
                          "will be reported from a completed download manifest.")]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != DATASET
        or not manifest.get("complete")
        or not manifest.get("subjects")
        or not set(manifest["subjects"]).issubset(SUBJECTS)
        or set(manifest["subjects"]) != set(manifest["recordings"])
        or sorted(entry["path"] for entry in manifest["files"])
        != sorted(selected_paths(manifest["subjects"]))
    ):
        raise ValueError("Provide a completed Oddball download manifest for this dataset.")
    recordings = manifest["recordings"]
    counts = Counter()
    for recording in recordings.values():
        counts.update(recording["event_counts"])
    channel_counts = [recording["eeg_channels"] for recording in recordings.values()]
    range_label = str(min(channel_counts))
    if min(channel_counts) != max(channel_counts):
        range_label += f" to {max(channel_counts)}"
    total = sum(entry["bytes"] for entry in manifest["files"])
    scope = "full cohort" if len(recordings) == len(SUBJECTS) else "subset"
    event_counts = ", ".join(f"{escape(code)} = {counts[code]:,}" for code, _, _ in EVENTS)
    extra_codes = set(counts) - {code for code, _, _ in EVENTS}
    details = [
        paragraph(f"The manifest covers <b>{len(recordings)} participants ({scope})</b>, "
                  f"{len(manifest['files'])} files and {total / 1024**3:.2f} GiB. "
                  f"EEG channels listed per participant: {range_label}."),
        paragraph(f"Event rows by code before preprocessing: {event_counts}."),
    ]
    if extra_codes:
        extra = ", ".join(f"{escape(code or '(blank)')} = {counts[code]:,}" for code in sorted(extra_codes))
        details.append(paragraph(f"Additional event codes to review: {extra}."))
    return details


def footer(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.line(46, 43, width - 46, 43)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(46, 27, "Nencki-Symfonia EEG/ERP  /  Data acquisition")
    canvas.drawRightString(width - 46, 27, str(document.page))
    canvas.restoreState()


def build(output, manifest_path):
    audit = audit_paragraphs(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, title="Oddball EEG dataset and acquisition",
        author="EEG Oddball project", leftMargin=46, rightMargin=46, topMargin=53, bottomMargin=56,
    )
    story = [
        paragraph("Oddball EEG dataset<br/>and acquisition", "title"),
        paragraph("The Nencki-Symfonia EEG/ERP release (OpenNeuro ds004621) contains EEG recordings "
                  "from 42 healthy adults across three experimental tasks and a resting-state protocol. "
                  "The study used a 128-electrode cap. Oddball metadata specifies a 1,000 Hz sampling rate."),
        paragraph("Recording selection", "heading"),
        report_table(("Recording", "Use in this project"), TASKS, [109, 380]),
        Spacer(1, 7),
        paragraph("One EEG segment aligned to a standard or target onset is a candidate for one "
                  "binary-classification sample. The first experiments will consider all available "
                  "EEG channels. Channel names, types and availability are checked before preprocessing."),
        paragraph("Files acquired", "heading"),
        report_table(("File", "Purpose"), FILES, [156, 333]),
        Spacer(1, 7),
        paragraph("The five recording files are collected per participant. The three JSON files are "
                  "shared. Other task recordings, participant metadata tables and auxiliary study files "
                  "are outside the download scope for this analysis."),
        PageBreak(),
        paragraph("Event codes and inventory", "title"),
        paragraph("The Oddball protocol uses frequent standards and two rarer stimuli. "
                  "The task metadata describes about 76% standards, 12% targets and 12% distractors. "
                  "Participants were instructed to press a button for targets."),
        paragraph("Mapping from raw events to binary labels", "heading"),
        report_table(("Event code", "Meaning", "Use"), EVENTS, [75, 250, 164]),
        Spacer(1, 8),
        paragraph("Use the stimulus onset and event_type in the BIDS event table to assign the "
                  "label. The button response is logged separately. A target onset still has the target "
                  "label if the participant misses the button press. The raw continuous signal and "
                  "event tables retain distractors, responses and recording markers for alignment and checks."),
        paragraph("Download inventory", "heading"),
        *audit,
        paragraph("Counts describe raw event rows. Epoch validation and artifact rejection will "
                  "determine the final numbers of usable samples and participants."),
        paragraph("Sources", "heading"),
    ]
    story.extend(paragraph(f'<link href="{url}" color="#12645C">{escape(label)}</link>')
                 for label, url in SOURCES)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Completed download manifest for inventory totals.")
    parser.add_argument("--output", type=Path, default=REPO / "docs" / "report" / "data_acquisition.pdf")
    args = parser.parse_args()
    try:
        build(args.output, args.manifest)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Report failed: {error}\n")
    print(f"PDF written to {args.output}")


if __name__ == "__main__":
    main()
