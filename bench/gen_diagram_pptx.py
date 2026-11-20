"""Generate WanDog architecture diagram as an editable PowerPoint.

Every box, zone, arrow, and label is a separate shape so you can drag them.

Run:  .venv/bin/python bench/gen_diagram_pptx.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

OUT = Path(__file__).resolve().parent.parent / "ME 3 Deck v2" / "diagram.pptx"

# ── colours ──────────────────────────────────────────────────────────────
BG       = RGBColor(0xFF, 0xFF, 0xFF)
PI_FILL  = RGBColor(0xEE, 0xF4, 0xFB)
PI_EDGE  = RGBColor(0x4A, 0x7F, 0xB5)
CLOUD_FILL = RGBColor(0xFD, 0xF1, 0xE7)
CLOUD_EDGE = RGBColor(0xC8, 0x79, 0x2F)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
TXT      = RGBColor(0x16, 0x20, 0x2B)
MUTED    = RGBColor(0x5B, 0x6B, 0x7C)
ACCENT   = RGBColor(0xB3, 0x40, 0x2F)
TOOL_FILL= RGBColor(0xF5, 0xF2, 0xFB)
TOOL_EDGE= RGBColor(0x6B, 0x5B, 0x95)

# ── helpers ──────────────────────────────────────────────────────────────
SLIDE_W, SLIDE_H = Inches(18.0), Inches(13.5)

def px(x):  # convert inch to Emu
    return Inches(x)

def add_zone(slide, x, y, w, h, fill, edge, label="", label_fs=18, label_color=None):
    """Rounded rectangle zone."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, px(x), px(y), px(w), px(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge
    shape.line.width = Pt(2)
    if label:
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(label_fs)
        p.font.bold = True
        p.font.color.rgb = label_color or TXT
        p.alignment = PP_ALIGN.CENTER
    return shape

def add_box(slide, x, y, w, h, title, subtitle, fill=WHITE, edge=None, title_fs=11, sub_fs=9):
    """Box with title line + subtitle lines."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, px(x), px(y), px(w), px(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge or STAGE_EDGE
    shape.line.width = Pt(1.2)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.1)
    # title
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(title_fs)
    p.font.bold = True
    p.font.color.rgb = TXT
    p.alignment = PP_ALIGN.LEFT
    # subtitle lines
    for line in subtitle.strip().split("\n"):
        p2 = tf.add_paragraph()
        p2.text = line.strip()
        p2.font.size = Pt(sub_fs)
        p2.font.color.rgb = MUTED
        p2.alignment = PP_ALIGN.LEFT
    return shape

def add_label(slide, x, y, text, fs=9, color=TXT, bold=False):
    """Floating text label."""
    shape = slide.shapes.add_textbox(px(x), px(y), px(12), px(0.6))
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(fs)
    p.font.color.rgb = color
    p.font.bold = bold
    return shape

def add_arrow(slide, x1, y1, x2, y2, color=None, width=1.5, dashed=False):
    """Straight arrow connector."""
    connector = slide.shapes.add_connector(
        1,  # straight
        px(x1), px(y1), px(x2), px(y2))
    connector.line.color.rgb = color or TXT
    connector.line.width = Pt(width)
    if dashed:
        connector.line.dash_style = 4  # dash
    return connector

STAGE_EDGE = RGBColor(0x3D, 0x5A, 0x80)

# ── build ────────────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

# Title
add_label(slide, 1.0, 0.4, "WanDog — Audio & Tool Path", fs=22, bold=True, color=TXT)

# ── Pi zone ──────────────────────────────────────────────────────────────
PI_X, PI_Y, PI_W, PI_H = 2.5, 2.0, 69.0, 10.5
add_zone(slide, PI_X, PI_Y, PI_W, PI_H, PI_FILL, PI_EDGE,
         "Raspberry Pi 5 (4 GB) — WanDog Agent", label_fs=16, label_color=PI_EDGE)

# ── Cloud zone ───────────────────────────────────────────────────────────
CLOUD_X, CLOUD_Y, CLOUD_W, CLOUD_H = 73.2, 2.0, 24.1, 10.5
add_zone(slide, CLOUD_X, CLOUD_Y, CLOUD_W, CLOUD_H, CLOUD_FILL, CLOUD_EDGE,
         "Cloud / Internet", label_fs=16, label_color=CLOUD_EDGE)

# ── Row 1: Audio input path ──────────────────────────────────────────────
R1Y = 4.5
R1H = 2.8

add_box(slide, 6.0, R1Y, 14.0, R1H, "Microphone",
        "USB Composite\n48 kHz mono", fill=WHITE)
add_box(slide, 22.5, R1Y, 14.0, R1H, "VAD",
        "SileroVADAnalyzer\nstreaming · 30 ms frames")
add_box(slide, 39.0, R1Y, 14.0, R1H, "ASR",
        "Sherpa-ONNX Streaming\nstreaming · 16 kHz in")
add_box(slide, 55.5, R1Y, 14.0, R1H, "LLM",
        "Ollama · Llama 3.1 8B\ntemp 0.7 · 300 tok\nthinking off", fill=WHITE)

# Row 1 arrows
add_arrow(slide, 20.0, R1Y + R1H/2, 22.5, R1Y + R1H/2)
add_arrow(slide, 36.5, R1Y + R1H/2, 39.0, R1Y + R1H/2)
add_arrow(slide, 53.0, R1Y + R1H/2, 55.5, R1Y + R1H/2)

# ── Row 2: Audio output path ────────────────────────────────────────────
R2Y = 8.0
R2H = 2.8

add_box(slide, 6.0, R2Y, 14.0, R2H, "Turn controller",
        "UserTurnController\nstart: VADUserTurnStart\nstop: SpeechTimeout 1.0 s", fill=WHITE)
add_box(slide, 22.5, R2Y, 14.0, R2H, "LLM client",
        "ClockedLLMService\ntemp 0.7 · 300 tok\nthinking off", fill=WHITE)
add_box(slide, 39.0, R2Y, 14.0, R2H, "TTS",
        "Piper\nen_US-ryan-medium\n48 kHz out")
add_box(slide, 55.5, R2Y, 14.0, R2H, "Speaker",
        "USB Composite\n48 kHz mono", fill=WHITE)

# Row 2 arrows
add_arrow(slide, 20.0, R2Y + R2H/2, 22.5, R2Y + R2H/2)
add_arrow(slide, 36.5, R2Y + R2H/2, 39.0, R2Y + R2H/2)
add_arrow(slide, 53.0, R2Y + R2H/2, 55.5, R2Y + R2H/2)

# Transcript arrow: ASR -> turn controller (diagonal)
add_arrow(slide, 53.0, R1Y, 20.0, R2Y + R2H, color=MUTED, width=1.2)
add_label(slide, 28.0, 6.3, "final transcript (text, not audio)", fs=8.5, color=MUTED)

# ── LLM client -> Cloud Ollama (elbow arrow) ────────────────────────────
# Up from LLM client, right, into cloud
add_arrow(slide, 29.5, R2Y, 29.5, R2Y - 1.5, color=CLOUD_EDGE, width=2.0)
add_arrow(slide, 29.5, R2Y - 1.5, CLOUD_X - 0.5, R2Y - 1.5, color=CLOUD_EDGE, width=2.0)
add_arrow(slide, CLOUD_X - 0.5, R2Y - 1.5, CLOUD_X - 0.5, 5.5, color=CLOUD_EDGE, width=2.0)
add_label(slide, 36.0, R2Y - 2.5, "prompt + context out   ·   tokens back", fs=8.5, color=CLOUD_EDGE)

# ── Cloud: Ollama API ───────────────────────────────────────────────────
add_box(slide, CLOUD_X + 2.5, 4.5, 20.0, 2.8, "Ollama cloud API",
        "https://ollama.com/v1\nOpenAI-compatible · streaming",
        fill=CLOUD_FILL, edge=CLOUD_EDGE)

# ── Cloud: Third-party APIs ─────────────────────────────────────────────
add_box(slide, CLOUD_X + 2.5, 8.0, 20.0, 2.5, "Third-party APIs",
        "web search · page fetch\nweather · GitHub REST",
        fill=CLOUD_FILL, edge=CLOUD_EDGE, title_fs=10.5)

# ── MCP tool path (bottom-left of Pi) ───────────────────────────────────
MX, MY, MW, MH = 3.0, 13.5, 28.0, 11.0
add_zone(slide, MX, MY, MW, MH, TOOL_FILL, TOOL_EDGE,
         "MCP Tool Path (on Pi)", label_fs=12, label_color=TOOL_EDGE)

# MCP boxes inside the MCP zone
add_box(slide, MX + 1.5, MY + 1.5, 11.0, 3.5, "ToolsLocalMCPServer",
        "stdio subprocess\nread_file · grep · bash\nlist_dir · edit_file\nsearch_document",
        fill=WHITE, edge=TOOL_EDGE, title_fs=9, sub_fs=7.5)

add_box(slide, MX + 15.0, MY + 1.5, 11.0, 3.5, "ToolsNetMCPServer",
        "stdio subprocess\nsearch · get_weather\nfetch_content",
        fill=WHITE, edge=TOOL_EDGE, title_fs=9, sub_fs=7.5)

# MCP servers -> Third-party APIs (crossing from Pi to Cloud, dashed)
add_arrow(slide, MX + MW, MY + MH/2, CLOUD_X, MY + MH/2, color=CLOUD_EDGE, width=1.8, dashed=True)
add_label(slide, MX + MW + 2, MY + MH/2 + 0.5, "tool servers call cloud APIs", fs=8.2, color=CLOUD_EDGE)

# ── Barge-in arrow ──────────────────────────────────────────────────────
add_arrow(slide, 55.5, R1Y + R1H, 55.5, R2Y, color=ACCENT, width=2.0)
add_label(slide, 56.0, 7.3, "barge-in", fs=9, color=ACCENT, bold=True)

# ── Footer notes ─────────────────────────────────────────────────────────
add_label(slide, 2.5, 12.8, "· No AEC — single speaker · No mute — always listening", fs=8, color=MUTED)
add_label(slide, 2.5, 13.3, "· VAD start → ASR start · VAD stop → ASR flush → transcript → LLM", fs=8, color=MUTED)

# ── Save ─────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
