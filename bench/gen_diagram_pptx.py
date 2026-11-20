"""Generate WanDog architecture diagram as an editable PowerPoint.

Every box, zone, arrow, and label is a separate shape so you can drag them.
Boxes are sized to fit their text — no more oversized rectangles.

Run:  .venv/bin/python bench/gen_diagram_pptx.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

OUT = Path(__file__).resolve().parent.parent / "ME 3 Deck v2" / "diagram.pptx"

# ── colours ──────────────────────────────────────────────────────
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
STAGE_EDGE = RGBColor(0x3D, 0x5A, 0x80)

# ── helpers ──────────────────────────────────────────────────────
SLIDE_W, SLIDE_H = Inches(14.1), Inches(10.6)

def px(x):
    return Inches(x)

def add_zone(slide, x, y, w, h, fill, edge, label="", label_fs=16, label_color=None):
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
    """Box with title line + subtitle lines. w and h are in inches."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, px(x), px(y), px(w), px(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge or STAGE_EDGE
    shape.line.width = Pt(1.2)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.10)
    tf.margin_right = Inches(0.10)
    tf.margin_top = Inches(0.06)
    tf.margin_bottom = Inches(0.06)
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
    shape = slide.shapes.add_textbox(px(x), px(y), px(12), px(0.5))
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
        connector.line.dash_style = 4
    return connector

# ── build ────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

# Title
add_label(slide, 0.5, 0.3, "WanDog — Audio & Tool Path", fs=20, bold=True, color=TXT)

# ── Pi zone ──────────────────────────────────────────────────────
PI_X, PI_Y, PI_W, PI_H = 1.0, 1.2, 10.5, 8.8
add_zone(slide, PI_X, PI_Y, PI_W, PI_H, PI_FILL, PI_EDGE,
         "Raspberry Pi 5 (4 GB) — WanDog Agent", label_fs=14, label_color=PI_EDGE)

# ── Cloud zone ───────────────────────────────────────────────────
CLOUD_X, CLOUD_Y, CLOUD_W, CLOUD_H = 12.0, 1.2, 2.1, 8.8
add_zone(slide, CLOUD_X, CLOUD_Y, CLOUD_W, CLOUD_H, CLOUD_FILL, CLOUD_EDGE,
         "Cloud / Internet", label_fs=12, label_color=CLOUD_EDGE)

# ── Row 1: Audio input path ──────────────────────────────────────
# Compact box sizes: ~1.6" wide × 0.8" tall
R1Y = 2.5
R1H = 0.8
BOX_W = 1.6
gap = 0.25
start_x = 2.0

add_box(slide, start_x, R1Y, BOX_W, R1H, "Microphone",
        "USB Composite\n48 kHz mono", fill=WHITE)
add_box(slide, start_x + BOX_W + gap, R1Y, BOX_W, R1H, "VAD",
        "SileroVADAnalyzer\nstreaming · 30 ms")
add_box(slide, start_x + 2*(BOX_W + gap), R1Y, BOX_W, R1H, "ASR",
        "Sherpa-ONNX Streaming\n16 kHz in")
add_box(slide, start_x + 3*(BOX_W + gap), R1Y, BOX_W, R1H, "LLM",
        "Ollama · Llama 3.1 8B\ntemp 0.7 · 300 tok", fill=WHITE)

# Row 1 arrows
for i in range(4):
    x = start_x + i*(BOX_W + gap)
    add_arrow(slide, x + BOX_W, R1Y + R1H/2, x + BOX_W + gap, R1Y + R1H/2)

# ── Row 2: Audio output path ─────────────────────────────────────
R2Y = 3.8
R2H = 0.8

add_box(slide, start_x, R2Y, BOX_W, R2H, "Turn controller",
        "VADUserTurnStart\nSpeechTimeout 1.0 s", fill=WHITE)
add_box(slide, start_x + BOX_W + gap, R2Y, BOX_W, R2H, "LLM client",
        "ClockedLLMService\ntemp 0.7 · 300 tok", fill=WHITE)
add_box(slide, start_x + 2*(BOX_W + gap), R2Y, BOX_W, R2H, "TTS",
        "Piper en_US-ryan-medium\n48 kHz out")
add_box(slide, start_x + 3*(BOX_W + gap), R2Y, BOX_W, R2H, "Speaker",
        "USB Composite\n48 kHz mono", fill=WHITE)

# Row 2 arrows
for i in range(4):
    x = start_x + i*(BOX_W + gap)
    add_arrow(slide, x + BOX_W, R2Y + R2H/2, x + BOX_W + gap, R2Y + R2H/2)

# transcript: ASR -> turn controller (diagonal)
asr_x = start_x + 2*(BOX_W + gap)
tc_x = start_x
add_arrow(slide, asr_x + BOX_W, R1Y, tc_x + BOX_W/2, R2Y + R2H)
add_label(slide, 3.5, 3.1, "final transcript (text, not audio)", fs=8, color=PI_EDGE)

# ── LLM client -> cloud (elbow right) ────────────────────────────
llm_client_x = start_x + (BOX_W + gap)
# Horizontal from LLM client right edge to cloud left edge
add_arrow(slide, llm_client_x + BOX_W, R2Y + R2H/2, CLOUD_X, R2Y + R2H/2,
          color=CLOUD_EDGE, width=1.8)
add_label(slide, 6.5, R2Y + R2H/2 + 0.2, "prompt + context out   ·   tokens back",
          fs=8, color=CLOUD_EDGE)

# ── Cloud boxes ──────────────────────────────────────────────────
add_box(slide, CLOUD_X + 0.15, R1Y, 1.8, 0.8, "Ollama API",
        "https://ollama.com/v1\nOpenAI-compatible",
        fill=CLOUD_FILL, edge=CLOUD_EDGE, title_fs=9, sub_fs=7.5)

add_box(slide, CLOUD_X + 0.15, R2Y + 1.5, 1.8, 0.8, "3rd-party APIs",
        "web search · weather\nGitHub REST",
        fill=CLOUD_FILL, edge=CLOUD_EDGE, title_fs=9, sub_fs=7.5)

# ── MCP tool path (bottom-left of Pi) ────────────────────────────
MX, MY, MW, MH = 1.5, 5.5, 8.5, 4.0
add_zone(slide, MX, MY, MW, MH, TOOL_FILL, TOOL_EDGE,
         "MCP Tool Path (on Pi)", label_fs=11, label_color=TOOL_EDGE)

# MCP boxes inside the MCP zone — compact
add_box(slide, MX + 0.3, MY + 1.0, 3.6, 1.8, "ToolsLocalMCPServer",
        "stdio subprocess\nread_file · grep · bash\nlist_dir · edit_file\nsearch_document",
        fill=WHITE, edge=TOOL_EDGE, title_fs=9, sub_fs=7.5)

add_box(slide, MX + 4.3, MY + 1.0, 3.6, 1.8, "ToolsNetMCPServer",
        "stdio subprocess\nsearch · get_weather\nfetch_content",
        fill=WHITE, edge=TOOL_EDGE, title_fs=9, sub_fs=7.5)

# MCP servers -> Third-party APIs (crossing from Pi to Cloud, dashed)
add_arrow(slide, MX + MW, MY + MH/2, CLOUD_X, MY + MH/2,
          color=CLOUD_EDGE, width=1.8, dashed=True)
add_label(slide, MX + MW + 0.3, MY + MH/2 + 0.2, "tool servers call cloud APIs",
          fs=7.5, color=CLOUD_EDGE)

add_label(slide, CLOUD_X + 0.15, R2Y + 1.5 - 0.3, "called by the tool servers",
          fs=7.5, color=CLOUD_EDGE)

# ── Annotations ──────────────────────────────────────────────────
add_label(slide, PI_X + 0.3, PI_Y + PI_H - 0.4,
          "⚠ No AEC  ·  No mic mute during TTS  ·  VAD = barge-in only",
          fs=9, color=ACCENT, bold=True)

# ── Save ─────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"Saved {OUT}  ({OUT.stat().st_size} bytes)")
