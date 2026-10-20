"""Source-level audit against the design restrictions. Visual checks remain manual."""
import re

import pytest

import config

ROOT = config.PROJECT_ROOT
UI_FILES = [
    ROOT / "app.py",
    *sorted((ROOT / "pages").glob("*.py")),
    ROOT / "utils" / "ui.py",
    ROOT / "utils" / "visualizations.py",
    ROOT / ".streamlit" / "config.toml",
]

RULES = [
    ("no gradients", re.compile(r"gradient", re.I)),
    ("no em dashes", re.compile("\u2014")),
    ("no glass effects", re.compile(r"backdrop-filter|blur\(", re.I)),
    ("no colored side borders", re.compile(r"border-(left|inline-start)", re.I)),
    ("no banned fonts", re.compile(r"\bInter\b|Space Grotesk|Instrument Serif")),
    ("no italics", re.compile(r"font-style\s*:\s*italic|<em>|<i>", re.I)),
    ("no buzzwords", re.compile(r"revolutionary|cutting-edge|next-generation|unlock the power", re.I)),
    ("no scroll or fade animation", re.compile(r"@keyframes|animation\s*:|IntersectionObserver|\bfade", re.I)),
    ("no cursor effects", re.compile(r"mousemove|<script", re.I)),
    ("no hover opacity changes", re.compile(r":hover[^}]*opacity", re.I)),
    ("no noise textures", re.compile(r"feTurbulence|grain", re.I)),
    ("no icon libraries", re.compile(r"lucide", re.I)),
    ("no decorative Streamlit effects", re.compile(r"st\.(balloons|snow)")),
]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2B55\uFE0F]")


def _read(path):
    return path.read_text(encoding="utf-8")


def test_ui_files_exist():
    missing = [str(p.relative_to(ROOT)) for p in UI_FILES if not p.exists()]
    assert not missing, f"Missing UI files: {missing}"


@pytest.mark.parametrize("name,pattern", RULES, ids=[r[0] for r in RULES])
def test_design_rule(name, pattern):
    offenders = [str(p.relative_to(ROOT)) for p in UI_FILES if p.exists() and pattern.search(_read(p))]
    assert not offenders, f"{name}: found in {offenders}"


def test_no_emojis():
    offenders = [str(p.relative_to(ROOT)) for p in UI_FILES if p.exists() and EMOJI.search(_read(p))]
    assert not offenders, f"emoji found in {offenders}"


def test_border_radii_use_the_two_allowed_values():
    radii = {r.strip() for r in re.findall(r"border-radius\s*:\s*([^;}]+)", _read(ROOT / "utils" / "ui.py"))}
    assert radii <= {"6px", "8px"}, radii


def test_theme_is_light():
    assert 'base = "light"' in _read(ROOT / ".streamlit" / "config.toml")