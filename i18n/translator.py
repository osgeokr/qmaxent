"""Locale detection and translation lookup for QMaxent.

Detects the QGIS UI language at import time and exposes a single tr()
function that returns the localised string, falling back to the English
key if no translation is found.

Supported locales:  ko (한국어),  en (English, default)
Adding a new language: create i18n/lang_XX.py with a STRINGS dict.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Locale detection
# ---------------------------------------------------------------------------

def _detect_locale() -> str:
    """Return a two-letter locale code from QGIS settings."""
    try:
        from qgis.core import QgsApplication
        locale = QgsApplication.locale()        # e.g. "ko", "en_US", "de"
        return locale[:2].lower()
    except Exception:
        import locale as _loc
        code = _loc.getdefaultlocale()[0] or "en"
        return code[:2].lower()


_LOCALE: str = _detect_locale()


def current_locale() -> str:
    return _LOCALE


def set_locale(code: str) -> None:
    """Override locale at runtime (useful for testing)."""
    global _LOCALE
    _LOCALE = code[:2].lower()
    _reload_strings()


# ---------------------------------------------------------------------------
# String table loading
# ---------------------------------------------------------------------------

_STRINGS: dict[str, str] = {}


def _reload_strings() -> None:
    global _STRINGS
    try:
        if _LOCALE == "ko":
            from .lang_ko import STRINGS
        else:
            STRINGS = {}          # English = use key as-is
        _STRINGS = STRINGS
    except ImportError:
        _STRINGS = {}


_reload_strings()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tr(text: str) -> str:
    """Return the localised version of *text*, or *text* itself if not found."""
    return _STRINGS.get(text, text)


def tooltip(text: str) -> str:
    """Wrap a tooltip string so Qt always word-wraps it.

    Qt's QToolTip word-wraps only when the input is rich text. Plain-text
    tooltips render on a single line that can run far past the screen
    edge, while strings that happen to look "rich-text-ish" (parentheses,
    semicolons, periods) wrap normally — which is why our tooltips were
    rendering inconsistently across widgets. Wrapping every tooltip in
    <qt>…</qt> tells Qt unambiguously that the content is rich text and
    should be wrapped to fit the screen, regardless of its inner
    punctuation. We pair it with explicit translation so callers write
    ``tooltip(tr("…"))`` in a single, uniform pattern.

    See https://doc.qt.io/qt-6/qtooltip.html — "Rich text displayed in a
    tool tip is implicitly word-wrapped".
    """
    return f"<qt>{text}</qt>"
