"""QMaxent internationalisation (i18n) module.

Usage anywhere in the plugin:
    from .i18n import tr
    label = tr("Presence Points")
"""

from .translator import tr, tooltip, set_locale, current_locale

__all__ = ["tr", "tooltip", "set_locale", "current_locale"]
