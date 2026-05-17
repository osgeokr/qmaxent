"""QMaxent internationalisation (i18n) module.

Usage anywhere in the plugin:
    from .i18n import tr
    label = tr("Presence Points")
"""

from .translator import current_locale, set_locale, tooltip, tr

__all__ = ["tr", "tooltip", "set_locale", "current_locale"]
