import json
from pathlib import Path

class LanguageManager:
    def __init__(self, default_language="en"):
        self.current_language = default_language
        self.translations = {}
        self.load_language(default_language)

    def load_language(self, language_code) -> bool:
        """load language file"""
        if language_code == "en":
            return True
        try:
            file_path = Path(__file__).parent.parent / f"locales/{language_code}.json"
            with open(file_path, "r", encoding="utf-8") as file:
                self.translations = json.load(file)
            self.current_language = language_code
            return True
        except FileNotFoundError:
            print(f"Language file not found: {language_code}")
            return False

    def _(self, key, default=None) -> str:
        """get translate text"""
        return self.translations.get(key, default if default else key)


# The translator the UI modules share; ``ui.init`` sets it from the chosen
# language, and ``_`` is the identity until then.
_ACTIVE: LanguageManager | None = None

LANGUAGES = {"en": "English", "ru": "Русский"}


def set_language(code: str) -> LanguageManager:
    global _ACTIVE
    _ACTIVE = LanguageManager(code)
    return _ACTIVE


def _(text: str) -> str:
    """Translate via the active LanguageManager; identity when none is set."""
    if _ACTIVE is None:
        return text
    return _ACTIVE._(text)
