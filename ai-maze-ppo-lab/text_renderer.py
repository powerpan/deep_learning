"""Text drawing with Chinese font support and a bitmap fallback."""

from pathlib import Path

import pygame

from pixel_text import draw_text as draw_pixel_text
from pixel_text import draw_text_centered as draw_pixel_text_centered


FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


class TextRenderer:
    """Use pygame.font when available; fall back to ASCII bitmap text."""

    def __init__(self):
        self.available = False
        self.font_path = self._find_font_path()
        self.cache = {}

        try:
            pygame.font.init()
            if self.font_path:
                pygame.font.Font(str(self.font_path), 18)
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = exc

    def draw(self, surface, text, x, y, color, size=18):
        if self.available:
            font = self._font(size)
            image = font.render(str(text), True, color)
            surface.blit(image, (int(x), int(y)))
            return

        draw_pixel_text(surface, text, x, y, color, scale=max(1, size // 8))

    def draw_centered(self, surface, text, center, color, size=18):
        if self.available:
            font = self._font(size)
            image = font.render(str(text), True, color)
            surface.blit(image, image.get_rect(center=center))
            return

        draw_pixel_text_centered(surface, text, center, color, scale=max(1, size // 8))

    def _font(self, size):
        key = int(size)
        if key not in self.cache:
            if self.font_path:
                self.cache[key] = pygame.font.Font(str(self.font_path), key)
            else:
                self.cache[key] = pygame.font.Font(None, key)
        return self.cache[key]

    def _find_font_path(self):
        for candidate in FONT_CANDIDATES:
            path = Path(candidate)
            if path.exists():
                return path
        return None
