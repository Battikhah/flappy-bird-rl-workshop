"""Original pixel artwork, built once per display; no simulation dependencies."""
from pathlib import Path

import pygame

INK = (30, 53, 68)
CREAM = (255, 246, 209)
# A deliberately small original bitmap alphabet keeps projection text crisp.
GLYPHS = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-./ ', (
'010101111101101','110101110101110','011100100100011','110101101101110',
'111100110100111','111100110100100','011100101101011','101101111101101',
'111010010010111','001001001101010','101101110101101','100100100100111',
'101111111101101','101111111111101','010101101101010','110101110100100',
'010101101111011','110101110101101','011100010001110','111010010010010',
'101101101101111','101101101101010','101101111111101','101101010101101',
'101101010010010','111001010100111','111101101101111','010110010010111',
'110001111100111','110001010001110','101101111001001','111100110001110',
'011100111101111','111001010010010','111101111101111','111101111001110',
'000010000010000','000000111000000','000000000000010','001001010100100',
'000000000000000')))


def text(label, scale=2, color=CREAM):
    """Rasterize cached caller labels with a consistent three-by-five pixel grid."""
    label = str(label).upper()
    surface = pygame.Surface((max(1, len(label) * 4 * scale), 5 * scale), pygame.SRCALPHA)
    for i, char in enumerate(label):
        for pixel, value in enumerate(GLYPHS.get(char, GLYPHS[' '])):
            if value == '1':
                pygame.draw.rect(surface, color, (i * 4 * scale + pixel % 3 * scale,
                                                  pixel // 3 * scale, scale, scale))
    return surface


def bird_frames(radius):
    """Fit original sprites inside the unchanged square bird geometry."""
    rows = (Path(__file__).resolve().parents[1] / 'assets/bird.txt').read_text().splitlines()
    palette = {'O': INK, 'Y': (255, 199, 57), 'H': (255, 233, 130),
               'W': CREAM, 'K': INK, 'B': (242, 111, 51), 'L': (217, 139, 40)}
    frames = []
    for wing in (-2, 0, 2):
        sprite = pygame.Surface((20, 12), pygame.SRCALPHA)
        for y, row in enumerate(rows):
            for x, char in enumerate(row):
                if char in palette:
                    sprite.set_at((x, y), palette[char])
        pygame.draw.polygon(sprite, INK, [(3, 8), (7, 6 + wing), (10, 8 + wing), (8, 11), (4, 11)])
        pygame.draw.polygon(sprite, (255, 220, 100), [(4, 8), (7, 7 + wing), (9, 8 + wing), (7, 10), (4, 10)])
        frames.append(pygame.transform.scale(sprite, (2 * radius, 2 * radius)))
    return frames


def pipe_tile(width, height, cap=False):
    """Keep every shaded pipe pixel within its exact collision rectangle."""
    tile = pygame.Surface((width, height))
    tile.fill(INK)
    for left, right, color in ((3, width - 3, (64, 151, 69)), (6, width - 8, (107, 193, 75)),
                               (9, 16, (190, 231, 110)), (19, 23, (145, 210, 86)),
                               (width - 12, width - 4, (43, 117, 65))):
        pygame.draw.rect(tile, color, (left, 2 if cap else 0, right-left, height - (4 if cap else 0)))
    if cap:
        pygame.draw.line(tile, (217, 244, 143), (5, 3), (width - 6, 3), 2)
    return tile
