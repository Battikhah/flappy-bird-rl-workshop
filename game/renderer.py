"""Cosmetic arcade presentation; simulation ticks are the only animation clock."""
import os
from functools import lru_cache

import pygame

from .art import CREAM, INK, bird_frames, pipe_tile, text


class Renderer:
    """Draw snapshots without changing physics, RNG, policies, or collision boxes."""

    def __init__(self, config):
        """Cache pixel art and provide optional integer-scaled letterboxed output."""
        pygame.display.init()
        self.config = config
        self.surface = pygame.Surface((config.width, config.height))
        self.presentation = pygame.Surface((config.width, config.height + 28))
        scale = max(1, min(4, int(os.environ.get('FLAPPY_SCALE', '1'))))
        fullscreen = os.environ.get('FLAPPY_FULLSCREEN') == '1'
        self.window = pygame.display.set_mode((0, 0) if fullscreen else
            (config.width * scale, (config.height + 28) * scale), pygame.FULLSCREEN if fullscreen else 0)
        pygame.display.set_caption('Flappy RL | Arcade')
        self.mode, self.detail = 'HUMAN', ''
        self.birds = bird_frames(config.bird_radius)
        self.rotations = {}
        for frame, sprite in enumerate(self.birds):
            for angle in range(-75, 31, 5):
                rotated = pygame.transform.rotate(sprite, angle)
                factor = 2 * config.bird_radius / max(rotated.get_size())
                self.rotations[frame, angle] = pygame.transform.scale(rotated,
                    (max(1, round(rotated.get_width()*factor)), max(1, round(rotated.get_height()*factor))))
        self.stem = pipe_tile(config.pipe_width, config.height)
        self.cap = pipe_tile(config.pipe_width, 20, True)
        self.sky = self._landscape()
        self.clouds = pygame.Surface((180, 50), pygame.SRCALPHA)
        for rect in ((10, 25, 150, 16), (28, 14, 110, 25), (48, 5, 54, 30)):
            pygame.draw.rect(self.clouds, (231, 249, 230), rect)
        self.ground = pygame.Surface((64, 28))
        self.ground.fill((197, 149, 89))
        pygame.draw.rect(self.ground, INK, (0, 0, 64, 3))
        pygame.draw.rect(self.ground, (112, 188, 88), (0, 3, 64, 6))
        pygame.draw.line(self.ground, (208, 233, 119), (0, 3), (63, 3), 2)
        for x in range(0, 64, 16):
            pygame.draw.polygon(self.ground, (236, 194, 120), [(x, 12), (x+8, 12), (x+2, 18), (x, 18)])
            pygame.draw.rect(self.ground, (166, 118, 75), (x+8, 23, 5, 2))

    def _landscape(self):
        """Build a quiet skyline behind the playable area once per window."""
        c = self.config
        sky = pygame.Surface((c.width, c.height))
        sky.fill((100, 194, 202))
        pygame.draw.rect(sky, (135, 211, 207), (0, 240, c.width, c.height-240))
        pygame.draw.rect(sky, (172, 225, 208), (0, 400, c.width, c.height-400))
        return sky

    @staticmethod
    @lru_cache(maxsize=256)
    def label(value, scale=2, color=CREAM):
        """Cache changing HUD strings with a bounded surface budget."""
        return text(value, scale, color)

    def _center(self, value, y, scale=2, color=CREAM):
        """Place bitmap labels without introducing font or platform dependencies."""
        label = self.label(value, scale, color)
        self.surface.blit(label, ((self.config.width-label.get_width())//2, y))

    def draw(self, env):
        """Return a native-size frame; repeated snapshots produce identical pixels."""
        c, s = self.config, self.surface
        distance = env._scroll_distance  # Cosmetic history, never fed to an agent.
        s.blit(self.sky, (0, 0))
        for i in range(4):
            x = (i * 190 - int(distance * .10)) % (c.width + 180) - 180
            s.blit(self.clouds, (x, 98 + i % 2 * 60))
        for i in range(8):
            x = (i * 92 - int(distance * .20)) % (c.width + 100) - 100
            y = 456 + (i % 3) * 16
            pygame.draw.rect(s, (124, 192, 175), (x, y, 64, c.height-y))
            pygame.draw.rect(s, (178, 221, 191), (x+6, y+8, 5, 8))
        for i in range(6):
            x = (i * 130 - int(distance * .35)) % (c.width+160) - 160
            pygame.draw.polygon(s, (83, 169, 137), [(x, c.height), (x, 552), (x+28, 528),
                (x+50, 528), (x+76, 548), (x+120, 548), (x+160, c.height)])
        for pipe in env.pipes:
            x = round(pipe.x)
            top = round(pipe.gap_center-c.gap_height/2)
            bottom = round(pipe.gap_center+c.gap_height/2)
            s.blit(self.stem, (x, 0), (0, 0, c.pipe_width, top))
            s.blit(self.cap, (x, top-20))
            s.blit(self.stem, (x, bottom), (0, 0, c.pipe_width, c.height-bottom))
            s.blit(self.cap, (x, bottom))
        # The ground's top edge is the actual world boundary; texture stays inside.
        offset = int(distance) % 64
        for x in range(-64, c.width+64, 64):
            s.blit(self.ground, (x-offset, c.height-3))
        pygame.draw.line(s, INK, (0, 0), (c.width, 0), 2)
        frame = (env.steps // 5) % 3
        angle = max(-75, min(30, round(-env.bird.velocity * 5 / 5) * 5))
        if env.done:
            angle = -75
        sprite = self.rotations[frame, angle]
        # Rotation is confined to the same visual square, never the hitbox.
        s.blit(sprite, sprite.get_rect(center=(c.bird_x, round(env.bird.y))))
        pygame.draw.rect(s, INK, (12, 12, c.width-24, 48), border_radius=4)
        s.blit(self.label(self.mode, 2, (157, 223, 181)), (24, 22))
        s.blit(self.label(f'SPEED {env.pipe_speed:g}', 2), (c.width-140, 22))
        recent = any(pipe.passed and pipe.x + c.pipe_width > c.bird_x-c.bird_radius-36
                     for pipe in env.pipes)
        self._center(str(env.score), 20 if recent else 22, 4,
                     (255, 216, 104) if recent else CREAM)
        if self.detail:
            self._center(self.detail[:52], 46, 1, (157, 223, 181))
        if env.steps == 0:
            self._panel('READY TO FLY', 'SPACE FLAP / ESC EXIT' if self.mode == 'HUMAN' else 'Q-TABLE PILOT READY')
        elif env.done:
            self._panel('FLIGHT COMPLETE' if env.score >= c.max_pipes else 'GAME OVER', f'{env.score} PIPES PASSED')
            self._center('RUN COMMAND AGAIN TO RESTART', 354, 2, INK)
        else:
            self._center('SPACE FLAP / ESC EXIT' if self.mode == 'HUMAN' else 'AI PILOT / ESC EXIT', c.height-24, 1, INK)
        self.window.fill(INK)
        width, height = self.window.get_size()
        scale = max(1, min(width//c.width, height//(c.height+28)))
        self.presentation.blit(s, (0, 0))
        # The textured floor lives BELOW y=height; the playable world stays intact.
        for x in range(-64, c.width+64, 64):
            self.presentation.blit(self.ground, (x-offset, c.height))
        size = (c.width*scale, (c.height+28)*scale)
        self.window.blit(pygame.transform.scale(self.presentation, size), ((width-size[0])//2, (height-size[1])//2))
        pygame.display.flip()
        return s

    def _panel(self, heading, subtitle):
        """Use small arcade cards only at existing ready and terminal snapshots."""
        pygame.draw.rect(self.surface, INK, (44, 258, self.config.width-88, 80))
        pygame.draw.rect(self.surface, CREAM, (48, 262, self.config.width-96, 72))
        self._center(heading, 276, 3, INK)
        self._center(subtitle, 310, 2, (61, 126, 104))

    def close(self):
        """Release the display while leaving the simulation independently usable."""
        self.label.cache_clear()
        pygame.display.quit()
