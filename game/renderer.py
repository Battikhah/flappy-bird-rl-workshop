"""Asset-free visualization; this module is imported only when render is called."""

import pygame


class Renderer:
    """Draw snapshots without owning physics, input, randomness, or pacing."""

    def __init__(self, config):
        """Initialize just the display and font subsystems, avoiding audio devices."""
        pygame.display.init()
        pygame.font.init()
        self.config = config
        self.surface = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption('Flappy RL — Phase 1')
        self.font = pygame.font.Font(None, 28)

    def draw(self, env):
        """Return the displayed frame; coordinates use the simulation's pixel units."""
        c = self.config
        self.surface.fill((146, 210, 235))
        for pipe in env.pipes:
            top = round(pipe.gap_center - c.gap_height / 2)
            bottom = round(pipe.gap_center + c.gap_height / 2)
            pygame.draw.rect(self.surface, (45, 150, 65), (round(pipe.x), 0, c.pipe_width, top))
            pygame.draw.rect(self.surface, (45, 150, 65), (round(pipe.x), bottom, c.pipe_width, c.height - bottom))
        # Match the square simulation hitbox rather than imply circular collisions.
        r = c.bird_radius
        pygame.draw.rect(self.surface, (255, 220, 40), (c.bird_x - r, round(env.bird.y) - r, 2 * r, 2 * r))
        pygame.draw.line(self.surface, (60, 60, 60), (0, 0), (c.width, 0), 3)
        pygame.draw.line(self.surface, (60, 60, 60), (0, c.height - 1), (c.width, c.height - 1), 3)
        label = self.font.render(f'Score: {env.score}   Speed: {env.pipe_speed:g} px/frame', True, (20, 20, 20))
        self.surface.blit(label, (12, 12))
        pygame.display.flip()
        return self.surface

    def close(self):
        """Close the display when a CLI session or notebook finishes rendering."""
        pygame.display.quit()
