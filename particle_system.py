import cv2
import numpy as np

from config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    FRICTION,
    GRAVITY,
    NUM_PARTICLES,
    PARTICLE_COLOR_BASE,
    PARTICLE_COLOR_DISTURBED,
    PARTICLE_RADIUS,
    PUSH_RADIUS,
    PUSH_STRENGTH,
    RETURN_FORCE,
)


class ParticleSystem:
    def __init__(self):
        cols = int(np.ceil(np.sqrt(NUM_PARTICLES * CANVAS_WIDTH / CANVAS_HEIGHT)))
        rows = int(np.ceil(NUM_PARTICLES / cols))

        grid_x = np.linspace(PARTICLE_RADIUS, CANVAS_WIDTH - 1 - PARTICLE_RADIUS, cols, dtype=np.float32)
        grid_y = np.linspace(PARTICLE_RADIUS, CANVAS_HEIGHT - 1 - PARTICLE_RADIUS, rows, dtype=np.float32)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

        self.home_x = mesh_x.ravel()[:NUM_PARTICLES].copy()
        self.home_y = mesh_y.ravel()[:NUM_PARTICLES].copy()
        self.x = self.home_x.copy()
        self.y = self.home_y.copy()
        self.vx = np.zeros(NUM_PARTICLES, dtype=np.float32)
        self.vy = np.zeros(NUM_PARTICLES, dtype=np.float32)
        self._boundary_glow = np.zeros(NUM_PARTICLES, dtype=bool)
        self._no_person_frames = 0
        self._settle_return_force = 0.05
        self._micro_drift_strength = 0.02
        self._boundary_glow_distance = 8.0
        self._rim_light_color = np.array((255, 250, 240), dtype=np.uint8)
        self._rng = np.random.default_rng()

        self._base_color = np.array(PARTICLE_COLOR_BASE, dtype=np.uint8)
        self._disturbed_color = np.array(PARTICLE_COLOR_DISTURBED, dtype=np.uint8)
        self._neighbor_offsets = (
            np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int32)
            if PARTICLE_RADIUS > 1
            else np.empty((0, 2), dtype=np.int32)
        )
        self._glow_offsets = np.array(
            [
                (0, 0),
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
                (2, 0),
                (-2, 0),
                (0, 2),
                (0, -2),
            ],
            dtype=np.int32,
        )

    def update(self, mask):
        if mask.shape[:2] != (CANVAS_HEIGHT, CANVAS_WIDTH):
            raise ValueError(
                f"Mask must have shape {(CANVAS_HEIGHT, CANVAS_WIDTH)}, got {mask.shape[:2]}."
            )

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        binary_mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        has_person = cv2.countNonZero(binary_mask) > 0
        if has_person:
            self._no_person_frames = 0
        else:
            self._no_person_frames += 1

        inverted_mask = cv2.bitwise_not(binary_mask)
        inside_distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        outside_distance = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        signed_distance = outside_distance - inside_distance

        gx = cv2.Sobel(signed_distance, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(signed_distance, cv2.CV_32F, 0, 1, ksize=3)

        x_indices = np.clip(self.x.astype(np.int32), 0, CANVAS_WIDTH - 1)
        y_indices = np.clip(self.y.astype(np.int32), 0, CANVAS_HEIGHT - 1)
        inside = binary_mask[y_indices, x_indices] > 128
        boundary_distance = np.where(
            inside,
            inside_distance[y_indices, x_indices],
            outside_distance[y_indices, x_indices],
        )
        self._boundary_glow = has_person & (boundary_distance <= self._boundary_glow_distance)

        push_x = gx[y_indices, x_indices]
        push_y = gy[y_indices, x_indices]
        push_magnitude = np.hypot(push_x, push_y)
        push_magnitude = np.maximum(push_magnitude, 1e-6)

        edge_distance = inside_distance[y_indices, x_indices]
        push_scale = np.clip((PUSH_RADIUS - edge_distance) / max(PUSH_RADIUS, 1), 0.0, 1.0).astype(np.float32)
        push_scale *= inside.astype(np.float32)

        self.vx += (push_x / push_magnitude) * PUSH_STRENGTH * push_scale
        self.vy += (push_y / push_magnitude) * PUSH_STRENGTH * push_scale

        settling = self._no_person_frames > 30
        return_force = self._settle_return_force if settling else RETURN_FORCE
        self.vx += (self.home_x - self.x) * return_force
        self.vy += (self.home_y - self.y) * return_force

        if not settling:
            self.vx += self._rng.uniform(
                -self._micro_drift_strength,
                self._micro_drift_strength,
                size=self.vx.shape,
            ).astype(np.float32)
            self.vy += GRAVITY

        self.vx *= FRICTION
        self.vy *= FRICTION

        self.x += self.vx
        self.y += self.vy

        clipped_x = np.clip(self.x, PARTICLE_RADIUS, CANVAS_WIDTH - 1 - PARTICLE_RADIUS)
        clipped_y = np.clip(self.y, PARTICLE_RADIUS, CANVAS_HEIGHT - 1 - PARTICLE_RADIUS)

        hit_x_bounds = clipped_x != self.x
        hit_y_bounds = clipped_y != self.y

        self.x = clipped_x
        self.y = clipped_y
        self.vx[hit_x_bounds] = 0.0
        self.vy[hit_y_bounds] = 0.0

        if settling:
            settled = (
                (np.abs(self.x - self.home_x) < 0.5)
                & (np.abs(self.y - self.home_y) < 0.5)
                & (np.abs(self.vx) < 0.05)
                & (np.abs(self.vy) < 0.05)
            )
            self.x[settled] = self.home_x[settled]
            self.y[settled] = self.home_y[settled]
            self.vx[settled] = 0.0
            self.vy[settled] = 0.0

    def draw(self, canvas):
        pixel_x = np.clip(np.round(self.x).astype(np.int32), 0, CANVAS_WIDTH - 1)
        pixel_y = np.clip(np.round(self.y).astype(np.int32), 0, CANVAS_HEIGHT - 1)
        glow = self._boundary_glow
        disturbed = (np.hypot(self.x - self.home_x, self.y - self.home_y) > 5.0) & ~glow
        base = ~(disturbed | glow)

        base_x = pixel_x[base]
        base_y = pixel_y[base]
        disturbed_x = pixel_x[disturbed]
        disturbed_y = pixel_y[disturbed]
        glow_x = pixel_x[glow]
        glow_y = pixel_y[glow]

        if base_x.size:
            canvas[base_y, base_x] = self._base_color
        if disturbed_x.size:
            canvas[disturbed_y, disturbed_x] = self._disturbed_color
        if glow_x.size:
            canvas[glow_y, glow_x] = self._rim_light_color

        for dx, dy in self._neighbor_offsets:
            if base_x.size:
                neighbor_x = base_x + dx
                neighbor_y = base_y + dy
                valid = (
                    (neighbor_x >= 0)
                    & (neighbor_x < CANVAS_WIDTH)
                    & (neighbor_y >= 0)
                    & (neighbor_y < CANVAS_HEIGHT)
                )
                canvas[neighbor_y[valid], neighbor_x[valid]] = self._base_color

            if disturbed_x.size:
                neighbor_x = disturbed_x + dx
                neighbor_y = disturbed_y + dy
                valid = (
                    (neighbor_x >= 0)
                    & (neighbor_x < CANVAS_WIDTH)
                    & (neighbor_y >= 0)
                    & (neighbor_y < CANVAS_HEIGHT)
                )
                canvas[neighbor_y[valid], neighbor_x[valid]] = self._disturbed_color

        for dx, dy in self._glow_offsets:
            if glow_x.size:
                neighbor_x = glow_x + dx
                neighbor_y = glow_y + dy
                valid = (
                    (neighbor_x >= 0)
                    & (neighbor_x < CANVAS_WIDTH)
                    & (neighbor_y >= 0)
                    & (neighbor_y < CANVAS_HEIGHT)
                )
                canvas[neighbor_y[valid], neighbor_x[valid]] = self._rim_light_color
