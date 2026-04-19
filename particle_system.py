import cv2
import numpy as np

from config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    FRICTION,
    GRAVITY,
    HAND_BLAST_RADIUS,
    HAND_BLAST_STRENGTH,
    HAND_ENERGY_BOOST,
    HAND_GATHER_RADIUS,
    HAND_GATHER_STRENGTH,
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
        self.energy = np.zeros(NUM_PARTICLES, dtype=np.float32)
        self._boundary_glow = np.zeros(NUM_PARTICLES, dtype=bool)
        self._no_person_frames = 0
        self._previous_mask = None
        self._hand_charge = {}
        self._burst_waves = []
        self._settle_return_force = 0.05
        self._micro_drift_strength = 0.02
        self._rim_light_color = np.array((255, 250, 240), dtype=np.uint8)
        self._hand_charge_rate = 0.045
        self._max_hand_charge = 1.8
        self._gather_radius_gain = 2.35
        self._gather_pull_gain = 2.9
        self._gather_velocity_damp = 0.24
        self._burst_duration = 12
        self._burst_radius_gain = 520.0
        self._burst_force_gain = 5.5
        self._rng = np.random.default_rng(17)
        self._field_width = max(320, CANVAS_WIDTH // 2)
        self._field_height = max(180, CANVAS_HEIGHT // 2)
        self._field_scale_x = self._field_width / CANVAS_WIDTH
        self._field_scale_y = self._field_height / CANVAS_HEIGHT
        self._field_push_radius = max(2.0, PUSH_RADIUS * self._field_scale_x)
        self._boundary_glow_distance = max(2.0, 8.0 * self._field_scale_x)
        self._empty_field = np.zeros((self._field_height, self._field_width), dtype=np.float32)

        self._base_color = np.array(PARTICLE_COLOR_BASE, dtype=np.uint8)
        self._disturbed_color = np.array(PARTICLE_COLOR_DISTURBED, dtype=np.uint8)
        self._hot_color = np.array((255, 244, 225), dtype=np.uint8)
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

    def update(self, mask, hands=None):
        if mask.shape[:2] != (CANVAS_HEIGHT, CANVAS_WIDTH):
            raise ValueError(
                f"Mask must have shape {(CANVAS_HEIGHT, CANVAS_WIDTH)}, got {mask.shape[:2]}."
            )

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        field_mask = cv2.resize(
            mask,
            (self._field_width, self._field_height),
            interpolation=cv2.INTER_NEAREST,
        )
        has_person = cv2.countNonZero(field_mask) > 0
        if has_person:
            self._no_person_frames = 0
        else:
            self._no_person_frames += 1

        if self._previous_mask is None:
            motion_map = self._empty_field
            motion_gx = self._empty_field
            motion_gy = self._empty_field
        else:
            motion_delta = cv2.absdiff(field_mask, self._previous_mask)
            motion_map = cv2.GaussianBlur(motion_delta.astype(np.float32) / 255.0, (15, 15), 0)
            motion_gx = cv2.Sobel(motion_map, cv2.CV_32F, 1, 0, ksize=3)
            motion_gy = cv2.Sobel(motion_map, cv2.CV_32F, 0, 1, ksize=3)

        self._previous_mask = field_mask.copy()

        field_x_indices = np.clip((self.x * self._field_scale_x).astype(np.int32), 0, self._field_width - 1)
        field_y_indices = np.clip((self.y * self._field_scale_y).astype(np.int32), 0, self._field_height - 1)

        motion_x = motion_gx[field_y_indices, field_x_indices]
        motion_y = motion_gy[field_y_indices, field_x_indices]
        motion_magnitude = np.maximum(np.hypot(motion_x, motion_y), 1e-6)
        motion_strength = np.clip(motion_map[field_y_indices, field_x_indices] * 1.8, 0.0, 1.0).astype(np.float32)

        if has_person:
            inverted_mask = cv2.bitwise_not(field_mask)
            inside_distance = cv2.distanceTransform(field_mask, cv2.DIST_L2, 5)
            outside_distance = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
            signed_distance = outside_distance - inside_distance
            gx = cv2.Sobel(signed_distance, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(signed_distance, cv2.CV_32F, 0, 1, ksize=3)

            inside = field_mask[field_y_indices, field_x_indices] > 128
            boundary_distance = np.where(
                inside,
                inside_distance[field_y_indices, field_x_indices],
                outside_distance[field_y_indices, field_x_indices],
            )
            self._boundary_glow = boundary_distance <= self._boundary_glow_distance

            push_x = gx[field_y_indices, field_x_indices]
            push_y = gy[field_y_indices, field_x_indices]
            push_magnitude = np.maximum(np.hypot(push_x, push_y), 1e-6)
            edge_distance = inside_distance[field_y_indices, field_x_indices]
            push_scale = np.clip(
                (self._field_push_radius - edge_distance) / max(self._field_push_radius, 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            push_scale *= inside.astype(np.float32)
        else:
            self._boundary_glow.fill(False)
            push_x = np.zeros_like(self.x)
            push_y = np.zeros_like(self.y)
            push_magnitude = np.ones_like(self.x)
            push_scale = np.zeros_like(self.x)

        self.vx += (push_x / push_magnitude) * PUSH_STRENGTH * push_scale
        self.vy += (push_y / push_magnitude) * PUSH_STRENGTH * push_scale
        self.vx += (
            (motion_x / motion_magnitude)
            * (PUSH_STRENGTH * 0.85)
            * motion_strength
        )
        self.vy += (
            (motion_y / motion_magnitude)
            * (PUSH_STRENGTH * 0.85)
            * motion_strength
        )

        if hands:
            seen_labels = set()
            for hand in hands:
                label = hand["label"]
                seen_labels.add(label)
                previous_charge = self._hand_charge.get(label, 0.0)
                dx = hand["x"] - self.x
                dy = hand["y"] - self.y
                distance = np.hypot(dx, dy)
                safe_distance = np.maximum(distance, 1e-6)

                if hand["is_closed"]:
                    charge = min(previous_charge + self._hand_charge_rate, self._max_hand_charge)
                    self._hand_charge[label] = charge
                    gather_radius = HAND_GATHER_RADIUS * (1.0 + (charge * self._gather_radius_gain))
                    gather_scale = np.clip((gather_radius - distance) / gather_radius, 0.0, 1.0)
                    gather_scale = gather_scale.astype(np.float32) ** 2.35
                    gather_strength = HAND_GATHER_STRENGTH * (0.85 + (charge * self._gather_pull_gain))
                    self.vx += (dx / safe_distance) * gather_strength * gather_scale
                    self.vy += (dy / safe_distance) * gather_strength * gather_scale
                    local_damping = np.clip(
                        gather_scale * (0.08 + (charge * self._gather_velocity_damp)),
                        0.0,
                        0.42,
                    )
                    self.vx *= 1.0 - local_damping
                    self.vy *= 1.0 - local_damping
                    self.energy = np.clip(
                        self.energy + (gather_scale * HAND_ENERGY_BOOST * (0.75 + (charge * 0.95))),
                        0.0,
                        1.45,
                    )

                if hand["just_opened"]:
                    blast_charge = max(previous_charge, 0.4)
                    blast_radius = HAND_BLAST_RADIUS + (blast_charge * self._burst_radius_gain)
                    blast_scale = np.clip((blast_radius - distance) / blast_radius, 0.0, 1.0)
                    blast_scale = np.sqrt(blast_scale.astype(np.float32))
                    blast_strength = HAND_BLAST_STRENGTH * (2.4 + (blast_charge * self._burst_force_gain))
                    self.vx += (-dx / safe_distance) * blast_strength * blast_scale
                    self.vy += (-dy / safe_distance) * blast_strength * blast_scale
                    self.energy = np.clip(
                        self.energy + (blast_scale * (1.35 + (blast_charge * 0.85))),
                        0.0,
                        1.45,
                    )
                    self._burst_waves.append(
                        {
                            "x": float(hand["x"]),
                            "y": float(hand["y"]),
                            "charge": float(blast_charge),
                            "radius": float(blast_radius),
                            "age": 0,
                        }
                    )
                    self._hand_charge.pop(label, None)
                elif not hand["is_closed"]:
                    faded_charge = max(previous_charge - 0.18, 0.0)
                    if faded_charge > 0.0:
                        self._hand_charge[label] = faded_charge
                    else:
                        self._hand_charge.pop(label, None)

            for label in list(self._hand_charge):
                if label not in seen_labels:
                    del self._hand_charge[label]
        else:
            self._hand_charge.clear()

        if self._burst_waves:
            remaining_waves = []
            max_wave_radius = float(np.hypot(CANVAS_WIDTH, CANVAS_HEIGHT))
            for burst_wave in self._burst_waves:
                burst_wave["age"] += 1
                progress = burst_wave["age"] / self._burst_duration
                if progress >= 1.0:
                    continue

                dx = self.x - burst_wave["x"]
                dy = self.y - burst_wave["y"]
                distance = np.hypot(dx, dy)
                safe_distance = np.maximum(distance, 1e-6)
                wave_radius = min(
                    burst_wave["radius"] * (0.22 + (progress * 0.9)),
                    max_wave_radius,
                )
                wave_band = 34.0 + (burst_wave["charge"] * 28.0)
                wave_scale = np.clip(1.0 - (np.abs(distance - wave_radius) / wave_band), 0.0, 1.0)
                wave_scale = wave_scale.astype(np.float32) ** 1.6
                wave_strength = HAND_BLAST_STRENGTH * (1.4 + (burst_wave["charge"] * 2.8)) * (1.0 - progress)
                self.vx += (dx / safe_distance) * wave_strength * wave_scale
                self.vy += (dy / safe_distance) * wave_strength * wave_scale
                self.energy = np.clip(
                    self.energy + (wave_scale * (0.3 + (burst_wave["charge"] * 0.35))),
                    0.0,
                    1.45,
                )
                remaining_waves.append(burst_wave)

            self._burst_waves = remaining_waves

        settling = self._no_person_frames > 30
        return_force = self._settle_return_force if settling else RETURN_FORCE
        self.vx += (self.home_x - self.x) * return_force
        self.vy += (self.home_y - self.y) * return_force

        energy_injection = (
            (push_scale * 0.7)
            + (motion_strength * 0.55)
            + (self._boundary_glow.astype(np.float32) * 0.4)
        )
        self.energy = np.clip((self.energy * 0.9) + energy_injection, 0.0, 1.25)

        if not settling:
            self.vx += self._rng.uniform(
                -self._micro_drift_strength,
                self._micro_drift_strength,
                size=self.vx.shape,
            ).astype(np.float32)
            self.vy += GRAVITY
        else:
            self.energy *= 0.86

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

    @staticmethod
    def _stamp_particles(canvas, x_coords, y_coords, colors, offsets):
        for dx, dy in offsets:
            stamped_x = x_coords + dx
            stamped_y = y_coords + dy
            valid = (
                (stamped_x >= 0)
                & (stamped_x < CANVAS_WIDTH)
                & (stamped_y >= 0)
                & (stamped_y < CANVAS_HEIGHT)
            )
            if np.any(valid):
                canvas[stamped_y[valid], stamped_x[valid]] = colors[valid]

    def draw(self, canvas):
        pixel_x = np.clip(np.round(self.x).astype(np.int32), 0, CANVAS_WIDTH - 1)
        pixel_y = np.clip(np.round(self.y).astype(np.int32), 0, CANVAS_HEIGHT - 1)
        glow = self._boundary_glow
        displaced = np.hypot(self.x - self.home_x, self.y - self.home_y) > 5.0
        energy_mix = np.clip(self.energy, 0.0, 1.0)[:, None]
        base_colors = np.where(
            displaced[:, None],
            self._disturbed_color,
            self._base_color,
        )
        colors = np.clip(
            base_colors + ((self._hot_color - base_colors) * energy_mix),
            0,
            255,
        ).astype(np.uint8)
        colors[glow] = self._rim_light_color

        regular_particles = ~glow
        if np.any(regular_particles):
            regular_x = pixel_x[regular_particles]
            regular_y = pixel_y[regular_particles]
            regular_colors = colors[regular_particles]
            canvas[regular_y, regular_x] = regular_colors
            if self._neighbor_offsets.size:
                self._stamp_particles(
                    canvas,
                    regular_x,
                    regular_y,
                    regular_colors,
                    self._neighbor_offsets,
                )

        if np.any(glow):
            self._stamp_particles(
                canvas,
                pixel_x[glow],
                pixel_y[glow],
                colors[glow],
                self._glow_offsets,
            )
