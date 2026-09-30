"""Анимация падения до первого касания: нижняя точка, углы и кнопки."""

import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from main import (
    END_TOLERANCE_DEG,
    H,
    R_B,
    R_T,
    SIDE_TOLERANCE_DEG,
    simulate_fall,
)


# Размеры модели заданы в сантиметрах, поэтому g тоже в см/с².
GRAVITY = 981.0
INITIAL_GAP = 54.0  # Начальная высота нижней точки над плоскостью, см.
PLAYBACK_SLOWDOWN = 5.0  # Во сколько раз замедлено воспроизведение.
FPS = 30

# Сохраняем объекты анимации и кнопок, пока открыты их окна.
_OPEN_TRIALS = {}


def rotate_y(x, y, z, theta):
    """Поворачивает точку или массив точек вокруг оси y."""
    c, s = math.cos(theta), math.sin(theta)
    return x * c + z * s, y, -x * s + z * c


def draw_angle(ax, vertex, direction, length, color, symbol, label):
    """Рисует два отрезка и дугу острого угла в сечении xz.

    Первый отрезок направлен вдоль поверхности, второй — горизонтально.
    Возвращает угол в радианах и концы отрезков для настройки масштаба.
    """
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    if direction[2] < 0:
        direction = -direction
    horizontal = np.array([1.0 if direction[0] >= 0 else -1.0, 0, 0])
    angle = math.atan2(abs(direction[2]), abs(direction[0]))

    surface_end = vertex + length * direction
    horizontal_end = vertex + length * horizontal
    ax.plot(*np.array([vertex, surface_end]).T, color=color, linewidth=2.8,
            label=label, zorder=5)
    ax.plot(*np.array([vertex, horizontal_end]).T, color=color, linewidth=2,
            linestyle="--", label=f"Горизонталь для {symbol}", zorder=5)

    # Дуга расположена в той же вертикальной плоскости, что и оба отрезка.
    radius = length * 0.25
    t = np.linspace(0, angle, 60)
    arc = vertex[:, None] + radius * (
        horizontal[:, None] * np.cos(t)
        + np.array([[0], [0], [1]]) * np.sin(t)
    )
    ax.plot(*arc, color=color, linewidth=2, zorder=6)
    text_point = vertex + radius * 1.4 * (
        horizontal * math.cos(angle / 2)
        + np.array([0, 0, math.sin(angle / 2)])
    )
    ax.text(*text_point, f"{symbol} = {math.degrees(angle):.2f}°",
            color=color, fontsize=11,
            ha="left" if horizontal[0] > 0 else "right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
            zorder=7)
    return angle, surface_end, horizontal_end


def _build_plot(theta_deg, result):
    """Создаёт геометрию при нулевом переносе по z."""
    theta = math.radians(theta_deg)
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    surfaces = []

    def add_surface(x, y, z, color, zorder):
        surface = ax.plot_surface(
            x, y, z, color=color, alpha=0.35, linewidth=0,
            rcount=x.shape[0], ccount=x.shape[1], zorder=zorder,
        )
        points = np.stack([x, y, z], axis=-1)
        faces = np.stack([
            points[:-1, :-1], points[:-1, 1:],
            points[1:, 1:], points[1:, :-1],
        ], axis=-2).reshape(-1, 4, 3)
        surfaces.append((surface, faces))

    # Для прямой образующей достаточно двух рядов точек по высоте.
    phi, v = np.meshgrid(np.linspace(0, 2 * math.pi, 81), [0.0, 1.0])
    radius = R_B + (R_T - R_B) * v
    x, y, z = rotate_y(radius * np.cos(phi), radius * np.sin(phi),
                       H * (v - 0.5), theta)
    add_surface(x, y, z, color="#a2b9c9", zorder=1)
    body_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Дно и крышка.
    phi, rho = np.meshgrid(np.linspace(0, 2 * math.pi, 81), [0.0, 1.0])
    for radius, height in [(R_B, -H / 2), (R_T, H / 2)]:
        x, y, z = rotate_y(radius * rho * np.cos(phi), radius * rho * np.sin(phi),
                           np.full_like(rho, height), theta)
        add_surface(x, y, z, color="#b1bcc4", zorder=2)
        ax.plot(x[-1], y[-1], z[-1], color="#657988", linewidth=1.2, zorder=3)

    # Самые нижние точки ободов лежат в сечении y = 0.
    bottom = np.array(rotate_y(R_B, 0, -H / 2, theta))
    top = np.array(rotate_y(R_T, 0, H / 2, theta))
    on_bottom = bottom[2] <= top[2]
    lowest = bottom if on_bottom else top
    contact_radius = R_B if on_bottom else R_T

    # Отрезки 1 и 2: диаметр ближайшего торца и горизонталь через нижнюю точку.
    end_color, side_color = "#087f8c", "#b8640a"
    end_direction = rotate_y(-1, 0, 0, theta)
    end_angle, end_tip, end_horizontal = draw_angle(
        ax, lowest, end_direction, 2 * contact_radius, end_color, "α",
        "Направление торца",
    )

    # Отрезки 3 и 4: нижняя образующая и горизонталь через её середину.
    # Перенос точки отсчёта вдоль прямой не меняет угол с горизонталью.
    midpoint = (bottom + top) / 2
    side_direction = top - bottom
    side_angle, side_tip, side_horizontal = draw_angle(
        ax, midpoint, side_direction, np.linalg.norm(side_direction) / 2,
        side_color, "β", "Нижняя образующая",
    )

    ax.plot(*lowest, marker="o", linestyle="none", color="crimson",
            markersize=9, markeredgecolor="white", zorder=8)
    point_label = ax.text(lowest[0], lowest[1] - 0.2, lowest[2] - 0.9, "",
            ha="center", va="top", color="crimson", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3),
            zorder=9)

    # Запоминаем исходные координаты: все эти элементы движутся вместе с телом.
    lines = [(line, np.array(line.get_data_3d(), dtype=float)) for line in ax.lines]
    texts = [(text, np.array(text.get_position_3d())) for text in ax.texts]

    # Масштаб охватывает весь путь от начального положения до касания.
    all_points = np.vstack([body_points, end_tip, end_horizontal, side_tip, side_horizontal])
    lower = all_points.min(axis=0) - [2, 2.5, 3.5]
    upper = all_points.max(axis=0) + [2, 2.5, 2]
    lower[2] = -4
    upper[2] += INITIAL_GAP - lowest[2]
    plane_x, plane_y = np.meshgrid([lower[0], upper[0]], [lower[1], upper[1]])
    ax.plot_surface(plane_x, plane_y, np.zeros_like(plane_x),
                    color="gray", alpha=0.1, linewidth=0, zorder=0)
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
           zlim=(lower[2], upper[2]), xlabel="x, см", ylabel="y, см", zlabel="z, см")
    ax.set_box_aspect(upper - lower)
    ax.view_init(elev=17, azim=-70)

    end_deg, side_deg = math.degrees(end_angle), math.degrees(side_angle)
    fig.suptitle("Падение термоса до первого касания", fontsize=16, y=0.97)
    fig.text(0.5, 0.935, f"Начальный наклон θ = {theta_deg:.2f}° · Исход касания: {result}",
             ha="center", fontsize=11)
    status = fig.text(0.5, 0.91, "", ha="center", fontsize=11)
    fig.text(0.27, 0.195, f"Торец: α = {end_deg:.2f}°", color=end_color,
             ha="center", fontsize=13)
    fig.text(0.73, 0.195, f"Боковая поверхность: β = {side_deg:.2f}°", color=side_color,
             ha="center", fontsize=13)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.125), frameon=False, fontsize=10)
    fig.text(0.5, 0.092,
             f"Плоскость: z = 0 · g = {GRAVITY / 100:g} м/с² · "
             f"Воспроизведение замедлено в {PLAYBACK_SLOWDOWN:g} раз",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.26, top=0.88)
    return fig, end_deg, side_deg, lowest, surfaces, lines, texts, point_label, status


class TrialAnimation:
    """Одно независимое окно с неизменным наклоном и падением вдоль z."""

    def __init__(self, theta_deg, result):
        if INITIAL_GAP < 0 or GRAVITY <= 0 or PLAYBACK_SLOWDOWN <= 0 or FPS <= 0:
            raise ValueError("Нужны высота ≥ 0 и положительные g, замедление и FPS.")
        self.theta_deg = theta_deg
        self.result = result
        self.contact_time = math.sqrt(2 * INITIAL_GAP / GRAVITY)
        (self.fig, self.end_deg, self.side_deg, self.lowest,
         self.surfaces, self.lines, self.texts, self.point_label,
         self.status) = _build_plot(theta_deg, result)

        self.new_button = Button(self.fig.add_axes([0.22, 0.025, 0.25, 0.045]),
                                 "Новое испытание ↗")
        self.replay_button = Button(self.fig.add_axes([0.53, 0.025, 0.25, 0.045]),
                                    "Повторить текущее")
        self.new_button.on_clicked(open_new_trial)
        self.replay_button.on_clicked(self.restart)

        self.timer = self.fig.canvas.new_timer(interval=round(1000 / FPS))
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        _OPEN_TRIALS[self.fig.number] = self
        self.restart()

    def _draw_frame(self, t):
        """z(t) = z(0) − g t² / 2 при нулевой начальной скорости."""
        self.physical_time = min(max(t, 0.0), self.contact_time)
        self.gap = (0.0 if self.physical_time >= self.contact_time else
                    max(0.0, INITIAL_GAP - GRAVITY * self.physical_time**2 / 2))
        shift = np.array([0.0, 0.0, self.gap - self.lowest[2]])
        for surface, vertices in self.surfaces:
            surface.set_verts(vertices + shift)
        for line, coordinates in self.lines:
            line.set_data_3d(coordinates + shift[:, None])
        for text, position in self.texts:
            text.set_position_3d(position + shift)

        p = self.lowest + shift
        self.point_label.set_text(
            f"Нижняя точка P\n({p[0]:.2f}; {p[1]:.2f}; {p[2]:.2f}) см"
        )
        phase = "Первое касание" if self.physical_time >= self.contact_time else "Падение"
        self.status.set_text(
            f"{phase} · t = {self.physical_time:.3f} с · Высота точки P: {self.gap:.2f} см"
        )

    def restart(self, event=None):
        """Повторяет падение с тем же наклоном, в том числе во время движения."""
        self.timer.stop()
        self._draw_frame(0.0)
        self.fig.canvas.draw()
        self.started_at = perf_counter()
        self.running = self.contact_time > 0
        if self.running:
            self.timer.start()

    def _tick(self):
        if not self.running:
            return
        t = (perf_counter() - self.started_at) / PLAYBACK_SLOWDOWN
        self._draw_frame(t)
        if self.physical_time >= self.contact_time:
            self.running = False
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self.timer.stop()
        self.running = False
        _OPEN_TRIALS.pop(self.fig.number, None)


def plot_trial(theta_deg, result):
    """Запускает заданное испытание; возвращает фигуру и два угла в градусах."""
    trial = TrialAnimation(theta_deg, result)
    return trial.fig, trial.end_deg, trial.side_deg


def open_new_trial(event=None):
    """Генерирует новый наклон и открывает отдельное окно, сохраняя прежние."""
    theta_deg, _, result = simulate_fall(END_TOLERANCE_DEG, SIDE_TOLERANCE_DEG)
    fig, end_deg, side_deg = plot_trial(theta_deg, result)
    print("\nНовое испытание")
    print(f"Начальный наклон оси: {theta_deg:.2f}°")
    print(f"Угол торца к горизонтали: {end_deg:.2f}°")
    print(f"Угол боковой поверхности к горизонтали: {side_deg:.2f}°")
    print(f"Результат: {result}")
    plt.show(block=False)
    return fig


def main():
    open_new_trial()
    plt.show()


if __name__ == "__main__":
    main()
