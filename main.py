import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Размеры термоса в сантиметрах.
R_B = 3
R_T = 3.5
H = 18

# Допуски в градусах: общий для дна и крышки, отдельный для бока.
END_TOLERANCE_DEG = 5
SIDE_TOLERANCE_DEG = 3

# Объёмы накопленных выборок: первые 10 бросков, первые 100 и так далее.
SAMPLE_SIZES = [10, 100, 1000, 10000, 100000]
PRINT_EACH_TRIAL = False  # True — дополнительно печатать каждый бросок.

OUTCOMES = (
    "Нижняя поверхность",
    "Верхняя поверхность",
    "Боковая сторона",
    "Нижнее ребро",
    "Верхнее ребро",
)


def simulate_fall(
    end_tolerance_deg=END_TOLERANCE_DEG,
    side_tolerance_deg=SIDE_TOLERANCE_DEG,
):
    """Проводит одно испытание и возвращает (theta_deg, fall_angle_deg, result).

    theta_deg — начальный угол оси от дна к крышке с вертикалью вверх.
    fall_angle_deg — меньший из углов торца и нижней образующей к горизонтали.
    result — одна из пяти категорий первого касания с учётом допусков.
    end_tolerance_deg применяется к дну и крышке, side_tolerance_deg — к боку.
    В этой геометрической модели ориентация до касания не меняется.
    """
    if not 0 <= end_tolerance_deg <= 90:
        raise ValueError("Допуск для торцов должен быть от 0 до 90 градусов.")
    if not 0 <= side_tolerance_deg <= 90:
        raise ValueError("Допуск для бока должен быть от 0 до 90 градусов.")

    # 1. Равномерно выбираем cos(theta), чтобы направления в пространстве
    # были равновероятны. Сам theta лежит от 0 до pi.
    c = random.uniform(-1, 1)
    theta = math.acos(c)
    s = math.sqrt(max(0.0, 1 - c * c))

    # 2. Углы торца и нижней образующей к горизонтальной плоскости.
    end_angle = min(theta, math.pi - theta)
    side_x = (R_T - R_B) * c + H * s
    side_z = H * c - (R_T - R_B) * s
    side_angle = math.atan2(abs(side_z), abs(side_x))
    fall_angle = min(end_angle, side_angle)

    # 3. Находим, какой из ободов ниже.
    bottom_z = -H / 2 * c - R_B * s
    top_z = H / 2 * c - R_T * s

    # 4. Каждую поверхность проверяем по её собственному допуску.
    end_tolerance = math.radians(end_tolerance_deg)
    side_tolerance = math.radians(side_tolerance_deg)
    # Равенство на границе допуска учитывает погрешность вычислений.
    end_within_tolerance = end_angle <= end_tolerance or math.isclose(
        end_angle, end_tolerance, rel_tol=0.0, abs_tol=1e-12
    )
    side_within_tolerance = side_angle <= side_tolerance or math.isclose(
        side_angle, side_tolerance, rel_tol=0.0, abs_tol=1e-12
    )

    # Если подходят обе поверхности, выбираем меньший угол к горизонтали.
    if end_within_tolerance and (not side_within_tolerance or end_angle <= side_angle):
        result = "Нижняя поверхность" if c >= 0 else "Верхняя поверхность"
    elif side_within_tolerance:
        result = "Боковая сторона"
    elif bottom_z <= top_z:
        result = "Нижнее ребро"
    else:
        result = "Верхнее ребро"

    return math.degrees(theta), math.degrees(fall_angle), result


def run_experiment(sample_sizes, print_each_trial=False):
    """Возвращает счётчики для выбранных N и историю частот в процентах.

    Проводится одна серия до максимального N. Каждая следующая выборка
    включает предыдущую; история содержит частоты после каждого броска.
    """
    if not sample_sizes or any(not isinstance(n, int) or n <= 0 for n in sample_sizes):
        raise ValueError("Объёмы выборок должны быть положительными целыми числами.")
    sample_sizes = sorted(set(sample_sizes))
    checkpoints = set(sample_sizes)
    counts = dict.fromkeys(OUTCOMES, 0)
    snapshots = {}
    history = {outcome: [] for outcome in OUTCOMES}

    for trial in range(1, sample_sizes[-1] + 1):
        theta_deg, fall_angle_deg, result = simulate_fall(
            end_tolerance_deg=END_TOLERANCE_DEG,
            side_tolerance_deg=SIDE_TOLERANCE_DEG,
        )
        counts[result] += 1

        if print_each_trial:
            print(
                f"{trial:6d} | θ = {theta_deg:7.2f}° | "
                f"α = {fall_angle_deg:6.2f}° | {result}",
                flush=True,
            )

        # Знаменатель — число уже проведённых испытаний, а не итоговый N.
        for outcome in OUTCOMES:
            history[outcome].append(counts[outcome] / trial * 100)

        if trial in checkpoints:
            snapshots[trial] = counts.copy()

    return snapshots, history


def plot_frequencies(history):
    """Строит накопленные относительные частоты всех пяти исходов."""
    n_trials = len(history[OUTCOMES[0]])
    trial_numbers = range(1, n_trials + 1)
    colors = ["#0072b2", "#d55e00", "#009e73", "#b24a97", "#332288"]
    line_styles = ["-", "--", "-.", "--", "-"]
    fig, ax = plt.subplots(figsize=(11, 7))

    for outcome, color, style in zip(OUTCOMES, colors, line_styles):
        frequencies = history[outcome]
        ax.plot(
            trial_numbers, frequencies, color=color, linestyle=style, linewidth=1.5,
            label=f"{outcome}: {frequencies[-1]:.3f}%",
        )

    # Логарифмическая ось X позволяет видеть и первые броски, и большие N.
    # По оси Y остаётся обычная шкала процентов, включая нулевые частоты.
    ax.set_xscale("log")
    ax.set_xlim(1, max(2, n_trials))
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " "))
    )
    ax.set_xlabel("Число испытаний n (логарифмическая шкала)")
    ax.set_ylabel("Относительная частота, %")
    ax.set_title(
        "Накопленные относительные частоты исходов\n"
        f"Допуски: торцы — {END_TOLERANCE_DEG:g}°, бок — {SIDE_TOLERANCE_DEG:g}°"
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
        frameon=False, title=f"Исходы и частоты после {n_trials:,} испытаний".replace(",", " "),
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.29)
    return fig, ax


def main():
    print(
        f"Допуски: торцы — {END_TOLERANCE_DEG:g}°, "
        f"бок — {SIDE_TOLERANCE_DEG:g}°."
    )
    print("Объёмы накопленных выборок:", ", ".join(map(str, SAMPLE_SIZES)))
    if PRINT_EACH_TRIAL:
        print("θ — начальный наклон оси к вертикали; α — угол падения.\n")

    snapshots, history = run_experiment(SAMPLE_SIZES, PRINT_EACH_TRIAL)

    for n, counts in snapshots.items():
        print(f"\nN = {n}")
        for outcome in OUTCOMES:
            count = counts[outcome]
            print(f"{outcome:<22}: {count:7d} | {count / n * 100:8.3f}%")
        print(f"Итого: {sum(counts.values())} испытаний, "
              f"{sum(counts.values()) / n * 100:.3f}%")

    fig, _ = plot_frequencies(history)
    output_path = Path(__file__).with_name("relative_frequencies.png")
    fig.savefig(output_path, dpi=160)
    print(f"\nГрафик сохранён: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
