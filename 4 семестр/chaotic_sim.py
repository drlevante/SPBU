"""
chaotic_sim.py

Модуль моделирования алгоритмов детерминированного и хаотического
имитационного отжига (Chaotic Simulated Annealing, CSA).
Реализует генераторы хаотических возмущений, тестовые целевые функции
и процедуру метаэвристической оптимизации с фиксацией метрики First Hitting Time (FHT).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


# =====================================================================
# 1. БАЗОВЫЙ ИНТЕРФЕЙС И КЛАССЫ ГЕНЕРАТОРОВ ВОЗМУЩЕНИЙ
# =====================================================================

class BasePerturbationGenerator(ABC):
    """Абстрактный базовый класс генератора возмущений xi_k in [-1, 1]."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Сброс внутреннего состояния генератора к начальным условиям."""
        pass

    @abstractmethod
    def step(self) -> float:
        """
        Генерация одного шага возмущения.

        Returns:
            Вещественное число в интервале [-1, 1].
        """
        pass


class PRNGGenerator(BasePerturbationGenerator):
    """
    Классический псевдослучайный генератор с равномерным распределением
    U(-1, 1) на базе numpy.random.Generator.
    """

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def step(self) -> float:
        return float(self._rng.uniform(-1.0, 1.0))


class LogisticMapGenerator(BasePerturbationGenerator):
    """
    Генератор на базе одномерного логистического отображения:
    z_{k+1} = 4 * z_k * (1 - z_k), z in (0, 1).
    Формирует центрированное возмущение: xi_k = 2 * z_k - 1 in [-1, 1].
    Инвариантная плотность распределения: Чебышёвская мера.
    """

    def reset(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Инициализируем точку вне сингулярных границ 0 и 1
        self._z = float(rng.uniform(0.05, 0.95))

    def step(self) -> float:
        self._z = 4.0 * self._z * (1.0 - self._z)
        # Защита от выхода за пределы (0, 1) из-за погрешностей округления
        if self._z <= 0.0 or self._z >= 1.0:
            self._z = (self._z + 1e-5 * np.pi) % 1.0
            if self._z == 0.0:
                self._z = 0.314159265
        return float(2.0 * self._z - 1.0)


class TentMapGenerator(BasePerturbationGenerator):
    """
    Генератор на базе симметричного кусочно-линейного отображения палатки:
    z_{k+1} = 2 * z_k, если z_k < 0.5, иначе 2 * (1 - z_k).
    Инвариантная мера: строго равномерная мера Лебега.
    Содержит регуляризацию для предотвращения битового схлопывания во float64.
    """

    def reset(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._z = float(rng.uniform(0.05, 0.95))

    def step(self) -> float:
        if self._z < 0.5:
            self._z = 2.0 * self._z
        else:
            self._z = 2.0 * (1.0 - self._z)

        # Регуляризация от зануления мантиссы при последовательных делениях/умножениях на 2
        if self._z <= 0.0 or self._z >= 1.0 or abs(self._z - 0.5) < 1e-15:
            self._z = (self._z + 1e-7 * np.e) % 1.0
            if self._z == 0.0:
                self._z = 0.5772156649

        return float(2.0 * self._z - 1.0)


class HenonMapGenerator(BasePerturbationGenerator):
    """
    Генератор на базе двумерного диссипативного отображения Энона:
    x_{k+1} = 1 - a * x_k^2 + y_k,
    y_{k+1} = b * x_k.
    Параметры аттрактора: a = 1.4, b = 0.3.
    Для формирования шага используется нормированная проекция x_k in [-1, 1].
    """

    def __init__(self, seed: Optional[int] = None, a: float = 1.4, b: float = 0.3) -> None:
        self.a = a
        self.b = b
        self._x_min = -1.28
        self._x_max = 1.28
        super().__init__(seed)

    def reset(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._x = float(rng.uniform(-0.5, 0.5))
        self._y = float(rng.uniform(-0.1, 0.1))
        # Пропуск 500 итераций переходного процесса для выхода на аттрактор
        for _ in range(500):
            self._step_internal()

    def _step_internal(self) -> None:
        x_next = 1.0 - self.a * (self._x ** 2) + self._y
        y_next = self.b * self._x
        # Реинициализация при редком уходе траектории в бесконечность
        if np.isnan(x_next) or abs(x_next) > 3.0:
            self._x, self._y = 0.1, 0.1
        else:
            self._x, self._y = x_next, y_next

    def step(self) -> float:
        self._step_internal()
        # Линейное отображение диапазона [-1.28, 1.28] в [-1.0, 1.0]
        x_clamped = np.clip(self._x, self._x_min, self._x_max)
        return float(2.0 * (x_clamped - self._x_min) / (self._x_max - self._x_min) - 1.0)


# =====================================================================
# 2. ТЕСТОВЫЕ МУЛЬТИМОДАЛЬНЫЕ ФУНКЦИИ ОПТИМИЗАЦИИ
# =====================================================================

@dataclass(frozen=True)
class BenchmarkFunction:
    """Контейнер описания тестовой функции оптимизации."""
    name: str
    func: Callable[[np.ndarray], float]
    bounds: Tuple[float, float]
    global_optimum: float
    center_optimum: bool  # True, если оптимум строго в центре диапазона


def sphere(x: np.ndarray) -> float:
    """Сферическая унимодальная функция: f(x) = sum(x_i^2)."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Высокомультимодальная функция Растригина с регулярной сеткой барьеров."""
    d = len(x)
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def griewank(x: np.ndarray) -> float:
    """Мультимодальная функция Гриванка со связностью координат."""
    indices = np.arange(1, len(x) + 1)
    sum_sq = np.sum(x ** 2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(indices)))
    return float(sum_sq - prod_cos + 1.0)


def shifted_sphere(x: np.ndarray, shift_val: float = 4.5) -> float:
    """
    Сферическая функция со смещением глобального минимума к краю бруса [a, b].
    Используется для проверки краевой топологической селективности (Yang et al., 2007).
    """
    return float(np.sum((x - shift_val) ** 2))


BENCHMARK_SUITE = {
    "Sphere": BenchmarkFunction(
        name="Sphere",
        func=sphere,
        bounds=(-5.12, 5.12),
        global_optimum=0.0,
        center_optimum=True,
    ),
    "Rastrigin": BenchmarkFunction(
        name="Rastrigin",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        global_optimum=0.0,
        center_optimum=True,
    ),
    "Griewank": BenchmarkFunction(
        name="Griewank",
        func=griewank,
        bounds=(-10.0, 10.0),
        global_optimum=0.0,
        center_optimum=True,
    ),
    "ShiftedSphere": BenchmarkFunction(
        name="ShiftedSphere",
        func=lambda x: shifted_sphere(x, shift_val=4.5),
        bounds=(-5.0, 5.0),
        global_optimum=0.0,
        center_optimum=False,
    ),
}


# =====================================================================
# 3. КЛАСС АЛГОРИТМА CHAOTIC SIMULATED ANNEALING (CSA)
# =====================================================================

@dataclass
class OptimizationResult:
    """Результаты единичного запуска алгоритма отжига."""
    best_x: np.ndarray
    best_energy: float
    fht: Optional[int]  # First Hitting Time: номер шага первого достижения epsilon-окрестности
    energy_history: np.ndarray


class ChaoticSimulatedAnnealing:
    """
    Алгоритм имитационного отжига с поддержкой хаотических возмущений состояния.
    Реализует критерий Метрополиса и масштабирование шага возмущения.
    """

    def __init__(
            self,
            benchmark: BenchmarkFunction,
            dimension: int,
            generator: BasePerturbationGenerator,
            t_init: float = 10.0,
            t_final: float = 1e-4,
            cooling_factor: float = 0.995,
            alpha_init: float = 0.2,
            alpha_decay: float = 0.998,
            max_iterations: int = 1500,
            epsilon: float = 1e-2,
            seed: Optional[int] = None,
    ) -> None:
        self.benchmark = benchmark
        self.dimension = dimension
        self.generator = generator
        self.t_init = t_init
        self.t_final = t_final
        self.cooling_factor = cooling_factor
        self.alpha_init = alpha_init
        self.alpha_decay = alpha_decay
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.seed = seed

    def optimize(self) -> OptimizationResult:
        """
        Запуск процедуры оптимизации.

        Returns:
            Объект OptimizationResult с найденными характеристиками.
        """
        rng = np.random.default_rng(self.seed)
        self.generator.reset()

        a, b = self.benchmark.bounds
        span = b - a

        # Случайная равномерная инициализация вектора состояния
        current_x = rng.uniform(a, b, size=self.dimension)
        current_energy = self.benchmark.func(current_x)

        best_x = current_x.copy()
        best_energy = current_energy

        t = self.t_init
        alpha = self.alpha_init

        fht: Optional[int] = None
        if abs(best_energy - self.benchmark.global_optimum) <= self.epsilon:
            fht = 0

        energy_history = np.zeros(self.max_iterations)

        for step_idx in range(self.max_iterations):
            # Векторное хаотическое/псевдослучайное возмущение
            perturbation = np.array([self.generator.step() for _ in range(self.dimension)])

            # Формирование пробного состояния: x_new = x_curr + alpha * span * xi
            candidate_x = current_x + alpha * span * perturbation
            candidate_x = np.clip(candidate_x, a, b)

            candidate_energy = self.benchmark.func(candidate_x)
            delta_e = candidate_energy - current_energy

            # Критерий Метрополиса
            if delta_e < 0.0:
                current_x = candidate_x
                current_energy = candidate_energy
            else:
                prob = np.exp(-delta_e / max(t, 1e-12))
                if rng.uniform(0.0, 1.0) < prob:
                    current_x = candidate_x
                    current_energy = candidate_energy

            # Обновление глобального рекорда
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = current_x.copy()

            # Фиксация времени первого достижения окрестности оптимума
            if fht is None and abs(best_energy - self.benchmark.global_optimum) <= self.epsilon:
                fht = step_idx + 1

            energy_history[step_idx] = best_energy

            # Геометрическое расписание охлаждения и редукции радиуса шага
            t = max(self.t_final, t * self.cooling_factor)
            alpha = max(1e-4, alpha * self.alpha_decay)

        return OptimizationResult(
            best_x=best_x,
            best_energy=best_energy,
            fht=fht,
            energy_history=energy_history,
        )