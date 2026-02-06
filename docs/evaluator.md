# Модуль Evaluator — Техническая документация

> **Версия**: 1.0.0  
> **Дата**: 06.02.2026  
> **Автор**: Андреев Данила

---

## Содержание

1. [Обзор](#обзор)
2. [Архитектура модуля](#архитектура-модуля)
3. [SMOP: Экспертная оценка](#smop-экспертная-оценка)
4. [Статистический анализ](#статистический-анализ)
5. [Визуализация результатов](#визуализация-результатов)
6. [Генерация отчётов](#генерация-отчётов)
7. [Схемы данных](#схемы-данных)
8. [API Reference](#api-reference)
9. [Примеры использования](#примеры-использования)
10. [Скриншоты интерфейсов](#скриншоты-интерфейсов)

---

## Обзор

Модуль **`evaluator`** отвечает за экспертную оценку качества сгенерированного кода, статистический анализ результатов и формирование отчётов для научной публикации.

### Основные компоненты

```
src/evaluator/
├── schemas.py       # Pydantic-схемы данных оценки
├── smop.py          # Логика экспертной оценки SMOP
├── statistics.py    # Статистический анализ
├── charts.py        # Визуализация (matplotlib)
├── report.py        # Генерация отчётов (HTML, JSON, LaTeX)
├── parser.py        # Парсинг сырых результатов
└── dashboard.py     # CLI-интерфейс для оценки
```

### Функциональные требования

| FR | Описание |
|----|----------|
| **FR-201** | Хранение оценок S, M, O, P для каждого прогона |
| **FR-202** | Валидация значений (0, 2, 4, 6, 8, 10) |
| **FR-203** | Расчёт интегрального показателя Q |
| **FR-204** | Загрузка критериев из `smop_criteria.yaml` |
| **FR-205** | Сохранение оценок в `evaluations/` |
| **FR-301** | Расчёт среднего, медианы, стандартного отклонения |
| **FR-302** | Расчёт 95% доверительного интервала для Q |
| **FR-303** | Агрегация по моделям |
| **FR-304** | Агрегация по категориям задач |
| **FR-306** | Корреляция между детерминизмом и качеством |
| **FR-307** | Межэкспертная надёжность (Каппа Коэна) |
| **FR-401** | Генерация JSON-отчёта |
| **FR-402** | Генерация HTML-отчёта |
| **FR-405** | Графики распределения оценок |
| **FR-406** | Экспорт таблиц в LaTeX |

---

## Архитектура модуля

### Поток данных

```
Эксперимент (ExperimentResult)
         ↓
    Парсинг (parser.py)
         ↓
Структура оценки (ExperimentEvaluation)
         ↓
 Экспертная оценка (smop.py, dashboard.py)
         ↓
Заполнение SMOP (S, M, O, P) → Расчёт Q
         ↓
Статистический анализ (statistics.py)
         ↓
Визуализация (charts.py) + Отчёты (report.py)
         ↓
Финальные артефакты (JSON, HTML, LaTeX, PNG)
```

### Основные классы

| Класс | Файл | Назначение |
|-------|------|------------|
| `SMOPScores` | `schemas.py` | Оценки S, M, O, P + вычисляемое Q |
| `RunEvaluation` | `schemas.py` | Оценка одного прогона генерации |
| `TaskEvaluation` | `schemas.py` | Агрегация по задаче (все прогоны) |
| `ExperimentEvaluation` | `schemas.py` | Оценка эксперимента (все задачи) |
| `SMOPCriteria` | `smop.py` | Загрузчик критериев оценки |
| `SMOPEvaluator` | `smop.py` | Менеджер оценок (CRUD) |
| `StatisticsCalculator` | `statistics.py` | Расчёт статистики |
| `ChartGenerator` | `charts.py` | Генератор графиков |
| `ReportGenerator` | `report.py` | Генератор отчётов |

---

## SMOP: Экспертная оценка

### Концепция

**SMOP** — методика экспертной оценки качества сгенерированного кода по четырём критериям:

| Критерий | Название | Описание | Вес |
|----------|----------|----------|-----|
| **S** | Syntax | Синтаксическая корректность | 1.0 |
| **M** | Meaning | Семантическая корректность | 1.0 |
| **O** | Optimization | Оптимальность и соответствие стандартам | 1.0 |
| **P** | Platform | Платформенная интеграция (метаданные) | 1.0 |

### Шкала оценки

Каждый критерий оценивается по **дискретной шкале**:

```
{0, 2, 4, 6, 8, 10}
```

Где:
- **10** — идеальное выполнение критерия
- **8** — незначительные недочёты
- **6** — требуются доработки
- **4** — существенные проблемы
- **2** — критические проблемы
- **0** — критерий не выполнен

### Интегральный показатель Q

Вычисляется как **среднее арифметическое** всех четырёх критериев:

$$
Q = \frac{S + M + O + P}{4}
$$

#### Уровни качества

| Диапазон Q | Уровень | Интерпретация |
|------------|---------|---------------|
| $Q \geq 8$ | **Высокое** | Код готов к использованию |
| $5 \leq Q < 8$ | **Приемлемое** | Требуются минорные доработки |
| $Q < 5$ | **Низкое** | Требуется значительная переработка |

### Критерии оценки (из `smop_criteria.yaml`)

#### S — Синтаксис

| Оценка | Описание |
|--------|----------|
| 10 | Код компилируется без ошибок и предупреждений |
| 8 | Код компилируется, 1–2 незначительных предупреждения |
| 6 | Требуется исправление 1–2 опечаток |
| 4 | Требуется исправление 3–5 синтаксических ошибок |
| 2 | Более 5 ошибок, но структура понятна |
| 0 | Код не компилируется, требуется переработка |

#### M — Семантика

| Оценка | Описание |
|--------|----------|
| 10 | Код полностью выполняет поставленную задачу |
| 8 | Основная логика верна, не выполнено одно второстепенное требование |
| 6 | Основная логика верна, не реализованы 2–3 требования |
| 4 | Реализовано более 50% логики |
| 2 | Реализовано менее 50% логики |
| 0 | Код не выполняет задачу |

#### O — Оптимальность

| Оценка | Описание |
|--------|----------|
| 10 | Код соответствует стандартам 1С, оптимален по производительности |
| 8 | Незначительные отклонения от стандартов |
| 6 | Код работоспособен, есть возможности оптимизации |
| 4 | Неэффективные решения: запросы в цикле, избыточные обращения к БД |
| 2 | Множественные антипаттерны, код медленный |
| 0 | Критические проблемы производительности |

#### P — Платформа

| Оценка | Описание |
|--------|----------|
| 10 | Все объекты метаданных и методы платформы использованы корректно |
| 8 | 1–2 неточности в именах реквизитов, легко исправимые |
| 6 | Ошибки в работе с типами данных или структурой объектов |
| 4 | До 30% обращений к несуществующим объектам |
| 2 | Более 30% обращений к несуществующим объектам |
| 0 | Код построен на вымышленной структуре конфигурации |

### Реализация

#### Класс `SMOPScores`

```python
class SMOPScores(BaseModel):
    S: Optional[int] = None  # Синтаксис
    M: Optional[int] = None  # Семантика
    O: Optional[int] = None  # Оптимальность
    P: Optional[int] = None  # Платформа
    
    @computed_field
    @property
    def Q(self) -> Optional[float]:
        """Интегральный показатель Q = (S + M + O + P) / 4"""
        scores = [self.S, self.M, self.O, self.P]
        filled = [s for s in scores if s is not None]
        if not filled:
            return None
        return sum(filled) / len(filled)
```

**Особенности**:
- Автоматическая валидация значений через Pydantic
- Q вычисляется динамически (computed field)
- Поддержка частичного заполнения (Q рассчитывается по доступным критериям)

#### Класс `SMOPEvaluator`

Менеджер для работы с оценками:

```python
evaluator = SMOPEvaluator("evaluations")

# Установить оценку
evaluator.set_score(
    evaluation,
    task_id="A1",
    model_id="google/gemini",
    run_index=0,
    metric="S",
    score=10
)

# Сохранить
evaluator.save(evaluation)
```

**Функции**:
- `set_score()` — установка одного критерия
- `set_all_scores()` — пакетная установка всех SMOP
- `save()` — сохранение в JSON с автоматическим обновлением статуса
- `load()` — восстановление сессии оценки

---

## Статистический анализ

Модуль `statistics.py` предоставляет инструменты для вычисления описательных статистик и проверки гипотез.

### Основные метрики

#### 1. Среднее арифметическое

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

```python
def calculate_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
```

#### 2. Медиана

$$
\tilde{x} = \begin{cases}
x_{(n+1)/2} & \text{если } n \text{ нечётное} \\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{если } n \text{ чётное}
\end{cases}
$$

```python
def calculate_median(values: List[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]
```

#### 3. Дисперсия (несмещённая оценка)

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

Где:
- $n$ — размер выборки
- $x_i$ — значение $i$-го элемента
- $\bar{x}$ — среднее арифметическое
- $n-1$ — поправка Бесселя (для несмещённости)

#### 4. Стандартное отклонение (несмещённая оценка)

$$
s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

```python
def calculate_std(values: List[float], mean: Optional[float] = None) -> float:
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
```

#### 5. Коэффициент вариации

$$
CV = \frac{s}{\bar{x}} \times 100\%
$$

Показывает относительную изменчивость данных. Используется для сравнения разброса выборок с различными средними значениями.

#### 6. Доверительный интервал (95%)

Используется **t-распределение Стьюдента** для малых выборок:

$$
CI_{95\%} = \left[\bar{x} - t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}, \quad \bar{x} + t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}\right]
$$

Где:
- $\bar{x}$ — среднее выборки
- $t_{\alpha/2, n-1}$ — критическое значение t-распределения (для $\alpha = 0.05$, двусторонний тест)
- $s$ — стандартное отклонение
- $n$ — размер выборки
- $\frac{s}{\sqrt{n}}$ — стандартная ошибка среднего (SEM)

**Стандартная ошибка среднего**:

$$
SEM = \frac{s}{\sqrt{n}}
$$

```python
def calculate_ci_95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        mean = calculate_mean(values)
        return (mean, mean)
    
    mean = calculate_mean(values)
    std = calculate_std(values, mean)
    
    # t-критические значения (приближение)
    t_critical = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        # ... до 100: 1.984
    }
    
    t = 1.96  # По умолчанию (большие выборки)
    for k in sorted(t_critical.keys()):
        if n <= k:
            t = t_critical[k]
            break
    
    margin = t * (std / math.sqrt(n))
    return (mean - margin, mean + margin)
```

**Особенности реализации**:
- Автоматический выбор t-критерия в зависимости от размера выборки
- Для $n \geq 100$ используется стандартное нормальное распределение ($z = 1.96$)

#### 7. Квартили (для Boxplot)

**Первый квартиль (Q1, 25-й процентиль)**:

$$
Q_1 = x_{\lfloor 0.25 \cdot (n+1) \rfloor}
$$

**Третий квартиль (Q3, 75-й процентиль)**:

$$
Q_3 = x_{\lfloor 0.75 \cdot (n+1) \rfloor}
$$

**Межквартильный размах (IQR)**:

$$
IQR = Q_3 - Q_1
$$

**Границы выбросов**:

$$
\begin{aligned}
\text{Нижняя граница} &= Q_1 - 1.5 \cdot IQR \\
\text{Верхняя граница} &= Q_3 + 1.5 \cdot IQR
\end{aligned}
$$

Значения за пределами этих границ считаются выбросами.

#### 8. Минимум и максимум

$$
\begin{aligned}
x_{min} &= \min_{i=1,\ldots,n} x_i \\
x_{max} &= \max_{i=1,\ldots,n} x_i
\end{aligned}
$$

**Размах (Range)**:

$$
R = x_{max} - x_{min}
$$

### Агрегация по моделям

Метод `aggregate_by_model()` класса `StatisticsCalculator`:

**Выходные данные**:

```python
ModelSummary(
    model_id="google/gemini",
    model_name="Gemini 3 Flash",
    tasks_count=5,
    runs_count=15,
    S=MetricStats(mean=8.5, std=1.2, median=9.0, min=6, max=10, count=15),
    M=MetricStats(...),
    O=MetricStats(...),
    P=MetricStats(...),
    Q=QualityStats(
        mean=8.2,
        std=1.1,
        median=8.5,
        min=6.0,
        max=10.0,
        ci_lower=7.6,
        ci_upper=8.8,
        count=15
    ),
    determinism_mean=85.3
)
```

### Корреляция: Детерминизм vs Качество

Вычисляется **коэффициент корреляции Пирсона**:

$$
r_{xy} = \frac{\text{Cov}(X, Y)}{\sigma_x \cdot \sigma_y} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

Где:
- $x_i$ — процент детерминизма для задачи $i$
- $y_i$ — средний показатель $Q$ для задачи $i$
- $\bar{x}, \bar{y}$ — средние значения
- $\text{Cov}(X, Y)$ — ковариация
- $\sigma_x, \sigma_y$ — стандартные отклонения

**Ковариация**:

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

**Альтернативная формула** (вычислительно удобная):

$$
r_{xy} = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{\sqrt{n\sum x_i^2 - (\sum x_i)^2} \cdot \sqrt{n\sum y_i^2 - (\sum y_i)^2}}
$$

```python
def calculate_correlation_det_quality(self) -> Optional[float]:
    pairs = []  # [(det, Q), ...]
    
    for task in self.evaluation.tasks:
        det = self._determinism_cache.get((task.task_id, task.model_id))
        if det and task.avg_Q is not None:
            pairs.append((det, task.avg_Q))
    
    if len(pairs) < 3:
        return None
    
    # Коэффициент Пирсона
    n = len(pairs)
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denom_x == 0 or denom_y == 0:
        return None
    
    return numerator / (denom_x * denom_y)
```

**Интерпретация**:
- $r = 1$ — полная положительная линейная связь
- $r \approx 0.7 \ldots 1$ — сильная положительная связь
- $r \approx 0.3 \ldots 0.7$ — умеренная положительная связь
- $r \approx 0 \ldots 0.3$ — слабая положительная связь
- $r = 0$ — нет линейной связи
- $r < 0$ — отрицательная связь (симметрично)
- $r = -1$ — полная отрицательная линейная связь

**Коэффициент детерминации** ($R^2$):

$$
R^2 = r_{xy}^2
$$

Показывает долю дисперсии $Y$, объясняемую линейной связью с $X$ (в процентах: $R^2 \times 100\%$).

### Межэкспертная надёжность (Каппа Коэна)

Используется для оценки согласованности между двумя экспертами.

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Где:
- $p_o$ — наблюдаемое согласие (доля совпадений)
- $p_e$ — ожидаемое случайное согласие

#### Алгоритм расчёта

Пусть $C = [c_{ij}]$ — матрица совпадений размера $k \times k$, где $c_{ij}$ — количество случаев, когда эксперт 1 присвоил категорию $i$, а эксперт 2 — категорию $j$.

**1. Наблюдаемое согласие**:

$$
p_o = \frac{1}{N} \sum_{i=1}^{k} c_{ii}
$$

где $N = \sum_{i=1}^{k} \sum_{j=1}^{k} c_{ij}$ — общее число оценок.

**2. Ожидаемое случайное согласие**:

$$
p_e = \frac{1}{N^2} \sum_{i=1}^{k} \left(\sum_{j=1}^{k} c_{ij}\right) \cdot \left(\sum_{j=1}^{k} c_{ji}\right)
$$

Или, через маргинальные суммы:

$$
p_e = \sum_{i=1}^{k} P(\text{Эксперт}_1 = i) \cdot P(\text{Эксперт}_2 = i)
$$

где:

$$
\begin{aligned}
P(\text{Эксперт}_1 = i) &= \frac{1}{N} \sum_{j=1}^{k} c_{ij} \\
P(\text{Эксперт}_2 = i) &= \frac{1}{N} \sum_{j=1}^{k} c_{ji}
\end{aligned}
$$

**3. Каппа Коэна**:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

**Граничные случаи**:
- Если $p_o = 1$ (полное согласие), то $\kappa = 1$
- Если $p_o = p_e$ (согласие на уровне случайности), то $\kappa = 0$
- Если $p_o < p_e$ (согласие хуже случайного), то $\kappa < 0$

```python
def _cohens_kappa(self, pairs: List[Tuple[int, int]]) -> Optional[float]:
    n = len(pairs)
    categories = sorted(set(p[0] for p in pairs) | set(p[1] for p in pairs))
    k = len(categories)
    
    # Матрица совпадений
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * k for _ in range(k)]
    
    for r1, r2 in pairs:
        i, j = cat_idx[r1], cat_idx[r2]
        matrix[i][j] += 1
    
    # Наблюдаемое согласие
    p_o = sum(matrix[i][i] for i in range(k)) / n
    
    # Ожидаемое случайное согласие
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)
    
    if p_e == 1:
        return 1.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa
```

**Интерпретация** (Landis & Koch, 1977):

| Значение $\kappa$ | Уровень согласия |
|-------------------|------------------|
| $\kappa < 0.00$ | Нет согласия (хуже случайности) |
| $0.00 \leq \kappa \leq 0.20$ | Незначительное |
| $0.21 \leq \kappa \leq 0.40$ | Слабое |
| $0.41 \leq \kappa \leq 0.60$ | Умеренное |
| $0.61 \leq \kappa \leq 0.80$ | Существенное |
| $0.81 \leq \kappa \leq 1.00$ | Почти полное |

**Стандартная ошибка каппы** (для проверки значимости):

$$
SE_\kappa = \sqrt{\frac{p_o(1-p_o)}{N(1-p_e)^2}}
$$

**z-тест** (проверка $H_0: \kappa = 0$):

$$
z = \frac{\kappa}{SE_\kappa}
$$

При $|z| > 1.96$ гипотеза $\kappa = 0$ отвергается на уровне значимости $\alpha = 0.05$.

---

## Визуализация результатов

Модуль `charts.py` использует **matplotlib** для создания научных графиков.

### Стили публикации

```python
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}
```

### Математические основы визуализаций

#### Нормализация для цветовых шкал

Для тепловых карт используется **min-max нормализация**:

$$
x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

где $x_{norm} \in [0, 1]$ используется для выбора цвета из палитры.

#### Радарная диаграмма (полигональная)

Координаты точки на радарной диаграмме для метрики $i$ со значением $v_i$:

$$
\begin{aligned}
\theta_i &= \frac{2\pi \cdot i}{k} \quad \text{(угол)} \\
r_i &= \frac{v_i}{v_{max}} \quad \text{(радиус)} \\
x_i &= r_i \cdot \cos(\theta_i) \\
y_i &= r_i \cdot \sin(\theta_i)
\end{aligned}
$$

где $k$ — количество метрик (для SMOP: $k = 4$).

#### Доверительные интервалы на графиках

Визуализация error bars с использованием стандартного отклонения:

$$
\text{error}_i = \pm t_{\alpha/2, n-1} \cdot \frac{s_i}{\sqrt{n_i}}
$$

где для каждой модели $i$ рассчитывается свой доверительный интервал.

### Доступные типы графиков

#### 1. Радарная диаграмма SMOP

```python
generator.plot_smop_radar()
```

Показывает профиль каждой модели по метрикам S, M, O, P на радарной диаграмме.

**Файл**: `{experiment_id}_radar.png`

#### 2. Столбчатое сравнение моделей

```python
generator.plot_models_comparison()
```

Горизонтальные столбцы для каждой модели с разбивкой по S, M, O, P.

**Файл**: `{experiment_id}_models_comparison.png`

#### 3. Интегральный Q по моделям

```python
generator.plot_q_by_model()
```

Столбчатая диаграмма с доверительными интервалами и пороговыми линиями (Q ≥ 8, Q ≥ 5).

**Файл**: `{experiment_id}_q_by_model.png`

#### 4. Распределение оценок (4 гистограммы)

```python
generator.plot_scores_distribution()
```

Показывает распределение значений для S, M, O, P отдельно.

**Файл**: `{experiment_id}_distribution.png`

#### 5. Boxplot Q по моделям

```python
generator.plot_boxplot_by_model()
```

Показывает медиану, квартили, выбросы для Q каждой модели.

**Файл**: `{experiment_id}_boxplot.png`

#### 6. Тепловая карта (Задачи × Модели)

```python
generator.plot_heatmap_tasks_models()
```

Цветовая карта значений Q для каждой комбинации задачи и модели.

**Файл**: `{experiment_id}_heatmap.png`

#### 7. Детерминизм vs Качество

```python
generator.plot_determinism_vs_quality()
```

Scatter plot с корреляцией между процентом детерминизма и Q.

**Файл**: `{experiment_id}_det_vs_quality.png`

#### 8. Сводный дашборд

```python
generator.plot_summary_dashboard()
```

Комплексная визуализация: Q с CI, метрики SMOP, прогресс оценки, распределение.

**Файл**: `{experiment_id}_dashboard.png`

### Генерация всех графиков

```python
from src.evaluator.charts import generate_charts

paths = generate_charts(
    evaluation,
    experiment,
    output_dir="reports/charts/experiment_A_123",
    formats=["png", "svg", "pdf"]
)
```

**Результат**:

```python
{
    "dashboard": [Path("...dashboard.png"), Path("...dashboard.svg")],
    "radar": [Path("...radar.png"), Path("...radar.svg")],
    # ...
}
```

---

## Генерация отчётов

Модуль `report.py` создаёт итоговые артефакты для публикации.

### Типы отчётов

#### 1. JSON-отчёт

Структурированные данные со всей статистикой:

```json
{
  "experiment_id": "experiment_A_20260205_232949",
  "generated_at": "2026-02-06T12:00:00",
  "summary": {
    "overall_Q": {
      "mean": 7.8,
      "std": 1.2,
      "ci_lower": 7.3,
      "ci_upper": 8.3
    },
    "by_metric": {
      "S": {"mean": 8.5, "std": 1.1},
      "M": {"mean": 7.2, "std": 1.5},
      "O": {"mean": 7.5, "std": 1.3},
      "P": {"mean": 8.0, "std": 1.2}
    }
  },
  "by_model": [...],
  "by_task": [...]
}
```

#### 2. HTML-отчёт

Интерактивный веб-отчёт с тёмной темой:

- Карточки с ключевыми метриками
- Таблицы сравнения моделей
- Таблицы по задачам
- Встроенные графики (Base64)

**Генерация**:

```python
from src.evaluator.report import generate_html_report

html_path = generate_html_report(
    evaluation,
    experiment,
    output_file="reports/experiment_A_123_report.html"
)
```

#### 3. LaTeX-таблицы

Экспорт таблиц для вставки в статью:

```latex
\begin{table}[h]
\centering
\caption{Сравнение моделей по критериям SMOP}
\begin{tabular}{lcccccc}
\toprule
Модель & S & M & O & P & Q & $\sigma_Q$ \\
\midrule
Claude Opus 4.5 & 8.5 & 7.8 & 8.2 & 8.0 & 8.13 & 0.92 \\
GPT-5.2 Codex & 9.0 & 8.5 & 8.8 & 7.5 & 8.45 & 1.05 \\
Gemini 3 Flash & 7.8 & 7.2 & 7.5 & 8.2 & 7.68 & 1.20 \\
\bottomrule
\end{tabular}
\end{table}
```

**Генерация**:

```python
from src.evaluator.report import generate_latex_tables

latex_path = generate_latex_tables(
    evaluation,
    experiment,
    output_file="reports/experiment_A_123_tables.tex"
)
```

### Класс `ReportGenerator`

```python
from src.evaluator.report import ReportGenerator

generator = ReportGenerator(evaluation, experiment)

# Все артефакты сразу
artifacts = generator.generate_all_reports(
    output_dir="reports",
    prefix="experiment_A_123"
)

# artifacts = {
#     "json": Path("...report.json"),
#     "html": Path("...report.html"),
#     "latex": Path("...tables.tex"),
#     "charts": {...}
# }
```

---

## Схемы данных

### `SMOPScores`

```python
class SMOPScores(BaseModel):
    S: Optional[int] = None
    M: Optional[int] = None
    O: Optional[int] = None
    P: Optional[int] = None
    
    @computed_field
    @property
    def Q(self) -> Optional[float]:
        scores = [self.S, self.M, self.O, self.P]
        filled = [s for s in scores if s is not None]
        return sum(filled) / len(filled) if filled else None
```

### `RunEvaluation`

```python
class RunEvaluation(BaseModel):
    run_index: int
    response_hash: str
    scores: SMOPScores
    comment: str = ""
    evaluated_at: Optional[datetime] = None
```

### `TaskEvaluation`

```python
class TaskEvaluation(BaseModel):
    task_id: str
    task_name: str
    model_id: str
    model_name: str
    runs: List[RunEvaluation]
    
    @computed_field
    @property
    def avg_Q(self) -> Optional[float]:
        qs = [r.scores.Q for r in self.runs if r.scores.Q is not None]
        return sum(qs) / len(qs) if qs else None
```

### `ExperimentEvaluation`

```python
class ExperimentEvaluation(BaseModel):
    experiment_id: str
    evaluator_id: str
    tasks: List[TaskEvaluation]
    started_at: datetime
    last_modified_at: Optional[datetime]
    status: Literal["in_progress", "completed"]
    
    @computed_field
    @property
    def progress_percent(self) -> float:
        return (self.evaluated_runs / self.total_runs * 100) 
            if self.total_runs > 0 else 0.0
```

### `MetricStats`

```python
class MetricStats(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0
```

### `QualityStats`

```python
class QualityStats(MetricStats):
    ci_lower: float = 0.0  # Нижняя граница 95% ДИ
    ci_upper: float = 0.0  # Верхняя граница 95% ДИ
```

---

## API Reference

### `SMOPEvaluator`

#### `__init__(evaluations_dir, criteria_path)`

Инициализация менеджера оценок.

**Параметры**:
- `evaluations_dir` (str): Директория для сохранения оценок
- `criteria_path` (str): Путь к `smop_criteria.yaml`

#### `load(experiment_id, evaluator_id) -> ExperimentEvaluation`

Загрузить оценку из JSON.

#### `save(evaluation) -> Path`

Сохранить оценку в JSON с обновлением статуса.

#### `set_score(evaluation, task_id, model_id, run_index, metric, score, comment)`

Установить оценку для одного критерия.

**Параметры**:
- `metric` (Literal["S", "M", "O", "P"]): Критерий
- `score` (int): Значение из {0, 2, 4, 6, 8, 10}

**Возвращает**: `bool` — успешность операции

#### `set_all_scores(evaluation, task_id, model_id, run_index, scores, comment)`

Установить все SMOP-оценки одновременно.

**Параметры**:
- `scores` (Dict[str, int]): `{"S": 10, "M": 8, "O": 6, "P": 8}`

### `StatisticsCalculator`

#### `__init__(evaluation, experiment)`

Инициализация калькулятора.

**Параметры**:
- `evaluation` (ExperimentEvaluation): Оценки
- `experiment` (ExperimentResult, optional): Сырые результаты для детерминизма

#### `calculate_summary() -> Dict[str, Any]`

Рассчитать общую сводку.

#### `aggregate_by_model() -> List[ModelSummary]`

Агрегация метрик по моделям.

#### `aggregate_by_task() -> List[TaskSummary]`

Агрегация метрик по задачам.

#### `calculate_correlation_det_quality() -> Optional[float]`

Коэффициент корреляции Пирсона между детерминизмом и Q.

#### `calculate_inter_rater_reliability(other_evaluation) -> Optional[Dict[str, float]]`

Каппа Коэна для каждой метрики.

**Возвращает**: `{"S": 0.85, "M": 0.78, "O": 0.82, "P": 0.79}`

### `ChartGenerator`

#### `__init__(evaluation, experiment, output_dir, formats)`

Инициализация генератора графиков.

**Параметры**:
- `formats` (List[str]): Список форматов ["png", "svg", "pdf"]

#### `plot_smop_radar() -> List[Path]`

Радарная диаграмма SMOP.

#### `plot_models_comparison() -> List[Path]`

Столбчатое сравнение моделей.

#### `plot_q_by_model() -> List[Path]`

Q по моделям с доверительными интервалами.

#### `generate_all() -> Dict[str, List[Path]]`

Сгенерировать все графики.

### `ReportGenerator`

#### `generate_json_report(evaluation, experiment, output_file) -> Path`

Генерация JSON-отчёта.

#### `generate_html_report(evaluation, experiment, output_file) -> Path`

Генерация HTML-отчёта.

#### `generate_latex_tables(evaluation, experiment, output_file) -> Path`

Экспорт таблиц в LaTeX.

---

## Примеры использования

### Пример 1: Проставление оценок

```python
from src.evaluator.smop import SMOPEvaluator
from src.evaluator.parser import parse_experiment_for_evaluation

# 1. Загрузить сырые результаты
experiment = load_json("raw_results/experiment_A_123.json")

# 2. Создать структуру оценки
evaluation = parse_experiment_for_evaluation(experiment, "expert_01")

# 3. Инициализировать оценщик
evaluator = SMOPEvaluator("evaluations")

# 4. Проставить оценки
evaluator.set_all_scores(
    evaluation,
    task_id="A1",
    model_id="google/gemini",
    run_index=0,
    scores={"S": 10, "M": 8, "O": 8, "P": 10},
    comment="Отличный код, минорные замечания по оптимизации"
)

# 5. Сохранить
evaluator.save(evaluation)
```

### Пример 2: Статистический анализ

```python
from src.evaluator.statistics import StatisticsCalculator

# 1. Загрузить оценку
evaluation = evaluator.load("experiment_A_123", "expert_01")

# 2. Загрузить сырые результаты (для детерминизма)
experiment = load_json("raw_results/experiment_A_123.json")

# 3. Создать калькулятор
calc = StatisticsCalculator(evaluation, experiment)

# 4. Общая сводка
summary = calc.calculate_summary()
print(f"Средний Q: {summary['overall_Q']['mean']:.2f}")
print(f"95% ДИ: [{summary['overall_Q']['ci_lower']:.2f}, "
      f"{summary['overall_Q']['ci_upper']:.2f}]")

# 5. По моделям
for model in calc.aggregate_by_model():
    print(f"{model.model_name}: Q={model.Q.mean:.2f} ± {model.Q.std:.2f}")

# 6. Корреляция
corr = calc.calculate_correlation_det_quality()
print(f"Корреляция детерминизм-качество: r={corr:.3f}")
```

### Пример 3: Генерация отчётов

```python
from src.evaluator.report import ReportGenerator

# 1. Создать генератор
generator = ReportGenerator(evaluation, experiment)

# 2. Все артефакты сразу
artifacts = generator.generate_all_reports(
    output_dir="reports",
    prefix="experiment_A_123"
)

print(f"JSON: {artifacts['json']}")
print(f"HTML: {artifacts['html']}")
print(f"LaTeX: {artifacts['latex']}")
print(f"Графики: {len(artifacts['charts'])} файлов")
```

### Пример 4: CLI-оценка через dashboard

```bash
# Запустить интерактивный интерфейс оценки
python main.py evaluate experiment_A_123 --evaluator expert_01

# Возобновить прерванную сессию (автоматическая загрузка)
python main.py evaluate experiment_A_123 --evaluator expert_01 --resume

# Сгенерировать отчёт после завершения оценки
python main.py report experiment_A_123
```

**Интерфейс CLI**:

![CLI Entrypoint](entrypoint_cli.png)

*Рисунок 1: Главное меню CLI-интерфейса фреймворка*

**Интерактивный TUI для оценки SMOP**:

![SMOP Evaluator TUI](smop_evaluator_tui_preview.png)

*Рисунок 2: Терминальный интерфейс (TUI) для экспертной оценки кода по методике SMOP*

### Пример 5: Межэкспертная надёжность

```python
# Два эксперта оценили один эксперимент
evaluation1 = evaluator.load("experiment_A_123", "expert_01")
evaluation2 = evaluator.load("experiment_A_123", "expert_02")

# Расчёт Каппа Коэна
calc = StatisticsCalculator(evaluation1)
kappa = calc.calculate_inter_rater_reliability(evaluation2)

for metric, k in kappa.items():
    print(f"{metric}: κ={k:.3f}")

# Пример вывода:
# S: κ=0.850 (почти полное согласие)
# M: κ=0.720 (существенное согласие)
# O: κ=0.680 (существенное согласие)
# P: κ=0.780 (существенное согласие)
```

---

## Заключение

Модуль **`evaluator`** предоставляет полный цикл обработки результатов эксперимента:

1. **Экспертная оценка** — методика SMOP с валидацией и автосохранением
2. **Статистический анализ** — описательные статистики, ДИ, корреляции, межэкспертная надёжность
3. **Визуализация** — научные графики в высоком разрешении (PNG, SVG, PDF)
4. **Отчёты** — JSON, HTML, LaTeX для публикации

Все компоненты интегрированы и обеспечивают воспроизводимость результатов исследования.

---

## Скриншоты интерфейсов

### CLI Entry Point

Главный интерфейс командной строки для управления экспериментами:

![CLI Entrypoint](entrypoint_cli.png)

**Возможности CLI**:
- Запуск экспериментов с различными конфигурациями
- Просмотр информации о балансе, моделях, задачах
- Генерация отчётов
- Управление экспертными оценками

### SMOP Evaluator TUI

Терминальный интерфейс пользователя (TUI) для проведения экспертной оценки:

![SMOP Evaluator TUI](smop_evaluator_tui_preview.png)

**Функции TUI**:
- Интерактивная навигация по задачам и прогонам
- Просмотр сгенерированного кода
- Проставление оценок S, M, O, P
- Автоматический расчёт Q
- Добавление комментариев эксперта
- Индикация прогресса оценки
- Автосохранение при выходе
- Горячие клавиши для ускорения работы

---

**Следующие разделы документации**:
- `docs/architecture.md` — Общая архитектура фреймворка
- `docs/methodology.md` — Научная методология эксперимента
- `docs/api.md` — Полное API фреймворка
