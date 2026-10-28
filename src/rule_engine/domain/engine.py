"""Проход потока токенов по графу сценариев и по шаблонам `any`.

Событие трогает только то, что его ждёт: рёбра бора и шаблоны `any` с его токеном находятся по
индексу, а снятые по сроку совпадения — по очереди сроков. Совпадения одного места хранит
`Front`: в нём они упорядочены так, что лучшее продолжение и истекающие сроки лежат с краёв.
"""

import heapq
import math
from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import count, islice
from typing import NamedTuple

from rule_engine.domain.graph import Edge, Node, Start, insert_scenario
from rule_engine.domain.rules import RuleSet, Scenario, parse_scenario
from rule_engine.domain.unordered import Unordered, build_unordered, insert_unordered

# сроки в очереди — суммы с округлением, а снятие проверяется точной формулой, поэтому из очереди
# достаются и сроки чуть позже события; доля от времени события
SLACK = 1e-9


@dataclass(frozen=True)
class Event:
    token: str
    score: float  # уверенность детектора: шаг принимает токен, только если она не ниже порога
    time: float  # секунды: когда токен сказан, по одной шкале для всего потока (ставит детектор)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"уверенность {self.score} токена {self.token!r} вне [0, 1]")
        if not math.isfinite(self.time):
            raise ValueError(f"время {self.time} токена {self.token!r} не конечное число")


class Step(NamedTuple):
    position: int  # сквозной номер события в потоке, между порциями не сбрасывается
    event: Event


@dataclass(frozen=True)
class Match:
    scenario: str
    steps: tuple[Step, ...]
    confidence: float  # общая уверенность: среднее геометрическое score шагов


class Partial(NamedTuple):
    """Частичное совпадение: его шаги и произведение их score."""

    steps: tuple[Step, ...]
    product: float

    @property
    def confidence(self) -> float:
        """Среднее геометрическое score шагов."""
        return self.product ** (1 / len(self.steps))

    def bound(self, length: int) -> float:
        """Уверенность в шаблоне длины `length`, если у оставшихся шагов score 1.0: выше не бывает."""
        return self.product ** (1 / length)

    def extend(self, step: Step) -> "Partial":
        return Partial(self.steps + (step,), self.product * step.event.score)

    def positions(self) -> tuple[int, ...]:
        return tuple(step.position for step in self.steps)


class Waiting(NamedTuple):
    """Частичное совпадение в снимке движка: где оно стоит и до какого момента ждёт."""

    place: Node | tuple[Unordered, frozenset[int]]  # вершина бора или (шаблон `any`, закрытые шаги)
    partial: Partial
    until: float  # секунды: срок продолжения


class Miss(NamedTuple):
    """Почему событие не продвинуло совпадение или не дало сработать сценарию.

    - `"score"`: score токена `got` ниже порога шага `need`. `place` — куда не пустило: вершина
      за ребром или состояние набора `any`.
    - `"late"`: токен подошёл бы, но пришёл в `got`, а ждали до `need`. `place` — как у `"score"`.
    - `"expired"`: совпадение снято: ждали до `need`, а токен в `got` ему не подходил. `place` —
      где оно стояло.
    - `"pruned"`: совпадение отсеяно: даже при score 1.0 у оставшихся шагов уверенность не выше
      `got`, а порог `need`. `place` — куда оно вошло.
    - `"confidence"`: шаблон сценария собран, но общая уверенность `got` ниже порога `need`.
    - `"repeat"`: шаблон собран, но его токен уже вошёл в выданное срабатывание сценария.
    - `"cooldown"`: шаблон собран, но сценарий молчит до `need`.

    `partial` — совпадение, которое не продвинулось; `None` — новое совпадение не началось.
    У трёх последних причин есть `scenario`, а `place` — `None`.
    """

    reason: str
    place: Node | tuple[Unordered, frozenset[int]] | None
    partial: Partial | None
    scenario: str | None
    need: float
    got: float


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: последний шаг у `a` не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и последний шаг, и произведение, остаётся начавшееся позже, а при равном начале — с более
    поздними шагами: так исход не зависит от того, в каком порядке совпадения пришли в вершину.
    """
    if (a.steps[-1].position, a.product) == (b.steps[-1].position, b.product):
        return a.positions() >= b.positions()
    return a.steps[-1].position >= b.steps[-1].position and a.product >= b.product


class Collector:
    """Шаблон `any` в движке: группы одинаковых шагов и состояния, где стоят совпадения."""

    __slots__ = ("pattern", "groups", "full", "states")

    def __init__(self, pattern: Unordered) -> None:
        self.pattern = pattern
        self.groups = pattern.groups()
        self.full = (1 << len(pattern.slots)) - 1  # маска собранного шаблона
        self.states: dict[int, Front] = {}  # маска закрытых шагов → совпадения, без пустых

    def free(self, filled: int, token: str, score: float) -> list[int]:
        """Номера шагов, которые событие может закрыть, если закрыты шаги из маски `filled`."""
        free = []
        for min_score, mask in self.groups.get(token, ()):
            open_ = mask & ~filled
            if open_ and score >= min_score:
                free.append((open_ & -open_).bit_length() - 1)
        return sorted(free)

    def place(self, filled: int) -> tuple[Unordered, frozenset[int]]:
        """Состояние так, как его видят снимок и отказы: шаблон и номера закрытых шагов."""
        slots = range(len(self.pattern.slots))
        return self.pattern, frozenset(index for index in slots if filled >> index & 1)


class Front:
    """Несравнимые совпадения одного места: вершины бора или состояния набора `any`.

    Совпадения идут по возрастанию последнего шага, а произведение score у них строго убывает:
    более старое остаётся, только если оно увереннее (`dominates`). Поэтому новое совпадение
    вытесняет только хвост, сроки истекают с головы, а самое уверенное стоит первым.
    """

    __slots__ = ("place", "items", "head")

    def __init__(self, place: Node | tuple[Collector, int]) -> None:
        self.place = place
        self.items: list[Partial] = []
        self.head = 0  # совпадения до `head` сняты по сроку

    def __len__(self) -> int:
        return len(self.items) - self.head

    def __iter__(self) -> Iterator[Partial]:
        return islice(self.items, self.head, None)

    def first(self) -> Partial:
        return self.items[self.head]

    def add(self, partial: Partial) -> int:
        """Добавляет совпадение, кончающееся на текущем событии; возвращает, насколько вырос фронт.

        Вытеснить новое может только последнее: лишь у него последний шаг может быть тем же.
        """
        items = self.items
        if len(self) and dominates(items[-1], partial):
            return 0
        size = len(self)
        while len(self) and dominates(partial, items[-1]):
            items.pop()
        items.append(partial)
        return len(self) - size

    def pop_first(self) -> Partial:
        partial = self.items[self.head]
        self.head += 1
        if self.head * 2 >= len(self.items):  # сжатие: в среднем O(1) на снятое совпадение
            del self.items[: self.head]
            self.head = 0
        return partial

    def since(self, fits: Callable[[Partial], bool]) -> int:
        """Номер первого совпадения, для которого `fits` истинно; дальше оно истинно у всех."""
        return bisect_left(self.items, True, self.head, key=fits)


class Engine:
    """Частичные совпадения хранятся между вызовами, поэтому поток можно подавать порциями.

    - Токен, не продолжающий частичное совпадение, пропускается: совпадение ждёт дальше.
      Это относится и к токенам вне словаря: их позиция и время учитываются как обычно.
    - У ребра есть задержка: совпадение продолжается по нему, только если от последнего шага
      совпадения прошло не больше неё. Посторонние токены этот срок не продлевают. Когда сроки
      всех рёбер из вершины истекли, совпадение снимается. Новое совпадение от корня задержку
      не проверяет.
    - У ребра есть порог: токен проходит по нему, только если его score не ниже порога. Иначе
      для этого ребра токена нет, и совпадение ждёт дальше. Новое совпадение тоже начинается,
      только если score токена не ниже порога ребра из корня.
    - Совпадение, продолжившееся по ребру, остаётся и в старой вершине: копия ждёт другого
      продолжения. Поэтому лишний токен со своим ребром не уводит совпадение в чужую ветку.
    - Уверенность совпадения — среднее геометрическое score его шагов. Сценарий срабатывает,
      только если она не ниже `min_confidence` сценария.
    - В вершине стоят только несравнимые совпадения: одно вытесняет другое, если у него
      последний шаг не раньше, а произведение score не меньше (`dominates`). Поэтому рядом могут
      стоять свежее совпадение и более уверенное.
    - Совпадение не остаётся в вершине, если даже при score 1.0 у всех оставшихся шагов оно не
      дотянет до порога ни одного сценария, чей шаблон кончается ниже (ранний отсев).
    - Сценарий срабатывает при входе в вершину, где заканчивается один из его шаблонов. Если
      из неё есть продолжения (более длинные шаблоны с тем же началом), совпадение остаётся
      активным.
    - Шаблоны сценариев с `order: "any"` идут мимо бора. Их совпадение — набор закрытых шагов:
      токен закрывает любой ещё открытый шаг со своим токеном, если проходит его порог и
      укладывается в задержку от последнего шага. `<дверь> <шаги>` и `<шаги> <дверь>` дают одно
      состояние `{<дверь>, <шаги>}`. Копия, вытеснение и ранний отсев — как в вершине бора;
      совпадение снимается, когда его задержка истекла.
    - Если на одном событии завершились несколько совпадений одного сценария, выдаётся не
      больше одного: последний токен у них общий. Выдаётся самое уверенное, при равной
      уверенности — начавшееся позже, а при равном начале — с более поздними шагами.
    - Срабатывания одного события выдаются от начавшегося раньше; при равном начале — в порядке
      объявления сценариев.
    - Срабатывание не выдаётся, если у него есть общий токен с выданным срабатыванием того же
      сценария (токены сравниваются по позиции в потоке) или если с выданного срабатывания
      этого сценария прошло меньше его `cooldown`. Невыданное срабатывание не занимает токены
      и не продлевает cooldown, а совпадение живёт дальше.
    - Время событий не убывает: событие раньше предыдущего — ошибка.
    - `add_scenario` дописывает пути в общий бор. Активные совпадения могут продолжаться по
      добавленным веткам.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._defaults = rules  # стандартные задержка и пороги не меняются
        self._vocabulary = rules.vocabulary
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        # сценарий → номер объявления: срабатывания с одним началом выдаются в этом порядке
        self._declared = {scenario.name: index for index, scenario in enumerate(rules.scenarios)}
        self._min_confidences = {
            scenario.name: rules.confidence_threshold(scenario) for scenario in rules.scenarios
        }
        self._starts: dict[Start, Node] = {}
        # токен → рёбра с ним: событие проверяет только их, а не все активные совпадения
        self._start_index: dict[str, list[tuple[Start, Node]]] = {}
        self._edge_index: dict[str, list[tuple[Node, Edge, Node]]] = {}
        for scenario in rules.scenarios:
            self._index_edges(insert_scenario(self._starts, rules, scenario))
        self._unordered = build_unordered(rules)
        self._unordered_owners = {
            (tuple(sorted(pattern.slots)), pattern.max_gap): pattern.scenario
            for pattern in self._unordered
        }
        self._collectors: list[Collector] = []
        self._collector_index: dict[str, list[Collector]] = {}  # токен → шаблоны `any` с ним
        self._index_unordered(self._unordered)
        self._nodes: dict[Node, Front] = {}  # вершина → несравнимые совпадения в ней, без пустых
        # (срок головы фронта, порядковый номер, фронт): срок может оказаться раньше настоящего,
        # тогда фронт просто вернётся в очередь, но позже настоящего — никогда
        self._deadlines: list[tuple[float, int, Front]] = []
        self._order = count()  # разводит равные сроки, чтобы куча не сравнивала фронты
        self._active_count = 0
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self._taken_count = 0  # сколько позиций во всех `_taken`
        self._cleaned = 0  # позиция, на которой `_taken` чистили в последний раз
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        self._position = 0
        self._last_time: float | None = None  # None, пока не пришло ни одного события
        self._misses: list[Miss] = []  # отказы последнего события

    def _index_edges(self, created: list[tuple[Node | None, Start | Edge, Node]]) -> None:
        for parent, edge, child in created:
            if parent is None:
                self._start_index.setdefault(edge.token, []).append((edge, child))
            else:
                self._edge_index.setdefault(edge.token, []).append((parent, edge, child))

    def _index_unordered(self, patterns: list[Unordered]) -> None:
        for pattern in patterns:
            collector = Collector(pattern)
            self._collectors.append(collector)
            for token in collector.groups:
                self._collector_index.setdefault(token, []).append(collector)

    def add_scenario(self, scenario: Scenario | dict) -> None:
        """Добавляет сценарий без перестроения графа и сброса текущих совпадений.

        Принимает `Scenario` или словарь в формате элемента `scenarios` из JSON. Проверяются
        только новые шаблоны; их токены добавляются в словарь после успешной вставки.
        Сохранившиеся префиксы доступны новому сценарию. События не переигрываются;
        снятые совпадения не восстанавливаются.
        Вызывать между `push`/`feed`; одновременные вызовы из разных потоков не поддерживаются.
        """
        if isinstance(scenario, dict):
            scenario = parse_scenario(scenario)
        if scenario.name in self._cooldowns:
            raise ValueError(f"сценарий {scenario.name!r} объявлен дважды")
        new_tokens = {
            step.token for pattern in scenario.patterns for step in pattern
        } - self._vocabulary
        vocabulary = self._vocabulary | new_tokens if new_tokens else self._vocabulary
        rules = RuleSet(
            vocabulary,
            (scenario,),
            self._defaults.max_gap,
            self._defaults.min_score,
            self._defaults.min_confidence,
        )
        if scenario.order == "sequence":
            # новое ребро может удлинить задержку вершины; её срок в очереди тогда раньше
            # настоящего, и фронт просто вернётся в очередь
            self._index_edges(insert_scenario(self._starts, rules, scenario))
        else:
            known = len(self._unordered)
            insert_unordered(self._unordered, self._unordered_owners, rules)
            self._index_unordered(self._unordered[known:])
        self._cooldowns[scenario.name] = scenario.cooldown
        self._declared[scenario.name] = len(self._declared)
        self._min_confidences[scenario.name] = rules.confidence_threshold(scenario)
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> frozenset[str]:
        """Текущий словарь, включая токены успешно добавленных сценариев."""
        return self._vocabulary

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def starts(self) -> dict[Start, Node]:
        """Рёбра из корня бора, по которому идёт движок."""
        return self._starts

    @property
    def unordered(self) -> list[Unordered]:
        return self._unordered

    @property
    def position(self) -> int:
        """Номер, который получит следующее событие."""
        return self._position

    @property
    def misses(self) -> list[Miss]:
        """Почему последнее событие что-то не продвинуло или не дало сработать сценарию."""
        return self._misses

    def snapshot(self) -> list[Waiting]:
        """Все частичные совпадения: в вершинах бора и в наборах `any`."""
        # в `_nodes` стоят только вершины с рёбрами: в остальных ждать нечего
        waiting = [
            Waiting(node, partial, partial.steps[-1].event.time + node.max_gap)
            for node, front in self._nodes.items()
            for partial in front
        ]
        waiting += [
            Waiting(
                collector.place(filled),
                partial,
                partial.steps[-1].event.time + collector.pattern.max_gap,
            )
            for collector in self._collectors
            for filled, front in collector.states.items()
            for partial in front
        ]
        return waiting

    def feed(self, events: Iterable[Event]) -> list[Match]:
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def push(self, event: Event) -> list[Match]:
        if self._last_time is not None and event.time < self._last_time:
            raise ValueError(
                f"время {event.time} токена {event.token!r} раньше предыдущего {self._last_time}"
            )
        self._last_time = event.time
        step = Step(self._position, event)
        self._position += 1

        matches: list[Match] = []
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием
        misses: list[Miss] = []
        # куда входят совпадения этого события; добавляются после обхода, чтобы событие не
        # продолжило совпадение, которое само же и продлило
        entries: list[tuple[Node | tuple[Collector, int], Partial]] = []

        def promising(node: Node, product: float) -> bool:
            """Ранний отсев: дотянет ли совпадение до порога хоть одного шаблона ниже вершины."""
            # вдали от `floor` хватает сравнения с ним; рядом — та же формула, что в отчёте об
            # отсеве, чтобы исход на самой границе не решало округление
            if product > node.floor * (1 + SLACK):
                return True
            if product < node.floor * (1 - SLACK):
                return False
            return any(product ** (1 / length) >= threshold for length, threshold in node.ahead)

        def prune(node: Node, partial: Partial) -> None:
            # ближе всего к порогу — шаблон с наибольшим запасом
            length, threshold = max(node.ahead, key=lambda item: partial.bound(item[0]) - item[1])
            misses.append(Miss("pruned", node, partial, None, threshold, partial.bound(length)))

        def collect(collector: Collector, filled: int, partial: Partial) -> None:
            pattern = collector.pattern
            if filled == collector.full:
                completed.setdefault(pattern.scenario, []).append(partial)
            elif partial.bound(len(pattern.slots)) >= pattern.min_confidence:  # ранний отсев
                entries.append(((collector, filled), partial))
            else:
                bound = partial.bound(len(pattern.slots))
                place = collector.place(filled)
                misses.append(Miss("pruned", place, partial, None, pattern.min_confidence, bound))

        def low_slots(collector: Collector, filled: int) -> list[int]:
            """Открытые шаги набора с токеном события, порог которых событие не прошло."""
            return [
                index
                for index in collector.free(filled, event.token, 1.0)
                if event.score < collector.pattern.slots[index].min_score
            ]

        def fire(scenario: str, partials: list[Partial]) -> None:
            taken = self._taken.setdefault(scenario, set())
            eligible = [
                partial
                for partial in partials
                if partial.confidence >= self._min_confidences[scenario]
                and taken.isdisjoint(position for position, _ in partial.steps)
            ]
            if not eligible:
                # уверенность ниже порога или повтор: токен уже вошёл в выданное срабатывание
                threshold = self._min_confidences[scenario]
                for partial in partials:
                    if partial.confidence < threshold:
                        confidence = partial.confidence
                        misses.append(
                            Miss("confidence", None, partial, scenario, threshold, confidence)
                        )
                    else:
                        misses.append(Miss("repeat", None, partial, scenario, 0.0, 0.0))
                return
            # при равной уверенности — начавшееся позже, при равном начале — с более поздними шагами
            best = max(eligible, key=lambda partial: (partial.confidence, partial.positions()))
            fired_at = self._fired_at.get(scenario)
            if fired_at is not None and event.time - fired_at < self._cooldowns[scenario]:
                until = fired_at + self._cooldowns[scenario]
                misses.append(Miss("cooldown", None, best, scenario, until, event.time))
                return
            taken.update(position for position, _ in best.steps)
            self._taken_count += len(best.steps)  # позиции новые: проверено выше
            self._fired_at[scenario] = event.time
            matches.append(Match(scenario, best.steps, best.confidence))

        self._expire(event, misses)
        for parent, edge, child in self._edge_index.get(event.token, ()):
            front = self._nodes.get(parent)
            if front is None:
                continue
            items = front.items
            # срок считается от последнего шага совпадения, а не от последнего токена потока;
            # фронт идёт от старых к свежим, поэтому в задержку ребра укладывается его хвост
            fresh = front.since(
                lambda partial: event.time - partial.steps[-1].event.time <= edge.max_gap
            )
            for partial in islice(items, front.head, fresh):
                last = partial.steps[-1].event.time
                misses.append(Miss("late", child, partial, None, last + edge.max_gap, event.time))
            if fresh == len(items):
                continue
            if event.score < edge.min_score:
                for partial in islice(items, fresh, None):
                    misses.append(Miss("score", child, partial, None, edge.min_score, event.score))
                continue
            if child.scenario is not None:
                completed.setdefault(child.scenario, []).extend(
                    items[index].extend(step) for index in range(fresh, len(items))
                )
            # все продолжения входят в одну вершину с одним последним шагом, поэтому остаётся
            # одно — с наибольшим произведением, то есть первое в хвосте
            best = items[fresh].extend(step)
            if promising(child, best.product):
                entries.append((child, best))
            if child.ahead:
                # отсеянные продолжения — с конца хвоста, где произведения наименьшие
                for index in range(len(items) - 1, fresh - 1, -1):
                    if promising(child, items[index].product * event.score):
                        break
                    prune(child, items[index].extend(step))
        for start, child in self._start_index.get(event.token, ()):
            if event.score < start.min_score:
                misses.append(Miss("score", child, None, None, start.min_score, event.score))
                continue
            partial = Partial((step,), event.score)  # новое совпадение задержку не проверяет
            if child.scenario is not None:
                completed.setdefault(child.scenario, []).append(partial)
            # у вершины без продолжений `ahead` пуст, и совпадение в ней не остаётся
            if promising(child, partial.product):
                entries.append((child, partial))
            elif child.ahead:
                prune(child, partial)
        for collector in self._collector_index.get(event.token, ()):
            pattern = collector.pattern
            length = len(pattern.slots)
            for filled, front in collector.states.items():
                # снятые по сроку уже ушли, а задержка у шагов набора одна: подходят все
                for index in collector.free(filled, event.token, event.score):
                    target = filled | 1 << index
                    if target == collector.full:
                        completed.setdefault(pattern.scenario, []).extend(
                            partial.extend(step) for partial in front
                        )
                        continue
                    # как в вершине бора: остаётся самое уверенное, отсеянные — с конца
                    best = front.first().extend(step)
                    if best.bound(length) >= pattern.min_confidence:
                        entries.append(((collector, target), best))
                    for back in range(len(front.items) - 1, front.head - 1, -1):
                        product = front.items[back].product * event.score
                        if product ** (1 / length) >= pattern.min_confidence:
                            break
                        partial = front.items[back].extend(step)
                        misses.append(
                            Miss(
                                "pruned",
                                collector.place(target),
                                partial,
                                None,
                                pattern.min_confidence,
                                partial.bound(length),
                            )
                        )
                for index in low_slots(collector, filled):
                    threshold = pattern.slots[index].min_score
                    place = collector.place(filled)
                    for partial in front:
                        misses.append(Miss("score", place, partial, None, threshold, event.score))
            # начать набор может любой его шаг
            for index in collector.free(0, event.token, event.score):
                collect(collector, 1 << index, Partial((step,), event.score))
            for index in low_slots(collector, 0):
                threshold = pattern.slots[index].min_score
                misses.append(
                    Miss("score", collector.place(0), None, None, threshold, event.score)
                )
        for place, partial in entries:
            if isinstance(place, Node):
                front = self._nodes.get(place)
                if front is None:
                    front = self._nodes[place] = Front(place)
            else:
                collector, filled = place
                front = collector.states.get(filled)
                if front is None:
                    front = collector.states[filled] = Front(place)
            queued = len(front) > 0  # у непустого фронта уже есть срок в очереди
            self._active_count += front.add(partial)
            if not queued:
                self._schedule(front)
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        for scenario, partials in completed.items():
            fire(scenario, partials)
        # порядок обхода задаёт индекс, а не поток, поэтому выдача упорядочена явно: сначала
        # начавшееся раньше
        matches.sort(key=lambda match: (match.steps[0].position, self._declared[match.scenario]))

        self._misses = misses
        # поиск самого старого совпадения обходит все, поэтому чистка идёт, только когда с
        # прошлой прошло не меньше событий, чем совпадений и запомненных позиций: в среднем O(1)
        if self._position - self._cleaned >= self._active_count + self._taken_count:
            self._forget_taken()
        return matches

    def _schedule(self, front: Front) -> None:
        """Ставит непустой фронт в очередь по сроку его самого старого совпадения."""
        place = front.place
        gap = place.max_gap if isinstance(place, Node) else place[0].pattern.max_gap
        deadline = front.first().steps[-1].event.time + gap
        heapq.heappush(self._deadlines, (deadline, next(self._order), front))

    def _expire(self, event: Event, misses: list[Miss]) -> None:
        """Снимает совпадения, срок которых к событию истёк: во фронте это голова."""
        now = event.time
        due: list[Front] = []
        while self._deadlines and self._deadlines[0][0] <= now + SLACK * (1 + abs(now)):
            due.append(heapq.heappop(self._deadlines)[2])
        for front in due:
            place = front.place
            while len(front):
                last = front.first().steps[-1].event.time
                if isinstance(place, Node):
                    if now - last <= place.max_gap:
                        break
                    partial = front.pop_first()
                    # есть ребро с токеном события: снятие объясняет опоздание на нём
                    late = [
                        Miss("late", child, partial, None, last + edge.max_gap, now)
                        for edge, child in place.children.items()
                        if edge.token == event.token
                    ]
                    deadline = last + place.max_gap
                    misses.extend(late or [Miss("expired", place, partial, None, deadline, now)])
                else:
                    collector, filled = place
                    deadline = last + collector.pattern.max_gap
                    if now <= deadline:
                        break
                    partial = front.pop_first()
                    # задержка у открытых шагов одна, и она истекла
                    reason = "late" if collector.free(filled, event.token, 1.0) else "expired"
                    misses.append(
                        Miss(reason, collector.place(filled), partial, None, deadline, now)
                    )
                self._active_count -= 1
            if len(front):
                self._schedule(front)
            elif isinstance(place, Node):
                del self._nodes[place]
            else:
                del place[0].states[place[1]]

    def _forget_taken(self) -> None:
        """Забывает позиции выданных срабатываний, которые уже не встретятся в новых.

        Новые срабатывания состоят из позиций не раньше начала самого старого активного
        совпадения.
        """
        fronts = [
            *self._nodes.values(),
            *(front for collector in self._collectors for front in collector.states.values()),
        ]
        oldest = min(
            (partial.steps[0].position for front in fronts for partial in front),
            default=self._position,
        )
        self._taken_count = 0
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
            self._taken_count += len(taken)
        self._cleaned = self._position
