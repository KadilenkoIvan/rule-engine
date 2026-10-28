"""Движок: проход потока токенов по бору сценариев и по шаблонам `any`.

Событие трогает только то, что его ждёт. Проход одного события (`Engine.push`):

1. снять совпадения с истёкшим сроком: их фронты достаёт очередь сроков (`Deadlines`);
2. продолжить совпадения бора по рёбрам с токеном события и начать новые от корня (`TrieMatcher`);
3. закрыть шаги наборов `any` и начать новые наборы (`SetMatcher`);
4. поставить продлённые совпадения на места — только теперь, чтобы событие не продлило их второй
   раз;
5. выдать срабатывания (`Firing`).
"""

from collections.abc import Iterable, Iterator

from rule_engine.domain.firing import Firing
from rule_engine.domain.front import Deadlines, Front
from rule_engine.domain.graph import Node, Start
from rule_engine.domain.matching import Event, Match, Miss, Partial, Step, Tick, Waiting
from rule_engine.domain.rules import RuleSet, Scenario, parse_scenario
from rule_engine.domain.set_matcher import SetMatcher
from rule_engine.domain.trie_matcher import TrieMatcher
from rule_engine.domain.unordered import Unordered

# снаружи эти типы берут из движка: он их отдаёт
__all__ = ["Engine", "Event", "Match", "Miss", "Partial", "Step", "Waiting"]


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
        self._trie = TrieMatcher(rules)
        self._sets = SetMatcher(rules)
        self._firing = Firing(rules)
        self._deadlines = Deadlines()
        self._active_count = 0
        self._cleaned = 0  # позиция, на которой выданные позиции забывали в последний раз
        self._position = 0
        self._last_time: float | None = None  # None, пока не пришло ни одного события
        self._misses: list[Miss] = []  # отказы последнего события

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
        if scenario.name in self._firing:
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
            self._trie.add(rules, scenario)
        else:
            self._sets.add(rules)
        self._firing.register(rules, scenario)
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
        return self._trie.starts

    @property
    def unordered(self) -> list[Unordered]:
        return self._sets.patterns

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
        return [
            Waiting(front.place, partial, partial.last_time + front.gap)
            for front in self._fronts()
            for partial in front
        ]

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
        tick = Tick(Step(self._position, event))
        self._position += 1

        self._expire(tick)
        self._trie.advance(tick)
        self._trie.start(tick)
        self._sets.advance(tick)
        self._settle(tick)
        matches = self._firing.fire(tick)

        self._misses = tick.misses
        self._forget_taken()
        return matches

    def _expire(self, tick: Tick) -> None:
        """Снимает совпадения, срок которых к событию истёк: во фронте это голова."""
        now = tick.event.time
        for front in self._deadlines.due(now):
            while front and front.expired(front[0], now):
                partial = front.pop_first()
                tick.misses.extend(front.expiry_misses(partial, tick.event))
                self._active_count -= 1
            if front:
                self._deadlines.schedule(front)
            else:
                front.release()

    def _settle(self, tick: Tick) -> None:
        """Ставит продлённые событием совпадения на места; новый фронт встаёт в очередь сроков."""
        for owner, key, partial in tick.entries:
            front = owner.front(key)
            queued = len(front) > 0  # у непустого фронта уже есть срок в очереди
            self._active_count += front.add(partial)
            if not queued:
                self._deadlines.schedule(front)

    def _forget_taken(self) -> None:
        """Забывает позиции выданных срабатываний, которые уже не встретятся в новых.

        Новые срабатывания состоят из позиций не раньше начала самого старого активного
        совпадения. Поиск самого старого обходит все совпадения, поэтому он идёт, только когда с
        прошлого прошло не меньше событий, чем совпадений и запомненных позиций: в среднем O(1).
        """
        if self._position - self._cleaned < self._active_count + self._firing.taken_count:
            return
        oldest = min(
            (partial.steps[0].position for front in self._fronts() for partial in front),
            default=self._position,
        )
        self._firing.forget_before(oldest)
        self._cleaned = self._position

    def _fronts(self) -> Iterator[Front]:
        yield from self._trie.all_fronts()
        yield from self._sets.all_fronts()
