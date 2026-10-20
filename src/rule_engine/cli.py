"""Командная строка: rule-engine <команда> …

  query     понять команду без исполнения (объяснение, гипотезы, решение)
  run       выполнить одну или несколько команд подряд в симуляторе
  repl      интерактивный режим (диалог, уточнения, :state, :inject, :alias …)
  rules     скомпилировать спецификацию и показать правила/предупреждения
  add-alias добавить синоним (с проверкой регрессии на размеченных запросах)
  rollback-alias  откатить синонимы к версии
  eval      оценка на размеченных запросах (+ опечатки, абляции, отчёт)
  bench     масштабирование: задержка и размер индекса на 100/1k/10k устройств
  mine      выучить сценарии и правила из журнала команд
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rule_engine.engine import Engine
from rule_engine.evaluation import (
    DEFAULT_DATASET,
    antonym_stress,
    evaluate,
    format_table,
    load_dataset,
    markdown_report,
    regression_diff,
    run_ablations,
    scaling_benchmark,
    summary_line,
    typo_variants,
)
from rule_engine.mining import RuleMiner, load_events, sessionize
from rule_engine.rules import RuleKind
from rule_engine.spec import scenario_to_json


def _engine(args: argparse.Namespace) -> Engine:
    return Engine.from_files(
        args.spec, args.overlay, backend=args.backend, stemmer=args.stemmer, seed=args.seed
    )


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


# ------------------------------------------------------------------ commands
def cmd_query(args: argparse.Namespace) -> int:
    engine = _engine(args)
    text = " ".join(args.text)
    result = engine.understand(text, args.room)
    decision = result.decision
    if args.json:
        _print_json(
            {
                "query": text,
                "normalized": result.query.text,
                "trace": result.query.trace,
                "decision": decision.status.value,
                "fired": decision.fired,
                "errors": [e.to_dict() for e in decision.errors],
                "hypotheses": [
                    {"rank": i + 1, "score": h.score, "command": engine.render_interpretation(h),
                     "risk": h.risk, "explain": h.explain}
                    for i, h in enumerate(result.interpretations[: args.top])
                ],
                "timings_ms": result.timings,
            }
        )
        return 0
    print(f"запрос:        {text}")
    print(f"нормализация:  {result.query.text}")
    for line in result.query.trace:
        print(f"               {line}")
    print(f"решение:       {decision.status.value}  (правило {', '.join(decision.fired)})")
    for error in decision.errors:
        print(f"               {error.code.value}: {error.message()}")
    print("гипотезы:")
    for i, h in enumerate(result.interpretations[: args.top]):
        print(f"  {i + 1}. [{h.score:.3f}] {engine.render_interpretation(h)}  (риск {h.risk})")
        if args.explain or i == 0:
            for line in h.explain:
                print(f"       {line}")
    print("время, мс:     " + ", ".join(f"{k}={v:.3f}" for k, v in result.timings.items()))
    return 0


def _apply_injections(engine: Engine, injections: list[str]) -> None:
    """``device:kind[:times]``; kind может сам содержать двоеточие (``drift:on``)."""
    for spec in injections:
        device, *rest = spec.split(":")
        times = 1
        if len(rest) > 1 and rest[-1].isdigit():
            times = int(rest.pop())
        engine.driver.inject(device, ":".join(rest), times)


def cmd_run(args: argparse.Namespace) -> int:
    engine = _engine(args)
    engine.session.context_room = args.room
    _apply_injections(engine, args.inject or [])
    commands = list(args.commands)
    if args.file:
        commands += [
            line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    for text in commands:
        print(f"> {text}")
        response = engine.handle(text)
        if args.json:
            _print_json(response.to_dict())
        else:
            print(response.render(verbose=args.verbose))
        print()
    if args.state:
        print("\n".join(engine.describe_world()))
    return 0


REPL_HELP = """команды:
  <текст>                     команда устройству (или ответ на уточнение)
  :room <id|->                комната-контекст (где стоит колонка)
  :state                      состояние устройств и сцены
  :why <текст>                объяснение без исполнения
  :inject <dev> <kind> [n]    отказ: timeout | late | error | fault | offline | online | drift:<state>
  :alias <ref> <синоним>      добавить синоним (ref вида device:living_tv)
  :rollback <версия>          откатить синонимы
  :scene <json>               заменить сцену (список объектов)
  :rules [kind]               правила (lexical, transition, safety, confusable, …)
  :policy                     правила политики
  :mine                       выучить сценарии из команд этой сессии
  :log                        журнал выполненных команд
  :verbose                    подробный вывод вкл/выкл
  :quit"""


def cmd_repl(args: argparse.Namespace) -> int:
    engine = _engine(args)
    engine.session.context_room = args.room
    verbose = args.verbose
    print("rule-engine REPL. :help — список команд.")
    while True:
        try:
            line = input(f"[{engine.session.context_room or '-'}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if not line.startswith(":"):
            print(engine.handle(line).render(verbose=verbose))
            continue
        cmd, _, rest = line[1:].partition(" ")
        try:
            if cmd in ("quit", "q", "exit"):
                return 0
            elif cmd == "help":
                print(REPL_HELP)
            elif cmd == "room":
                engine.session.context_room = None if rest in ("", "-") else rest
            elif cmd == "state":
                print("\n".join(engine.describe_world()))
            elif cmd == "why":
                result = engine.understand(rest, engine.session.context_room)
                print(f"{result.decision.status.value} {result.decision.fired}")
                for h in result.interpretations[:5]:
                    print(f"  [{h.score:.3f}] {engine.render_interpretation(h)}")
                    for e in h.explain:
                        print(f"      {e}")
            elif cmd == "inject":
                parts = rest.split()
                engine.driver.inject(parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 1)
                print(f"инъекция: {parts}")
            elif cmd == "alias":
                ref, _, alias = rest.partition(" ")
                warnings = engine.alias_warnings(ref, alias)
                report = engine.add_alias(ref, alias)
                print(f"версия {report.version}: " + ("уже есть" if report.duplicate else f"добавлен «{alias}» → {ref}"))
                for c in report.conflicts:
                    print(f"  конфликт: «{c.alias}» уже указывает на {c.ref}")
                for w in warnings:
                    print(f"  ⚠ {w}")
            elif cmd == "rollback":
                print(f"удалено записей: {engine.rollback_aliases(int(rest))}")
            elif cmd == "scene":
                engine.update_scene(json.loads(rest))
                print("\n".join(engine.describe_world()[-1:]))
            elif cmd == "rules":
                for rule in engine.ruleset.rules:
                    if not rest or rule.kind.value == rest:
                        print(rule)
            elif cmd == "policy":
                print("\n".join(engine.policy.describe()))
            elif cmd == "mine":
                sessions = sessionize(engine.event_log, gap_ms=10**12)
                result = RuleMiner(engine.spec, min_support=0.0, max_gap=3).mine(sessions)
                _print_json(result.to_json())
            elif cmd == "log":
                for event in engine.event_log:
                    print(event)
            elif cmd == "verbose":
                verbose = not verbose
                print(f"verbose={verbose}")
            else:
                print(f"неизвестная команда :{cmd}; :help")
        except Exception as exc:
            print(f"ошибка: {exc}")


def cmd_rules(args: argparse.Namespace) -> int:
    engine = _engine(args)
    ruleset = engine.ruleset
    if args.json:
        _print_json(ruleset.to_json())
        return 0
    _print_json(ruleset.summary())
    for warning in ruleset.warnings:
        print(f"⚠ {warning}")
    kinds = {RuleKind(k) for k in args.kind} if args.kind else {
        RuleKind.SAFETY, RuleKind.CONFUSABLE, RuleKind.SCENARIO, RuleKind.APPLICABILITY
    }
    for rule in ruleset.rules:
        if rule.kind in kinds:
            print(rule)
    print("\nполитика:")
    for line in engine.policy.describe():
        print(f"  {line}")
    return 0


def cmd_add_alias(args: argparse.Namespace) -> int:
    overlay = args.overlay or str(Path.cwd() / "overlay.json")
    args.overlay = overlay
    engine = _engine(args)
    examples = load_dataset(args.dataset)
    warnings = engine.alias_warnings(args.ref, args.alias)

    def change(e: Engine):
        return e.add_alias(args.ref, args.alias, args.hint)

    report, diff = regression_diff(engine, examples, change)
    if report.duplicate:
        print("синоним уже есть — ничего не изменилось")
        return 0
    print(f"добавлен «{args.alias}» → {args.ref}, версия словаря {report.version}, оверлей: {overlay}")
    for conflict in report.conflicts:
        print(f"⚠ конфликт: «{conflict.alias}» уже указывает на {conflict.ref} — будет переспрашивать")
    for warning in warnings:
        print(f"⚠ опасная близость к антониму: {warning}")
    print(f"регрессия на {len(examples)} размеченных запросах: изменилось {len(diff)}")
    for query, before, after in diff:
        print(f"  «{query}»: {before[0]} → {after[0]}")
    return 0


def cmd_rollback_alias(args: argparse.Namespace) -> int:
    overlay = args.overlay or str(Path.cwd() / "overlay.json")
    args.overlay = overlay
    engine = _engine(args)
    removed = engine.rollback_aliases(args.version)
    print(f"откат к версии {args.version}: удалено записей {removed}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    engine = _engine(args)
    examples = load_dataset(args.dataset)
    if args.typos:
        examples += typo_variants(examples, args.typos, known_word=engine.lexicon.known_word)
    report = evaluate(engine, examples, f"{args.backend}/{args.stemmer}", args.repeats)
    print(summary_line(report))
    by_cat = report.by_category()
    cols = ["n", "p_at_1", "recall_at_3", "decision_acc", "silent_error", "dangerous_silent", "over_clarify"]
    print(format_table([{"category": c, **m} for c, m in by_cat.items()], ["category", *cols]))
    ablations = run_ablations(examples, args.repeats) if args.ablation else None
    if ablations:
        print()
        for r in ablations:
            print(summary_line(r) + f" index={r.index_bytes / 1024:.0f}KB build={r.build_ms:.0f}ms")
    scaling = scaling_benchmark(args.sizes) if args.scaling else None
    stress = antonym_stress() if args.antonym_stress else None
    if stress:
        print()
        print(format_table(stress, ["antonym_margin", "n", "correct_ok", "antonym_ok", "asked", "other"]))
    failures = report.failures()
    if failures:
        print(f"\nошибки ({len(failures)}):")
        for o in failures[:40]:
            print(f"  «{o.example.query}» [{o.example.category}] ожидалось {o.example.expect}, получено {o.status}")
    if args.report:
        Path(args.report).write_text(markdown_report(report, ablations, scaling, stress), encoding="utf-8")
        print(f"\nотчёт: {args.report}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rows = scaling_benchmark(args.sizes, args.backends, args.queries)
    cols = ["devices", "backend", "build_ms", "index_kb", "p_at_1", "recall_at_3", "p50_ms", "p95_ms", "p99_ms"]
    print(format_table(rows, cols))
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    if args.apply and not args.overlay:
        args.overlay = str(Path.cwd() / "overlay.json")
    engine = _engine(args)
    events = load_events(args.logs)
    resolved = []
    unresolved = 0
    for event in events:
        if "text" in event and "action" not in event:
            result = engine.understand(event["text"], event.get("room"))
            top = result.decision.top
            if result.decision.status.value not in ("OK", "CONFIRM") or top is None:
                unresolved += 1
                continue
            for command in top.commands:
                for device in command.targets:
                    resolved.append({**event, "action": command.action, "device": device})
        else:
            resolved.append(event)
    sessions = sessionize(resolved, args.gap_min * 60 * 1000)
    miner = RuleMiner(engine.spec, args.min_support, max_gap=args.max_gap, min_confidence=args.min_confidence)
    result = miner.mine(sessions)
    if unresolved:
        result.notes.append(f"не удалось разобрать текстов: {unresolved}")
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"сохранено: {args.out}")
    if args.apply:
        for scenario in result.scenarios:
            engine.add_scenario(scenario)
        print(f"сценарии добавлены в оверлей {engine.overlay_path}; вызов: «привычка N» (--overlay {engine.overlay_path})")
    for note in result.notes:
        print(f"· {note}")
    print(f"\nсценарии-кандидаты ({len(result.scenarios)}):")
    for scenario in result.scenarios:
        data = scenario_to_json(scenario)
        steps = " → ".join(f"{s['action']}({s['target'].get('device')})" for s in data["steps"])
        print(f"  {scenario.id} [support={scenario.support:.2f}]: {steps}")
    print(f"\nправила следования ({len(result.rules)}):")
    for rule in result.rules[: args.top]:
        print(f"  {rule}  (conf={rule.data['confidence']}, support={rule.data['support']:.2f})")
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rule-engine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=None, help="путь к spec.json (по умолчанию встроенный)")
    parser.add_argument("--overlay", default=None, help="JSON-оверлей с добавленными синонимами")
    parser.add_argument("--backend", default="trie", choices=["trie", "symspell", "ngram", "brute"])
    parser.add_argument("--stemmer", default="light", choices=["light", "pymorphy"])
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("query", help="понять команду без исполнения")
    p.add_argument("text", nargs="+")
    p.add_argument("--room")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("run", help="выполнить команды в симуляторе")
    p.add_argument("commands", nargs="*")
    p.add_argument("--file", help="файл с командами, по одной в строке")
    p.add_argument("--room")
    p.add_argument("--inject", action="append", help="dev:kind[:times], напр. living_tv:timeout:2")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--state", action="store_true", help="в конце показать состояние устройств")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("repl", help="интерактивный режим")
    p.add_argument("--room")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_repl)

    p = sub.add_parser("rules", help="скомпилированные правила")
    p.add_argument("--kind", action="append", choices=[k.value for k in RuleKind])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_rules)

    p = sub.add_parser("add-alias", help="добавить синоним с проверкой регрессии")
    p.add_argument("ref", help="например device:living_tv или action:turn_off")
    p.add_argument("alias")
    p.add_argument("--hint", help="тип устройства-подсказка для синонима действия")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.set_defaults(func=cmd_add_alias)

    p = sub.add_parser("rollback-alias", help="откатить синонимы к версии")
    p.add_argument("version", type=int)
    p.set_defaults(func=cmd_rollback_alias)

    p = sub.add_parser("eval", help="оценка качества")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--typos", type=int, default=0, help="сколько опечаток генерировать на запрос")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--scaling", action="store_true")
    p.add_argument("--antonym-stress", action="store_true", help="кривая порога antonym_margin")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    p.add_argument("--repeats", type=int, default=3, help="повторов для замера задержки (берётся минимум)")
    p.add_argument("--report", help="записать markdown-отчёт")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bench", help="масштабирование на синтетических словарях")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    p.add_argument("--backends", nargs="+", default=["trie", "symspell", "ngram", "brute"])
    p.add_argument("--queries", type=int, default=200)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("mine", help="выучить сценарии из журнала")
    p.add_argument("logs", help="JSONL: {session?, ts, action, device} или {session?, ts, text}")
    p.add_argument("--min-support", type=float, default=0.3)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--max-gap", type=int, default=2)
    p.add_argument("--gap-min", type=float, default=30.0, help="пауза (мин), разрывающая сессию")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out")
    p.add_argument("--apply", action="store_true", help="сохранить сценарии в оверлей (по умолчанию ./overlay.json)")
    p.set_defaults(func=cmd_mine)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
