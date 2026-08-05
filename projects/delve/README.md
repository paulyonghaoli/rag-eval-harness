# Delve

A D&D-flavored card roguelike for mobile. Pre-production — this folder is
currently design exploration only, no code yet.

*Delve* is a working title. It is deliberately not a D&D trademark; see the
licensing note below.

## The pitch

> Four adventurers. One die, face-up on the table, that both you and the
> dungeon are fighting over. Every card you draw belongs to one of your
> party — and when one of them dies, their cards die with them.

## Current direction

Single-player PvE roguelike first. A competitive PvP mode ("Delve" proper,
3-minute simultaneous-reveal matches) is a later expansion built on the same
engine, card pool, and art — not a separate game.

The design rests on four things the genre doesn't already have:

- **The party is the deckbuilding system.** You recruit heroes, not cards;
  each brings their own card block and stands on the board with their own HP.
- **The Fate Die.** A single public d20, rerolled each round *before* anyone
  commits, that monsters key off as well as you. Randomness resolves before
  the decision it affects, never after — variance becomes an input to skill
  rather than a verdict on it.
- **Short rest vs. long rest.** Long rests heal you and advance a Dungeon
  Alarm that permanently buffs the rest of the run.
- **Skill-check encounters.** Commit cards against a DC, so utility cards
  carry real weight and the meta can't collapse to pure damage.

## Contents

| Path | What it is |
|---|---|
| `docs/01-concept-exploration.md` | Market positioning, three candidate structures, the Fate Die, licensing, Unity architecture |
| `docs/02-pve-roguelike.md` | The chosen direction in depth — run structure, party system, rest economy, skill checks, business model, risks |

## Planned structure

Nothing below exists yet. Recorded here so the layout is a decision rather
than an accident:

| Path | Purpose |
|---|---|
| `sim/` | Deterministic rules engine — pure C#, **zero `UnityEngine` references**. The core of the project. |
| `sim.tests/` | Rules unit tests |
| `balance/` | Headless run simulator and balance reports |
| `cards/` | Card data + effect DSL definitions |
| `unity/` | Unity 6 client (consumes `sim/` as a compiled assembly) |

The zero-Unity-dependency rule on `sim/` is the load-bearing constraint. It
buys server-authoritative play, headless balance simulation at scale,
testable rules, and free replays — and it is very expensive to retrofit.

## Next step

Build `sim/` and `balance/` before any Unity work. Thousands of scripted runs
would answer, cheaply, four questions that can each invalidate the design:

1. Is the Fate Die's variance survivable, or does it decide runs?
2. Does the Dungeon Alarm produce a losing-player-loses-harder spiral?
3. Does any single party composition dominate?
4. What is the win-rate curve across floors?

## Licensing note

D&D's SRD 5.1/5.2 are available under CC-BY-4.0, which covers ability scores,
saving throws, classes, spell names, and most monsters for commercial use with
attribution. The *Dungeons & Dragons* name, its settings, and the signature
monsters (beholder, mind flayer, displacer beast, umber hulk, yuan-ti, and
the named fiend races) are Product Identity and are **not** available.

So this is mechanically D&D-derived and brand-independent, with its own
setting and its own marquee monsters. Get an IP lawyer's review before any
naming or key art is locked.

---

Unrelated to the RAG eval harness at the repository root; it lives here only
because this is where the work was started.
