# D&D-flavored Mobile Card Game — Concept Exploration

Status: exploration / pre-greenlight. Nothing here is committed design.

---

## 1. Where the market actually is (2026)

The reference points named in the brief are four *different* businesses, and
conflating them is the most common way this kind of project dies:

| Game | Core loop | What it monetizes | Session length |
|---|---|---|---|
| Hearthstone | Curve-out board combat, no opponent-turn interaction | Packs, expansions | 8–12 min |
| MTG Arena | Stack-based instant-speed interaction, land resource | Packs, drafts | 15–25 min |
| Pokémon TCG Pocket | **Pack opening is the loop**; battles are secondary | Pack currency, collection | 3–5 min |
| 我叫MT / 剑与远征-likes | Auto-battle, party assembly, idle progression | Gacha heroes, power ladder | 30 sec bursts |

Two of these are *skill games with a collection attached*. Two are
*collection games with a battle attached*. Pick one. A game that tries to be
both ends up with a competitive ladder poisoned by pay-to-win and a collection
loop too slow to satisfy collectors.

**Recommendation: skill game first, collection second** — but with the
Pokémon Pocket lesson applied, which is that *opening things* must feel good
daily even when you don't want to play a match.

The hard truth on competition: pure PvP constructed TCG is one of the most
brutal categories on mobile. Legends of Runeterra had a top-tier team, a huge
IP, and generous economy, and still got gutted. Marvel Snap succeeded by
changing the *shape* of a match (6 turns, simultaneous reveal, 12-card decks),
not by making a better Hearthstone. Any pitch here needs a structural change,
not a reskin.

---

## 2. What D&D actually gives us that the competition can't copy

Most "D&D card game" pitches use D&D only as art direction — dragons, taverns,
a rogue with a dagger. That's worth nothing; Hearthstone already owns
high-fantasy card art in the public mind. The defensible material is the
*mechanical* vocabulary that only D&D has:

1. **The d20.** No competitor has a shared, visible, manipulable randomizer.
2. **The party.** Four persistent characters who level *during* the adventure —
   not disposable minions.
3. **Ability scores.** STR/DEX/CON/INT/WIS/CHA as a six-axis identity system.
4. **Skill checks and saving throws.** Degrees of success, not binary resolution.
5. **The dungeon.** Rooms, doors, and forward progress as a board topology.
6. **The DM.** A genuinely asymmetric second role — nobody else can ship this.
7. **Rests.** Short/long rest as a deliberate tempo-vs-resource decision.

Items 1, 2 and 6 are the ones I'd build a company on.

---

## 3. The central design problem: randomness

D&D is a game about dice. Competitive card games die when players feel the
dice beat them. Hearthstone's most hated cards are its RNG cards; MTG players
tolerate variance because it happens at *deck construction and draw*, not at
*resolution*.

**The fix — and I think this is the game's actual invention:**

> Randomness resolves **before** the decision, never after it.

### The Fate Die

A single d20 sits in the middle of the board, face up, showing a number.
Both players can see it. It rerolls at the top of each round — **and the roll
happens before either player commits cards.**

Cards then read against that known value:

- `Cleave — Requires the Fate Die at 11+. Deal 4 damage.`
- `Lucky Halfling — While the Fate Die is 5 or less, your heroes have +2 ATK.`
- `Guidance — Nudge the Fate Die up or down by 2.`
- `Portent — Look at next round's die and set it to any value. (Once per game.)`
- `Reckless Attack — Reroll the Fate Die. You must keep the new result.`

The consequences of this are large:

- **A "nat 20" becomes something you engineer**, not something you're handed.
  The emotional payload of the crit survives; the injustice does not.
- **It creates a public information clock.** Both players plan around the same
  number, which produces bluffing and reads — the poker-flop dynamic, not the
  slot-machine dynamic.
- **Die manipulation is a whole card axis** and therefore a whole archetype
  family (Divination decks that control the die; Barbarian decks that want it
  swingy; Halfling/Lucky decks that profit at the low end).
- **It's legible in one glance on a phone screen.** A single big number.

Losing to a die you could see, on a turn you chose to commit into it, reads as
your mistake. That is the entire trick.

---

## 4. Three candidate structures

### Concept A — **"Delve"** (lead candidate)

*3-minute simultaneous-reveal PvP over a 3-room dungeon corridor.*

- Board is three rooms, revealed one per round (rounds 1, 2, 3), each with a
  D&D chamber effect: *Shrine* (heroes here heal 2), *Collapsing Bridge*
  (-2 HP to all here), *Treasure Vault* (winner of this room draws 2),
  *Antimagic Field* (spells can't be played here).
- Both players commit cards face-down each round, then reveal simultaneously.
  This halves match length, removes turn timers as a UX problem, and makes
  async/reconnect trivial.
- **Heroes persist and level.** A hero that survives a round gains +1/+1 and
  unlocks its level-2 ability. Your board is a party that grows, not a stream
  of disposable minions. This is the emotional difference from Snap.
- Win 2 of 3 rooms after 6 rounds.
- The Fate Die drives everything above.

**Why this one:** shortest path to a shippable, differentiated, phone-native
match; the persistence layer gives the D&D fantasy that Snap structurally
can't; simultaneous reveal is a proven mobile fit.

### Concept B — **"Ability Score economy"** (a system, not a game)

Cards cost from six pools instead of one mana bar. Deckbuilding literally
becomes character building.

Real depth, real differentiation, and real risk: six resource types is a
tutorial nightmare on a 6-inch screen and roughly triples the balancing
surface. **Recommended compromise:** keep costs single-currency (a simple
1-per-turn mana bar, zero friction), but give cards *requirements* met by your
current party — `Requires STR 13+`. Your board state gates your hand.

Simple costs, deep synergy, one number to read. This folds into Concept A
rather than competing with it.

### Concept C — **"Dungeon Master"** (the long-term moat)

Asymmetric: one player builds an encounter — monsters, traps, room layout,
a budget — and one to four players delve it.

This is the thing no competitor can ship, and it's an infinite-content engine
via UGC, which the D&D audience will absolutely produce. It is also an
asymmetric-balance problem that has humbled better studios than most, and it
should not be in the v1 scope. Ship it as the year-2 pillar once the core
combat math is battle-tested.

---

## 5. Recommended shape

**Concept A as the core, B folded in as the synergy layer, C on the roadmap.**

Three modes:

- **Delve (PvP ranked)** — 3 minutes. The competitive spine.
- **Campaign (PvE roguelike)** — an 8-encounter run, deck grows between fights,
  boss at the end. Slay-the-Spire DNA, which is proven premium-friendly on
  mobile, and it's where you sell content rather than power.
- **Dungeon Master (year 2)** — build-and-share encounters.

### Design pillars

1. **You are the party, not a wizard playing minions.** Characters persist,
   level, and are remembered between rooms.
2. **The die is a resource, not a verdict.** Every roll is public and precedes
   the decision it affects.
3. **A delve fits in a bus stop.** Three minutes, one hand, no turn timer anxiety.

---

## 6. The legal question — this materially shapes the product

"D&D" is Wizards of the Coast's trademark, and this is the first thing to get
right for anything commercial.

- **SRD 5.1 and SRD 5.2 are released under Creative Commons Attribution 4.0.**
  Classes, ability scores, the d20 mechanic, saving throws, spell names,
  most monsters, and the core rules vocabulary are usable commercially with
  attribution, with no license grant needed from WotC.
- **Excluded (Product Identity, do not touch):** the *Dungeons & Dragons* and
  *D&D* names and logos, Forgotten Realms and other settings, and the signature
  monsters — beholder, mind flayer/illithid, displacer beast, umber hulk,
  carrion crawler, yuan-ti, slaad, githyanki/githzerai, and the named fiend
  races.

So: the product can be **mechanically and vocabularily D&D**, and must be
**brand-independently its own IP**. Practically that means an original setting
and original marquee monsters — which is a good thing commercially anyway,
because you own the IP you're building value in instead of renting it.

Budget a real IP lawyer review before any art or naming is locked. Working
titles that avoid the trademark: *Initiative*, *Nat 20*, *Delve*, *Torchbearer*.

---

## 7. Business model

The D&D-adjacent audience skews older, higher-income, and is unusually
hostile to predatory monetization — this cohort talks publicly and loudly
about it. Loot-box-forward design is also under active regulatory pressure in
the EU and Brazil.

Recommended:

- **Competitive card pool is obtainable through play.** Do not sell ladder power.
- **Monetize the collector and the content consumer:** cosmetics (card backs,
  dice skins, hero art variants, board/room art, crit animations), a battle
  pass, and **paid campaign chapters** — premium PvE content is a clean sale
  to exactly this audience.
- **A wildcard/crafting system** so a player can target the deck they want.
  Runeterra was too generous and starved; Snap's acquisition was too random
  and enraged. The middle is targeted crafting with a real but reachable cost.
- **Dice skins are an unusually good cosmetic slot** — the Fate Die is the
  single most-looked-at object on screen, in every match, for both players.

Rough sanity numbers for a title of this class: 18–30 months to soft launch,
25–45 people, $3–8M. A 6-person team can get to a genuine vertical slice in
3–4 months. Anyone promising commercial quality materially under that is
promising a prototype.

---

## 8. Unity architecture — the decisions that are expensive to reverse

1. **The rules engine is a pure C# assembly with zero `UnityEngine`
   references.** Deterministic, no floats in game logic, no
   `Random` outside a seeded PRNG, no dictionary-iteration-order dependence.
   This one decision buys: server-authoritative play (anti-cheat), headless
   balance simulation at millions of matches, unit-testable rules,
   client-side prediction, and replays. Every card game that skipped this
   regretted it.
2. **Replays = seed + input log.** Free once the sim is deterministic. Powers
   spectating, esports, bug reproduction, and cheat detection.
3. **Cards are data, not code.** ScriptableObject authoring in-Editor,
   compiled to a versioned data blob the server also loads. An effect DSL
   (composable triggers/conditions/effects) so 95% of new cards ship without a
   client build. Card balance changes must not require App Store review.
4. **Addressables + remote content** for art, with content versioning tied to
   the ruleset version.
5. **Backend:** run the *same sim DLL* server-side on .NET. This is the strongest
   argument for a custom .NET service over Nakama/PlayFab-as-logic-host — one
   implementation of the rules, never two that drift.
6. **Perf targets:** 60fps on an iPhone SE2 / Snapdragon 6-tier, <150MB initial
   download, <4s cold start. Card games lose players to load times, not framerate.
7. Unity 6 LTS, URP, 2D-first. Spine or similar for card art motion.

---

## 9. Open questions I'd want answered before greenlight

1. **PvP-first or PvE-first?** PvE is far cheaper to validate and monetize but
   has a lower ceiling; PvP needs a live population from day one.
2. **Is there an existing team/budget, or is this a solo/small-team effort?**
   The answer changes the recommendation from Concept A to a much tighter
   single-player roguelike.
3. **Own IP or license?** Recommendation is own IP on SRD mechanics.
4. **Which market first?** 我叫MT in the reference set implies China/CN-market
   interest, which brings a completely different monetization norm (gacha is
   expected, not resented) and a publishing-license requirement (版号) that adds
   a year of lead time. This is a fork in the road, not a detail.

---

## 10. Next step I'd propose

Before any Unity work: **build the deterministic rules engine and a headless
match simulator**, and prove the Fate Die math is sound. That's pure C#,
it needs no Editor, and it's the artifact the Unity project would be built
*around* rather than bolted onto.

Concretely, a balance harness that:

- implements the Concept A ruleset as a deterministic sim,
- runs scripted AI policies against each other over millions of matches,
- reports win-rate by archetype, average game length, first-player advantage,
  and card-level win-rate deltas,
- flags cards outside a target band.

First-player advantage in a simultaneous-reveal game and the variance profile
of the Fate Die are both answerable this way *in an afternoon*, and both are
capable of killing the design. Better to find out now.
