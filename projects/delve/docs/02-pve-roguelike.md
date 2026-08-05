# Concept: PvE Card Roguelike

Follows from `01-concept-exploration.md`, which recommended this as the
lower-risk path. Still exploration — nothing committed.

---

## 1. Why this is the right first game, not the consolation prize

A single-player roguelike is usually pitched as the cheap option. Here it's
also the *better* one, for four reasons specific to this design:

- **No population requirement.** A PvP ladder is dead on arrival without
  concurrent players; a roguelike is fun with one.
- **Variance stops being a liability.** The Fate Die's biggest risk in PvP was
  players feeling cheated. Against a dungeon, a wild swing is a story. The
  mechanic can be *bolder* here, not more timid.
- **You can sell content instead of power.** Chapters, classes, and campaigns
  are clean purchases. No pay-to-win problem to solve.
- **It de-risks the PvP game rather than replacing it.** Same deterministic
  sim, same card DSL, same art pipeline, same ability-requirement system. The
  PvP delve becomes an expansion built on a proven, balanced card pool.

The competition is real but beatable: Slay the Spire, Monster Train, Balatro,
Across the Obelisk, Wildfrost. What none of them have is a *party* and a
*skill check*.

---

## 2. The 30-second hook

> Four adventurers. One die, face-up on the table, that both you and the
> dungeon are fighting over. Every card you draw belongs to one of your
> party — and when one of them dies, their cards die with them.

---

## 3. Run structure

A run is a descent: **3 floors × 6–8 nodes**, boss at the bottom of each.
Target 30–45 minutes, with every individual encounter at 2–3 minutes and
instant suspend/resume — the mobile constraint is that you can put the phone
down mid-floor, not that runs must be short.

Node types: Combat, Elite, **Skill Check**, Tavern (recruit), Shrine
(upgrade), Merchant, Rest, Boss.

---

## 4. The party is the deckbuilding system

This is the structural difference from every roguelike above.

- You start with **one hero** — your class choice, and their ~10-card block.
- You **recruit up to three more** across the run. Each recruit adds their own
  ~6-card block to the shared draw pile *and* stands on the board as a unit
  with its own HP.
- So "should I take this card?" becomes "**should I take this person?**" —
  a chunkier, more legible, more emotional decision than Spire's card picks,
  and one that carries its own art, voice, and personality.
- **Ability requirements carry over from the PvP concept unchanged.**
  `Requires STR 13+` now reads against your *party's* combined scores. Recruit
  a wizard and your INT-gated cards switch on across the whole deck. That's a
  combinatorial synergy web with almost no added UI.

Heroes level during the run: **L2** ability bump, **L3 subclass choice** (the
big branch point, and an authentically beloved D&D moment), **L5** capstone.

---

## 5. The Fate Die in PvE — better than in PvP

One d20, face-up, rerolled at the top of each round, visible before anyone
commits. Now the *monsters key off it too*:

- `Goblin Ambusher — While the Fate Die is 8 or less, attacks twice.`
- `Ancient Warden — While the Fate Die is 15+, immune to spells.`

So the die is contested ground. You're not merely spending it, you're denying
it to the enemy — which converts a randomizer into a **tug-of-war over shared
public state**. Every round asks "who does this number favour, and can I move
it?" That's a genuinely novel combat texture, and it's cheap to render: one
big number in the middle of the screen.

---

## 6. Short rest vs. long rest — the run's central tension

Slay the Spire's campfire is a binary (heal or upgrade). D&D has a better
version already built, and it's one of the most-loved parts of the tabletop
game:

- **Short rest** — spend Hit Dice to heal a little. Free.
- **Long rest** — full HP, spell slots restored, upgrade a card. **Advances the
  Dungeon Alarm by one.**

The Alarm permanently buffs every future encounter on the run. So the question
is never "do I need healing" but "**can I afford to need healing**" — a
compounding, run-shaped decision instead of a per-node one. It also creates
the classic D&D party argument as a single-player choice, which is exactly the
fantasy being sold.

---

## 7. Skill-check encounters — the thing no card roguelike does

A non-combat node presents a situation and a DC. You commit cards from hand;
each card carries skill values; the Fate Die adds. Beat the DC for the good
outcome.

`A collapsed stair blocks the way. DEX 15 to climb, STR 18 to clear it.`

Two payoffs:

1. **Utility cards become genuinely valuable.** In every other deckbuilder,
   non-combat cards are dead weight, so the meta collapses to damage. Here a
   deck needs an answer to problems that aren't monsters — that's a second
   deckbuilding axis the genre doesn't have.
2. **Fail forward.** A failed check shouldn't be a wall; it should be a worse
   door. You take exhaustion and go through anyway, or you find another route.
   That's how the tabletop game handles failure, and it's better roguelike
   design than a flat "you lose the reward."

---

## 8. Death, and why it matters here

Downed heroes make **death saves** — three failures and they're gone for the
run. Their cards leave your deck with them.

This is the emotional stake no deckbuilder has. Losing a card is an
inconvenience. Losing *Wren the half-elf ranger, who has been with you since
floor one, along with the six cards she brought*, is a story you tell someone.
It also creates a real mid-run tactical dilemma: spend the turn on the revive,
or push for the kill.

---

## 9. Meta-progression — the trap to avoid

Unlock **variety, never power**. Persistent stat upgrades that make runs
easier are what turn a skill roguelike into an idle game, and they destroy the
difficulty curve that makes mastery feel like anything.

- Unlock: new classes, new heroes in the recruit pool, new relics, new floors.
- Difficulty: a Spire-style ascension ladder ("Depths 1–20") for the players
  who want it.
- Do **not** ship: permanent +HP, permanent +damage, or anything that makes
  attempt #40 mechanically easier than attempt #4.

---

## 10. Business model

The audience for this — D&D-literate, older, higher-income, roguelike-fluent —
is the same audience that paid full price for Slay the Spire and Balatro on
mobile without complaint, and is unusually hostile to gacha.

Recommended: **premium, or free-first-floor-then-unlock.** Then sell expansion
chapters (new floors, new classes, new card pools) as real content drops.
Cosmetics — especially **dice skins**, since the Fate Die is the most-looked-at
object on screen — are an easy secondary.

This is a fundamentally smaller business than a successful F2P TCG, and a
dramatically more likely one. It also funds the PvP mode from a position of
having a balanced card pool and an existing audience.

---

## 11. Biggest risks

1. **Party + deck + die + alarm + skill checks may be too many systems.** The
   honest test is whether a first-time player understands the board in 10
   seconds. If not, the ability-requirement layer is the first thing to cut.
2. **Skill-check encounters could feel like a minigame bolted on** rather than
   part of the deck's purpose. They only work if combat cards and check cards
   are the *same cards*, used differently.
3. **Content volume.** Roguelikes live or die on run variety, and that's a
   content cost that scales linearly with how long you want players to stay.
4. **The Alarm might just read as a punishment for playing badly** — a
   losing-player-loses-harder spiral. Needs to be testable early.

---

## 12. Next step

Same recommendation as before, now aimed at this design: **a deterministic C#
rules engine plus a headless run simulator.** No Unity required.

What it would answer in an afternoon, with scripted AI policies over thousands
of runs:

- Is the Fate Die's variance profile survivable, or does it decide runs?
- Does the Alarm produce a death spiral?
- Does any single party composition dominate?
- What's the actual win rate curve across floors?

Every one of those is capable of invalidating the design, and all four are
cheaper to answer in code than in Unity — and cheaper in Unity than after
launch.
