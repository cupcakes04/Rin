# Rin docs

```mermaid
flowchart TB
    ST["state.md<br/>where we are · what's next"]
    DIR["direction.md<br/>why · principle · roles"]
    DEC["decisions.md<br/>agreed · proposed · open"]
    IDE["ideas/<br/>holy grail · actions · tracking · …"]
    MS["milestones/<br/>next one in detail"]
    REF["models/ · data.md · lessons/rin_v1.md"]
    BR["brainstorms/<br/>brainstorm.md inbox + processed archive"]
    ST --> MS
    DIR --> DEC --> MS
    IDE --> DEC
    BR --> IDE & DEC
    REF -.-> MS
```

Read in this order: [state.md](state.md) → [direction.md](direction.md) → [decisions.md](decisions.md) → [ideas/holy_grail.md](ideas/holy_grail.md) → [milestones/README.md](milestones/README.md).

Reference: [models/](models/README.md) · [data.md](data.md) · [lessons/rin_v1.md](lessons/rin_v1.md) · [ideas/](ideas/README.md) · [brainstorms/](brainstorms/README.md)
