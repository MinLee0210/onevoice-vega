# docs

Notes, TODOs, decisions, and submission drafts. Nothing here is code —
if it can be derived from source, it doesn't belong here.

```
docs/
├── TODO.md         running task list (edit directly, group by phase)
├── notes/           freeform dated notes: meetings, research findings,
│                    device bring-up logs
├── decisions/       one file per open decision/risk from CLAUDE.md
│                    (e.g. MADLAD-vs-Opus-MT.md, rust-vs-cpp.md) —
│                    record the question, options, and the call once made
├── proposals/       drafts for phase submissions (Phase 2 tech spec,
│                    pitch deck source, business viability writeup)
└── glossary/        design notes for the industrial glossary engine
                     (termbase schema, constrained-decoding approach) —
                     the actual termbase data lives in data/glossary/
```

Naming: `notes/YYYY-MM-DD-topic.md`, `decisions/short-slug.md`.
