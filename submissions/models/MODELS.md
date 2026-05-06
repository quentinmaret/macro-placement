# Model Descriptions

---

### ``core.py``

#### Description

Copyable legality-first baseline.

This file is the default starting point for new models. It starts from the
initial benchmark placement, legalizes hard macros with small moves, keeps soft
macros unchanged, and leaves a clear refinement hook for future work.

---

### ``rl_local_policy.py``

#### Description

RL-style local policy template.

This file keeps the same legality-first workflow as `core.py`, but changes the
refinement stage into a sequential policy over local macro moves. It is meant
to show what an RL-flavored model looks like in this repo before we build a
full training stack.

#### Notes

- Starts from the initial placement and legalizes first.
- Builds a hard-macro graph from benchmark nets.
- Visits movable hard macros one by one.
- Generates local legal actions for each macro.
- Scores those actions with an interpretable policy over simple features.
- Keeps the structure easy to replace with a learned policy later.

---

### ``initializer.py``

#### Description

Ensemble initializer template.

This file focuses on generating strong starting placements rather than running
an extended local search policy. It keeps the same copyable model structure as
the other templates, but organizes the logic as an initializer ensemble.

#### Notes

- Includes the raw benchmark placement as a safe baseline candidate.
- Adds a graph clustering / hierarchy initializer for movable hard macros.
- Legalizes every candidate with the shared `core.py` scaffold.
- Scores candidates with a lightweight connectivity-aware surrogate.
- Is intentionally structured so more initializer methods can be added later.

#### How To Use

Run it on one benchmark:

```bash
uv run evaluate submissions/models/initializer.py -b ibm01
```

Run it on all IBM benchmarks:

```bash
uv run evaluate submissions/models/initializer.py --all
```

What it does at runtime:

1. Builds an ensemble of initialization candidates.
2. Includes the original benchmark placement as a fallback candidate.
3. Builds a graph hierarchy candidate from hard-macro connectivity.
4. Legalizes every candidate with the shared `core.py` logic.
5. Scores the legal candidates and returns the best one.

#### How To Extend

To add another initializer method:

1. Open `submissions/models/initializer.py`.
2. Add a new name to `self.initializer_sequence`.
3. Implement a matching `run_<name>_initializer(self, benchmark)` method.
4. Return a full placement tensor from that method.
5. Let the shared legalization and candidate scoring pipeline handle the rest.

Current ensemble members:

- `benchmark_anchor`
- `graph_hierarchy`

---

## Summary Table

| Model | Main Idea | Status | ibm01 | All IBM | Runtime Notes | Keep Going? |
|------|------|------|------|------|------|------|
| `core.py` | Minimal-displacement legalization baseline | Ready | Pending | Pending | Designed for debugging and extension | Yes |
| `rl_local_policy.py` | Sequential RL-style local action policy over legal moves | Ready | Pending | Pending | Educational RL bridge before full training | Yes |
| `initializer.py` | Ensemble initializer with graph clustering / hierarchy seeding | Ready | Valid on `ibm01` | Pending | Good place to add more seeding methods | Yes |
