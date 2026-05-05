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

## Summary Table

| Model | Main Idea | Status | ibm01 | All IBM | Runtime Notes | Keep Going? |
|------|------|------|------|------|------|------|
| `core.py` | Minimal-displacement legalization baseline | Ready | Pending | Pending | Designed for debugging and extension | Yes |
| `rl_local_policy.py` | Sequential RL-style local action policy over legal moves | Ready | Pending | Pending | Educational RL bridge before full training | Yes |
