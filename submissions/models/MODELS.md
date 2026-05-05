# Model Descriptions

---

### ``core.py``

#### Description

Copyable legality-first baseline.

This file is the default starting point for new models. It starts from the
initial benchmark placement, legalizes hard macros with small moves, keeps soft
macros unchanged, and leaves a clear refinement hook for future work.

---

### ``simple.py``

#### Description

Just an example for the moment but we'd put smth lke "testing the rl with ..."

#### Results

```
Costs computed:
    - Wirelength:  0.128768
    - Density:     1.276113
    - Congestion:  2.248285
    - Proxy Cost:  1.890967
```

---

## Summary Table

| Model | Main Idea | Status | ibm01 | All IBM | Runtime Notes | Keep Going? |
|------|------|------|------|------|------|------|
| `core.py` | Minimal-displacement legalization baseline | Ready | Pending | Pending | Designed for debugging and extension | Yes |
| `simple.py` | Example placeholder | Placeholder | 1.890967 | Pending | Replace with real runtime notes later | Maybe |
