# Notes

```
.
├── submissions/
│   └── models/             # all different models we build
│       ├── MODELS.md       # all notes of models
│       └── core.py         # copyable legality-first baseline and template
└── NOTES.md                # This
```

# Workflow

- `submissions/models/core.py` is the default starting point for new work.
- When trying a new idea, copy `core.py` to a new file such as `submissions/models/simple.py` or `submissions/models/rl_v1.py`.
- Keep each copied file focused on one hypothesis so we can compare models clearly.
- Write free-form notes for each model in `submissions/models/MODELS.md`.


- Step 1: copy `submissions/models/core.py` to a new model file.
- Step 2: rename the class if you want, but keep a class with a `place(self, benchmark)` method.
- Step 3: first make sure the copied file runs on `ibm01` without errors.
- Step 4: only after the file runs cleanly should we change the refinement logic.
- Step 5: once a model works on `ibm01`, compare it against more IBM benchmarks before trusting it.


# Testing

Run the baseline template on the first IBM benchmark

```bash
uv run evaluate submissions/models/core.py -b ibm01
```

When you copy the template into a new model, run the same command with the new file

```bash
uv run evaluate submissions/models/simple.py -b ibm01
```

```
git submodule update --init external/MacroPlacement           # run once after clone
uv sync                                                       # run once after clone

uv run evaluate submissions/models/core.py -b ibm01           # first benchmark sanity check
uv run evaluate submissions/models/simple.py -b ibm01         # copied model sanity check

uv run evaluate submissions/models/core.py                    # default single-benchmark run
uv run evaluate submissions/models/core.py --all              # all IBM benchmarks
uv run evaluate submissions/models/core.py --vis              # save visualization
```
