# Notes

```
.
├── submissions/
│   └── models/             # all different models we build
│       ├── MODELS.md       # all notes of models
│       └── core.py         # baseline model to create new models from
└── NOTES.md                # This
```

- Make a python file like ``submissions/models/simple.py`` and add a description of the model in ``submissions/models/MODELS.md`` to keep track of what we do.
- Reference ``submissions/models/core.py`` as starting point, contains simple scaffolding for easy testing
- Test for erros by running only on first benchmark ``uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
``

# Testing

```
git submodule update --init external/MacroPlacement                     # run once after clone
uv sync                                                                 # run once after clone

uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01      # only first benchmark

uv run evaluate submissions/examples/greedy_row_placer.py
uv run evaluate submissions/examples/greedy_row_placer.py --all
uv run evaluate submissions/examples/greedy_row_placer.py --vis
```