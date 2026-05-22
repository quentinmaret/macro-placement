## Reflections

Our basic method works by running a tournament that selects and prunes the best chains of our poorly named “initializers.” In reality, these are more like deterministic refiners than true initializers.

We tried several approaches to improve our proxy score, but came away with a few main takeaways:

1. **It is hard to beat the human expert baselines.**  
   The IBM baselines were difficult to beat with any deterministic placement method. In hindsight, this should have been a signal to switch earlier to an RL-based placer, but this was a long and busy quarter for us.

2. **Refinement is a limited fix.**  
   This is something we also should have realized earlier. Improving and iterating on refiners is inherently limited because refinement is bounded by the basin that the initial placer puts the solution in.

3. **Iterate, experiment, and optimize a known strong implementation.**  
   Do not try to jerry-rig a better solution from scratch, especially in a heavily researched field that we were exploring for the first time.

4. **Give yourself more time.**  
   This was hard with our schedules, but we should have carved out more time at the beginning so that our iterative workflow was in place for multiple weeks and adding new experiments was easy.

I want to come back to macro placement, or another placement-style problem, and apply RL more seriously. I should have used this opportunity to sharpen my skills in that area.
