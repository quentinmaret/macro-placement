## Reflections
--
Our basic method works by running a tornument which selects and prunes the best chains of our (poorly named) initializers.  
These are more determinisitc refiners than initializers. We tried a bunch of things to improve our proxy score but came   
with a few basic takeaways.  
    1. It's hard to beat the human expert, the IBM baselines were difficult to beat with any deterministic placement method.  
        In hindsight this should have been our signal to switch to a RL placer, but it has been as long and busy quarter for us.  
    2. Refinement is a limited fix. Again something we should have realized, but our improving and iterating on refiners is a   
        limited approach, bounded by the basin that he placer puts it in.   
    3. Iterate, experiment, and optimize a known strong implementation, don't try and jerry rig a better solution, especially in  
        a heavily researched field that you are exploring for the first time  
    4. Give yourself more time, hard with our schedules but we should have carved out more time in the beginning so that   
        our iterative flow is in places for multiple weeks and adding is easy  
I want to come back to Macro-placement or any other placement style problem and apply RL. I should have taken this oppurtunity   
to sharpen my skills in this regard  