# ai-projB
# Ideas

# Param ideas:

## MCTS
Right now it seems like:
sims = 70, max_depth = 30 and c = 0.25 is giving pretty decent results

## Some things that would would be sick if you checked out and tried to make better:

1. Sometimes when there is only a few moves available the agent will grow even if its not beneficial at all, even though I de-weighted growing. Would be sick if you could figure that out.

1. Check out the new `best_child` function in `mcts.py`. I tried a bunch of different strats to change the weighting of certain moves during play. For example in early game its pretty much a random choice between going towards the centre or growing. Then it gets back to the normal formula near the end of the game.
Throughout the whole game, there's a penalty for allowing opponent multi-jumps.
Also there is a strong bias for multi-jumps during the whole game. It seems to work out pretty well. Like in early game we usually get at least 1 frog in the last row via some crazy multi-jump. 

1. Check out dynamic_c, this is a function that changes the c value based on how far we are into the game. 

1. Check out the Strat class and the RandomStrat in the `rangomagent` folder. Maybe implement other strats if you wanna try them out. Weighted random would be good and also just using the count heuristic thing in mcts.

To run the mcts agent against the random agent you should change the current .
You just need to run
```bash
python -m referee randomagent agent
```

Here red will be a random agent and blue will be mcts, just swap arguments to change order.

1. Although I've tried to make it smarter. In the start of the game it kind of plays like the random agent. Doesn't seem to smart. Maybe check that out
