# AlphaZero Training Performance Improvement Plan

## Overview

This document outlines a systematic experimental approach to improve training performance and break the first-player bias currently observed in the JGGame AlphaZero implementation.

---

## Experiment 1: Baseline Run (Validate Fixes)

### Goal
Confirm the architecture/reward fixes enable learning at all

### Configuration
- **cpuct:** 3 (current setting)
- **numEps:** 100 (episodes per iteration)
- **numMCTSSims:** 200 (MCTS simulations)
- **tempThreshold:** 20 (temperature decay)
- **Iterations:** 30

### Expected Data Volume
- ~1,500 examples/iteration × 30 = **~45,000 total training examples**
- **Estimated time:** 6-8 hours (hardware dependent)

### Success Criteria Timeline

#### Iterations 1-10 (Baseline Phase)
- **Expected:** P1 win rate stays near 100%, losses barely decrease
- **Normal:** Network is learning the "P1 always wins" pattern

#### Iterations 10-20 (Critical Window)
- **Target metrics:**
  - P1 win rate drops to 80-90%
  - Value loss (`train/epoch/v_loss`) stabilizes below 0.8
  - New models occasionally beat old ones (`arena-result/is_new_better` > 0)
  - Games start getting longer (indicates defense learning)

#### Iterations 20-30 (Validation Phase)
- **Target metrics:**
  - P1 win rate: 60-70%
  - New model acceptance rate: 30-50%
  - Clear tactical play visible in game logs

### Decision Point
- **If by iteration 25:** P1 win rate still >95% → **STOP**, proceed to Experiment 2
- **If P1 win rate <80%:** **SUCCESS!** Continue to 50 iterations for full convergence

---

## Experiment 2: Exploration Parameter Sweep

### Goal
Find optimal exploration constant to break P1 bias faster

### When to Run
**Only run this if Experiment 1 fails to show improvement by iteration 25**

### Configurations to Test

Run these in parallel (preferred) or sequentially:

| Config | cpuct | Iterations | Expected Outcome | Use Case |
|--------|-------|------------|------------------|----------|
| A (Low) | 1 | 20 | Baseline (likely fails) | Comparison point |
| B (Mid) | 4 | 20 | Most likely to work | Recommended |
| C (High) | 8 | 20 | High exploration | Slower but robust |

### What to Watch
- Which configuration shows P1 win rate dropping first?
- Monitor exploration vs exploitation trade-off
- Games should show more variety in move sequences

### Decision Point
- Pick the configuration with fastest P1 win rate decline
- Continue winner to 50 iterations
- If all fail, proceed to Experiment 3

---

## Experiment 3: Network Capacity Test

### Goal
Determine if the network lacks capacity to learn defensive strategies

### When to Run
**Only if Experiments 1-2 both fail to show improvement**

### Configuration Change
In `JGNet.py`, modify:
```python
nn_args = dotdict(
    {
        # ...
        "num_channels": 2048,  # Increase from 1024
    }
)
```

### Rationale
- Defense is typically harder to learn than offense
- A larger network provides more representational capacity
- Trade-off: Slower training, higher memory usage

### Settings
- Use best `cpuct` from Experiment 2
- Run for 20 iterations
- Monitor for improvement in defensive play

### Expected Impact
- Slower convergence initially
- Better final performance if capacity was the bottleneck

---

## Experiment 4: MCTS Simulation Depth

### Goal
Test if more planning time helps discover defensive strategies

### When to Run
As a secondary optimization after finding a working baseline

### Configuration
- **numMCTSSims:** 400 (double current 200)
- **Iterations:** 15

### Trade-offs
- **Pros:** Higher quality training data, better move discovery
- **Cons:** 2x slower training time

### Expected Outcome
- More consistent training signal
- Potentially faster learning (fewer iterations needed)

---

## Key Metrics to Monitor

Track these metrics in WandB or log files:

### Primary Metrics

1. **`arena-result/first_player_win_rate`**
   - **Current:** ~1.0 (100%)
   - **Target (iter 25):** <0.7 (70%)
   - **Target (iter 50):** 0.55-0.65 (healthy balance)

2. **`arena-result/is_new_better`**
   - **Current:** ~0.0
   - **Target (iter 20):** >0.3 (30% improvement rate)
   - **Target (iter 50):** >0.5 (50% improvement rate)

### Secondary Metrics

3. **`train/epoch/v_loss`** (Value prediction accuracy)
   - **Target:** Stabilize below 0.7
   - **Pattern:** Should show downward trend

4. **`train/epoch/pi_loss`** (Policy learning)
   - **Target:** Gradual decrease
   - **Pattern:** May fluctuate initially

5. **Average game length**
   - **Current:** ~10-15 turns
   - **Expected improvement:** 15-25 turns (indicates defense)

---

## Success Indicators by Phase

### Early Signs (Iterations 5-15)
- ✓ Games get longer (defense emerging)
- ✓ `v_loss` stops oscillating wildly
- ✓ Occasional draws appear in arena games
- ✓ Move variety increases in self-play

### Medium Term (Iterations 15-30)
- ✓ P1 win rate: 70-80%
- ✓ New model beats old: 40% of the time
- ✓ Clear improvement in move quality (visible in game logs)
- ✓ More examples of blocking/defensive moves

### Long Term (Iterations 30-50)
- ✓ P1 win rate: 55-65%
- ✓ New model consistently beats old (>50%)
- ✓ Arena games show tactical play
- ✓ Training loss stabilizes

---

## Recommended Execution Plan

### Phase 1: Initial Run
1. **Start Experiment 1** with current configuration
2. Let run overnight or for ~8 hours
3. Review metrics at iteration 15-20

### Phase 2: Decision Point (Iteration 20-25)
- **If P1 win rate <80%:** Continue to iteration 50 ✓
- **If P1 win rate >90%:** Launch Experiment 2 (cpuct sweep)
- **If partial improvement:** Extend to iteration 40 before deciding

### Phase 3: Optimization (If Phase 1 Succeeds)
- Run Experiment 4 (higher MCTS sims) for comparison
- Fine-tune `cpuct` if needed
- Consider curriculum learning or other advanced techniques

### Phase 4: Troubleshooting (If All Fail)
- Review game logs manually for patterns
- Test Experiment 3 (larger network)
- Consider fundamental game balance issues

---

## Estimated Timeline

| Experiment | Iterations | Est. Runtime | Total Time |
|------------|------------|--------------|------------|
| Exp 1 (Baseline) | 30 | 15 min/iter | 7.5 hours |
| Exp 2 (Config A) | 20 | 15 min/iter | 5 hours |
| Exp 2 (Config B) | 20 | 15 min/iter | 5 hours |
| Exp 2 (Config C) | 20 | 15 min/iter | 5 hours |
| Exp 3 (Large Net) | 20 | 25 min/iter | 8.3 hours |
| Exp 4 (Deep MCTS) | 15 | 30 min/iter | 7.5 hours |

**Total worst-case time:** ~38 hours (if all experiments needed)
**Expected time:** ~12-20 hours (Exp 1 + follow-up)

---

## Notes

- All iterations include self-play, training, and arena evaluation
- Runtime estimates assume GPU training (MPS/CUDA)
- Consider running Experiment 2 configs in parallel if you have multiple GPUs
- Save checkpoints frequently to allow rollback if needed
- Monitor WandB dashboard regularly during critical iterations (15-25)

---

## Next Steps

1. ✓ Review this plan
2. Launch Experiment 1 with current configuration
3. Set up monitoring/alerts for key metrics
4. Check progress at iteration 15 and 25
5. Follow decision tree based on results
