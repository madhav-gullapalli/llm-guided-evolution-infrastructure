# Search-efficiency prompt groups (USE mode)

Prompt groups for evolving the surrogate-usage strategy (`SurrogateUsageOptimizer`, USE mode). Each group has 3
templates, and the pipeline picks one at random per individual, the same as the other `naslib/*` groups. Use one with
`--prompt_groups "naslib/search_efficiency_v4"` (or v1–v3).

| group | what it adds | why |
|---|---|---|
| `search_efficiency` (v1) | Describes the code as the usage strategy; goal = fewer evaluations to reach 99% of the best accuracy; basic "keep runnable" rules | `naslib/general` describes the code as a surrogate model and targets ranking only |
| `search_efficiency_v2` | "How this code is run" section: which methods `eval.py` calls, the candidate / acquisition / ensemble interfaces, the tools that exist, no new imports | in v1, 10 of 15 individuals only edited `new_epoch` / `_acquisition`, which `eval.py` replaces, so they had no effect |
| `search_efficiency_v3` | Goal adds top-10% ranking; "What we already know" feedback with measured baselines; research-backed ideas (local search, exploration, larger ensemble); exact `mutate` / `get_hash` usage | v2's only gain came from `self.k = 1`, which lowered overall Kendall Tau |
| `search_efficiency_v4` | Same as v3, except the goal targets overall Kendall Tau (about 0.39) at about 45 evaluations, plus 3 feedback lines on the speed/Tau trade-off | v3's best strategy lowered overall Kendall Tau (0.354 vs 0.392) |

Notes:
- v3 and v4 assume the seed is BANANAS with `self.k = 1` ("This code already refits after every evaluation"). With the
  default `usage_bananas.py`, that feedback line is wrong. Use v1/v2 with the default seed, or add a k = 1 seed.
- The numbers in the v3/v4 feedback come from re-tests on NAS-Bench-201 CIFAR-100 with a 10-architecture random warm-up
  and 100 evaluations, averaged over 20 search seeds.

Results (fresh seeds 20–39; evaluations to reach 99% of the best accuracy / overall Kendall Tau):
- original BANANAS (k = 10): 88.3 / 0.392
- BANANAS with k = 1: 59.5 / 0.329
- best v3 individual: 44.7 / 0.354
- best v4 individual: 61.4 / 0.385
