# Results

> **Canonical artifact:** `results/results.json` (20 runs). The tables below were derived directly from that file.

## Summary

Mean ± sample standard deviation across five seeds (0–4). Bold is the best value in each column.

| Activation | Test accuracy | Test accuracy 95% CI | Train BCE | Test BCE | Epochs to BCE ≤ 0.30 | First-layer grad norm (last 10) | Dead units | Saturated units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sigmoid | 95.11 ± 0.61% | [94.36, 95.87] | 0.1434 | **0.1390** | 48.8 ± 16.6 | 0.0641 | 0.00% | 1.25% |
| Tanh | **95.78 ± 0.30%** | [95.40, 96.16] | 0.1158 | 0.1430 | 6.6 ± 1.7 | 0.1729 | 0.00% | 6.80% |
| ReLU | 94.56 ± 1.73% | [92.41, 96.70] | 0.0982 | 0.1751 | **3.6 ± 0.5** | **0.2386** | 13.75% | 0.00% |
| LeakyReLU (α = 0.01) | 95.56 ± 0.79% | [94.58, 96.53] | **0.0909** | 0.1751 | **3.6 ± 0.5** | 0.2354 | **0.00%** | **0.00%** |

Confidence intervals are t-based 95% intervals over five seeds (mean ± 2.776 × s.d. / √5, df = 4). Treat them as descriptive; five seeds are not enough for strong interval claims.

## Per-seed results

| Activation | Seed | Test acc. | Train loss | Test loss | Conv. epoch | Grad norm | Dead | Saturated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sigmoid | 0 | 94.44% | 0.1359 | 0.1293 | 64 | 0.0580 | 0.00% | 2.38% |
| Sigmoid | 1 | 95.56% | 0.1434 | 0.1346 | 60 | 0.0578 | 0.00% | 0.93% |
| Sigmoid | 2 | 94.44% | 0.1519 | 0.1592 | 27 | 0.0684 | 0.00% | 0.73% |
| Sigmoid | 3 | 95.56% | 0.1399 | 0.1318 | 35 | 0.0630 | 0.00% | 1.11% |
| Sigmoid | 4 | 95.56% | 0.1457 | 0.1402 | 58 | 0.0734 | 0.00% | 1.10% |
| Tanh | 0 | 96.11% | 0.1243 | 0.1447 | 5 | 0.1755 | 0.00% | 4.93% |
| Tanh | 1 | 96.11% | 0.1095 | 0.1432 | 7 | 0.1675 | 0.00% | 8.85% |
| Tanh | 2 | 95.56% | 0.1172 | 0.1473 | 5 | 0.1803 | 0.00% | 9.53% |
| Tanh | 3 | 95.56% | 0.1145 | 0.1424 | 9 | 0.1893 | 0.00% | 3.12% |
| Tanh | 4 | 95.56% | 0.1134 | 0.1376 | 7 | 0.1519 | 0.00% | 7.57% |
| ReLU | 0 | 95.56% | 0.0998 | 0.1795 | 4 | 0.2192 | 16.67% | 0.00% |
| ReLU | 1 | 95.00% | 0.0904 | 0.1521 | 4 | 0.2230 | 17.71% | 0.00% |
| ReLU | 2 | 91.67% | 0.1209 | 0.1815 | 3 | 0.3520 | 10.42% | 0.00% |
| ReLU | 3 | 96.11% | 0.0926 | 0.1653 | 4 | 0.1735 | 15.62% | 0.00% |
| ReLU | 4 | 94.44% | 0.0875 | 0.1971 | 3 | 0.2252 | 8.33% | 0.00% |
| LeakyReLU | 0 | 96.11% | 0.0981 | 0.1575 | 4 | 0.2205 | 0.00% | 0.00% |
| LeakyReLU | 1 | 95.00% | 0.0930 | 0.1542 | 4 | 0.2776 | 0.00% | 0.00% |
| LeakyReLU | 2 | 96.11% | 0.0919 | 0.1703 | 3 | 0.2214 | 0.00% | 0.00% |
| LeakyReLU | 3 | 96.11% | 0.0832 | 0.1880 | 4 | 0.2267 | 0.00% | 0.00% |
| LeakyReLU | 4 | 94.44% | 0.0884 | 0.2057 | 3 | 0.2310 | 0.00% | 0.00% |

## Statistical tests

Tests compare final test accuracy across the four activation groups (n = 5 per group).

| Test | Statistic | p-value | Interpretation |
|---|---:|---:|---|
| One-way ANOVA | F = 1.429 | 0.271 | No evidence of a mean-accuracy difference |
| Kruskal-Wallis | H = 4.119 | 0.249 | Rank-based result agrees with ANOVA |

Neither test rejects the null at α = 0.05. The result is consistent with the visual picture in the accuracy plot: the means are close, and the within-group variation (especially for ReLU) is comparable to the between-group gaps.

**Power caveat.** Five seeds per group provide low power for small effects. The correct reading is “not detected here,” not “no difference exists.”

## Observations

### Convergence is where the activations separate

| Activation | Mean convergence epoch | Relative to Sigmoid |
|---|---:|---:|
| Sigmoid | 48.8 | 1.0× |
| Tanh | 6.6 | 7.4× faster |
| ReLU | 3.6 | 13.6× faster |
| LeakyReLU | 3.6 | 13.6× faster |

The speed gap is large, stable across seeds, and visible in the training-loss curves. Sigmoid also has the largest convergence spread (27 to 64 epochs), which matches its sensitivity to early gradient attenuation.

### Gradient flow explains the speed gap

Mean first-layer gradient norm over the last ten epochs:

| Activation | Grad norm | Relative to ReLU |
|---|---:|---:|
| Sigmoid | 0.0641 | 0.27× |
| Tanh | 0.1729 | 0.72× |
| ReLU | 0.2386 | 1.00× |
| LeakyReLU | 0.2354 | 0.99× |

ReLU and LeakyReLU preserve a strong gradient signal to the first layer. Sigmoid does not: its derivative is at most 0.25, and deep composition shrinks the signal further. Tanh sits in the middle, with a larger derivative near zero but still vulnerable to saturation.

### Test loss tells a different story than test accuracy

Sigmoid has the lowest mean test BCE (0.1390) despite being the slowest to converge and finishing third in mean accuracy. ReLU and LeakyReLU have the lowest training BCE but the highest test BCE (0.1751). The ranking depends on which quantity you care about: calibration/loss or classification accuracy. Reporting only accuracy would hide this tradeoff.

### ReLU pays a dead-unit tax; LeakyReLU avoids it

ReLU dead-unit fractions by seed: 16.67%, 17.71%, 10.42%, 15.62%, 8.33% (mean 13.75%). The worst seed also produced the lowest test accuracy in the entire experiment (91.67%, ReLU seed 2). We cannot attribute that single outcome to dead units alone, but the pattern is consistent with the known failure mode: a dead ReLU has zero gradient and cannot recover.

LeakyReLU changes the negative slope from 0 to 0.01 and eliminates dead units entirely in this run, with no convergence penalty.

### Tanh saturates but still performs well

Tanh had 6.80% saturated hidden units on average, higher than Sigmoid's 1.25%. This looks counterintuitive until you separate *saturation* from *optimization speed*: Tanh's outputs are zero-centered, which keeps gradients better conditioned than Sigmoid's positive-only outputs, and its derivative at zero is 1.0 rather than 0.25. Saturation is real; it is not the whole story.

## Artifact checksums

| Artifact | SHA-256 |
|---|---|
| `results/results.json` | `42b2d5a51ded02dfcc7b796a2e0818b6ddf30a3e907389287539663b3e373428` |
| `results/activation_summary.csv` | Regenerated from `results.json` by `activation_benchmark.analysis` |

The figure PNGs are deterministic outputs of the plotting code given the same `results.json`; their exact bytes can vary with Matplotlib/font rendering across environments. The numbers they depict cannot.