# Methodology

> **Question.** When data, architecture, initialization, optimizer, and training budget are held fixed, how do Sigmoid, Tanh, ReLU, and LeakyReLU differ in optimization speed, gradient health, and generalization?

This document specifies the experiment precisely enough to reproduce it. The implementation lives in `src/activation_benchmark`; the canonical artifact is `results/results.json`.

## 1. Research design

The benchmark is a **single-factor, repeated-measures design**. The factor is the hidden-layer activation function with four levels. The replication unit is a random seed. Every other design choice is held constant.

| Element | Choice |
|---|---|
| Task | Binary classification on the two-moons dataset |
| Activations | Sigmoid, Tanh, ReLU, LeakyReLU (negative slope 0.01) |
| Architecture | 2 → 32 → 32 → 32 → 1 |
| Seeds | 0, 1, 2, 3, 4 |
| Runs | 4 activations × 5 seeds = 20 |
| Epochs per run | 150 |
| Primary outcome | Test accuracy after the final epoch |
| Secondary outcomes | Convergence epoch, final losses, first-layer gradient norm, dead/saturated unit fractions |

The design deliberately excludes early stopping, learning-rate schedules, dropout, batch normalization, weight decay, and per-activation tuning. Those choices would confound the comparison: a gain could come from the activation or from the extra tuning budget.

## 2. Data

The dataset is generated once with scikit-learn:

~~~python
X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
~~~

A stratified 70/30 split produces 420 training and 180 test examples, with 210/210 and 90/90 class balance respectively. The split seed is 42.

The feature scaler is fitted on the **training set only**:

~~~python
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
~~~

Both splits are stored as `float32`. Fitting the scaler on the full dataset would leak test-set statistics into training; the implementation avoids that.

Two moons is a useful stress test because it is nonlinear but low-dimensional. A deep MLP can solve it comfortably, so differences in **optimization behavior** remain visible even when accuracy is near ceiling.

## 3. Model

For input $x \in \mathbb{R}^2$, the network computes

~~~text
h1 = phi(W1 x + b1)      W1: 32 x 2,  b1: 32
h2 = phi(W2 h1 + b2)     W2: 32 x 32, b2: 32
h3 = phi(W3 h2 + b3)     W3: 32 x 32, b3: 32
logit = W4 h3 + b4       W4: 1 x 32,  b4: 1
~~~

The output is a single logit; `BCEWithLogitsLoss` applies the sigmoid internally. The predicted class is `logit >= 0`, equivalent to a probability threshold of 0.5.

### Parameter count

| Layer | Weight shape | Weight params | Bias params | Total |
|---|---|---:|---:|---:|
| Input → hidden 1 | 32 × 2 | 64 | 32 | 96 |
| Hidden 1 → hidden 2 | 32 × 32 | 1,024 | 32 | 1,056 |
| Hidden 2 → hidden 3 | 32 × 32 | 1,024 | 32 | 1,056 |
| Hidden 3 → output | 1 × 32 | 32 | 1 | 33 |
| **Total** | | **2,144** | **97** | **2,241** |

Activation functions add no parameters, so all four models have exactly the same capacity.

## 4. Initialization

Weights are drawn with Xavier-uniform initialization:

~~~text
W ~ U(-a, a),   a = sqrt(6 / (fan_in + fan_out))
b = 0
~~~

The seed is set before the model is constructed. Activation modules consume no random numbers, so for a given seed every activation receives **element-wise identical initial weights and biases**. This is the key control: any difference in the learning curves is attributable to the nonlinearity, not to a lucky initialization.

## 5. Optimization

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 150 |
| Loss | Binary cross-entropy with logits |
| Shuffle | Uniform permutation, regenerated each epoch |
| Threads | 1 PyTorch CPU thread |

The minibatch generator is seeded with `seed + 1000`. Each activation therefore sees the same sequence of minibatches for a given seed, up to the numerical effects of the forward pass.

There is no early stopping. Every run is evaluated after exactly 150 epochs. The convergence epoch is a **post-hoc diagnostic**, not a training decision.

## 6. Metrics

### Test accuracy

`mean(prediction == label)` on the 180 held-out examples, with predictions thresholded at a logit of zero.

### Convergence epoch

The first epoch at which the epoch-average training loss is at most **0.30**. If a run never reaches the threshold, the value is null; all 20 reported runs reached it. The threshold is a reporting device chosen before analysis, not a tuned parameter.

### First-layer gradient norm

For each minibatch, the L2 norm of the gradient of the loss with respect to the first linear layer's weight matrix is recorded. The per-epoch value is the mean over batches. The reported statistic is the mean of the final ten epochs. It measures whether useful signal still reaches the earliest layer late in training.

### Dead units

For each ReLU hidden layer, a unit is dead if its output is at most `1e-8` for **every training example** at the end of training. The reported fraction is the mean over the three hidden layers. LeakyReLU and the saturating activations have no dead-unit definition and are assigned zero.

### Saturated units

For Sigmoid, a unit is saturated if its output is below 0.01 or above 0.99. For Tanh, a unit is saturated if `|h| > 0.99`. The fraction is averaged over hidden layers. ReLU and LeakyReLU are assigned zero because their outputs are unbounded and the notion does not apply in the same way.

### Timing

Wall-clock training time per run and mean milliseconds per epoch are recorded for context. They are hardware-dependent and should not be compared across machines.

## 7. Statistical analysis

The unit of analysis is a training run. For each activation we report the mean and sample standard deviation (ddof = 1) across five seeds.

Two omnibus tests compare the four accuracy distributions:

- **One-way ANOVA** — parametric, assumes approximately normal residuals and equal variances.
- **Kruskal-Wallis H** — rank-based, no normality assumption.

No post-hoc pairwise tests are reported because neither omnibus test rejects the null at α = 0.05. With five seeds per group, the study is powered to detect large effects only; a non-significant result is not evidence that the activations are equivalent.

## 8. Controls

- **Same data and split** for all runs.
- **Same scaler**, fitted on training data only.
- **Same architecture and parameter count** (2,241).
- **Same initial weights per seed** across activations.
- **Same optimizer, learning rate, batch size, and epoch budget.**
- **Same minibatch order per seed** via the `seed + 1000` generator.
- **No per-activation tuning**, early stopping, or checkpoint selection.
- **Single-thread execution** for comparable timings.

## 9. Threats to validity

**Single task and architecture.** Two moons with three hidden layers of width 32 is a controlled probe, not a survey. Conclusions may not transfer to images, sequences, transformers, or much wider networks.

**Small replication.** Five seeds per activation yield wide confidence intervals. The ANOVA has low power; a true difference of a few tenths of a percentage point would be difficult to detect.

**Near-ceiling accuracy.** All models exceed 94.5% test accuracy. The task may be too easy to expose generalization differences that appear on harder benchmarks.

**One optimization configuration.** A different learning rate, batch size, or depth could reorder the convergence ranking. The claim is about this configuration, not about activations in general.

**Metric definitions.** Dead and saturated fractions depend on thresholds and on measuring activations on the training set at the final epoch. A unit that is temporarily dead, or dead only on test data, is not counted.

**Implementation details.** PyTorch's `ReLU`, `LeakyReLU`, `Tanh`, and `Sigmoid` are standard, but numerical details and CPU kernels can vary across versions. Accuracy was stable across our runs; exact bitwise reproducibility is not guaranteed.

**Timing.** Wall-clock numbers include Python overhead and are not a hardware benchmark.

## 10. What would change the conclusion

A follow-up should vary at least one of: dataset hardness, depth, width, learning rate, batch size, initialization scale, and the number of seeds. If the accuracy ordering is stable across those changes, it is a property of the activations; if it flips, it is a property of the configuration. The repository is structured so that such sweeps can reuse the same data, model, and analysis code.

## References

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
2. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *ICML*.
3. Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities improve neural network acoustic models. *ICML Workshop on Deep Learning for Audio, Speech and Language*.
4. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers. *ICCV*.
6. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
7. Hochreiter, S. (2001). The vanishing gradient problem during learning recurrent neural nets and problem solutions. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*.
8. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (2002). Efficient backprop. In *Neural Networks: Tricks of the Trade*.
9. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
11. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*.
12. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.