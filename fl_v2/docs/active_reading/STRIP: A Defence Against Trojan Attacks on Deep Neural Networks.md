# Title：

STRIP: A Defence Against Trojan Attacks on Deep Neural Networks

## Threat model (attacker knowledge / control / budget)

The attacker aims to return a **trojaned / backdoored model** that behaves normally on clean inputs but is hijacked when a secret trigger is present. The paper mainly focuses on **input-agnostic triggers**, meaning that any input image stamped with the trigger is misclassified into the attacker’s target class.

The attacker is assumed to have strong capability:

- full access to the training dataset;
- white-box access to the DNN model / architecture;
- control over the trigger’s pattern, location, and size.

In an FL context, this is **not** originally a malicious-client threat model. The paper does not define clients, client updates, FedAvg, non-IID data, malicious client ratios, or server aggregation. STRIP is closer to an **outsourced / centralized training + deployed-model runtime detection** setting.

The paper discusses federated learning only in related work, noting that FL can be vulnerable to trojan attacks, but STRIP itself is not a robust aggregation defense.

---

## Defender knowledge (clean data? labeled? attack-aware?)

The defender has a small **held-out clean validation set**, but does **not** have access to trojaned / poisoned samples. The defender also does not know the trigger’s shape, location, size, or target class.

STRIP does not recover the trigger and does not require access to internal neurons. It only needs to:

- perturb incoming inputs;
- feed the perturbed replicas into the deployed model;
- observe the model’s output probabilities / softmax;
- compute prediction entropy.

The held-out data do not need to be used with ground-truth labels in the STRIP mechanism. Unlike SentiNet, STRIP does not rely on the ground-truth labels of either the incoming input or the randomly drawn held-out samples. It mainly uses the **randomness / entropy of model predictions under perturbation**.

The defender is attack-aware in the sense that they know a trojan / backdoor threat may exist and therefore deploy STRIP. However, the defender is not trigger-aware: the trigger is secret, arbitrary in shape and color, may appear at any location, and may have any size.

---

## Mechanism, one sentence

**STRIP strongly perturbs each incoming input by superimposing it with multiple clean held-out images; if the perturbed replicas still produce stable, low-entropy predictions, the input is considered likely to be trojaned.**

Short version:

**Perturb the input; low entropy under perturbation implies trigger-dominated behavior.**

The core mechanism is that clean inputs should be **input-dependent**: once strongly perturbed, their predictions should vary or become uncertain. In contrast, an input-agnostic trigger can dominate the model’s prediction even after perturbation, causing the perturbed inputs to remain confidently classified into the attacker’s target class.

---

## The single quantitative claim that survives if everything else is noise

In the tested **CIFAR10** and **GTSRB** input-agnostic trigger settings, using `N = 100` perturbed replicas, STRIP reports **0% FAR** at preset FRR values including **1% FRR**.

The paper further observes that, for these settings, the minimum entropy of tested clean inputs is larger than the maximum entropy of tested trojaned inputs, producing a clear clean / trojan entropy gap.

A safe thesis-style wording is:

**In the tested input-agnostic trigger settings on CIFAR10 and GTSRB, STRIP achieves 0% false acceptance rate at 1% false rejection rate by exploiting a clear entropy gap between clean and triggered inputs.**

Do **not** overstate this as:

**STRIP achieves 0% FAR / FRR for all backdoors.**

The result is demonstrated under specific datasets, triggers, models, and input-agnostic attack assumptions. Also, Table III reports that for MNIST with the square trigger, FAR is 1.85% when FRR is set to 1%, so the strongest defensible quantitative claim should be tied to the specific CIFAR10 / GTSRB settings rather than generalized to all cases.

---

## The bypass / failure mode

The most important failure mode is **source-label-specific / partial backdoor**.

STRIP is designed for input-agnostic triggers:

**any input + trigger → target class**

A source-label-specific backdoor instead behaves like:

**specific source classes + trigger → target class**

and:

**other classes + trigger → normal or non-target behavior**

The paper explicitly states that source-label-specific triggers are outside its main threat model and remain an important challenge. It notes that STRIP, Neural Cleanse, and SentiNet mainly focus on input-agnostic trojan attacks and appear ineffective against source-label-specific triggers under the assumption that the defender has no trojaned samples.

The reason is mechanistic: STRIP relies on the trigger causing stable target predictions under strong perturbation. But in a partial backdoor, the trigger only works when the source-class condition is preserved. Random superimposition may destroy or weaken that source condition, so the perturbed triggered input may no longer produce consistently low entropy.

A second important adaptive risk is **entropy manipulation**. If the attacker knows STRIP detects low-entropy behavior, they can train the model so that perturbed trojaned inputs produce random-looking predictions, making clean and trojaned entropy distributions more similar.

The paper evaluates such an adaptive attack and reports that it can preserve high attack success while changing the entropy behavior, although the authors argue that the clean entropy distribution becomes abnormal and can still indicate a malicious model.

---

## What this implies for a general defense

STRIP shows that input-agnostic backdoors can leave a clear **output-space behavioral signature**: under strong perturbation, triggered inputs remain unusually stable and low-entropy. This is useful for runtime input filtering and deployed-model auditing.

However, a single entropy-based runtime detector is not a general backdoor defense. A more general defense must handle:

1. **Runtime input filtering**  
   Detect whether an incoming input is trigger-dominated at test time.

2. **Training-time / server-side auditing**  
   In FL, inspect whether the global model or client updates are becoming backdoored.

3. **Source-conditional and client-conditional backdoors**  
   Handle attacks that activate only for specific source classes, clients, regions, driving scenarios, or data distributions.

4. **Adaptive attacks**  
   Evaluate attackers that explicitly optimize against the entropy statistic used by STRIP.

5. **Representation-space analysis**  
   Go beyond softmax entropy and inspect whether triggers hijack penultimate-layer features, classifier heads, or backbone representations.

For my thesis, the main implication is:

**STRIP is useful as a deployment-side or server-audit baseline for FL-based autonomous driving, but it is not itself a robust aggregation defense. Its value is in showing that input-agnostic triggers can produce perturbation-invariant behavior, while source-specific, context-dependent, and adaptive backdoors are more realistic and harder FL threat models.**