# Title:

DBA: Distributed Backdoor Attacks Against Federated Learning

## Threat model (attacker knowledge / control / budget)

The attacker controls several malicious clients in a federated learning system. Each malicious client has full control over its own local training process, including local data poisoning, local learning rate, local epochs, poison ratio, scale factor, and poisoning schedule.

The attacker cannot modify the server-side aggregation rule, cannot directly tamper with benign clients, and cannot alter other clients' updates.

The attack is not a strict one-to-one source-to-target attack. It is better understood as an all-to-one targeted backdoor attack: samples from non-target classes are poisoned with a trigger and relabeled to one fixed target class.

The attack budget is distributed across multiple malicious clients. Instead of one attacker inserting the full global trigger, each malicious client inserts only one local part of the trigger. In the image experiments, the global trigger is usually split across four malicious clients. In the tabular LOAN experiment, six low-importance features are split across three malicious clients.

The paper studies two attack modes:

- **A-M / multiple-shot attack:** malicious clients repeatedly inject local-trigger updates across multiple rounds.
- **A-S / single-shot attack:** malicious clients use scaling to make one attack round strong enough to survive aggregation.

A key caveat is that the single-shot setting depends on a large scaling factor, which improves attack persistence but can also make malicious updates easier to detect by magnitude-based defenses.

## Defender knowledge (clean data? labeled? attack-aware?)

The server mainly observes client updates and performs aggregation. The paper does not assume that the defender has labeled clean validation data for explicit backdoor detection.

The evaluated defenses are aggregation-level defenses:

- **RFA:** replaces standard averaging with an approximate geometric median to reduce the influence of outlier updates.
- **FoolsGold:** downweights clients that repeatedly submit highly similar gradient updates, aiming to mitigate sybil-style poisoning attacks.

These defenses are attack-aware in the broad sense that they are designed for poisoning or centralized backdoor attacks, but they are not specifically designed for distributed partial-trigger attacks.

This matters because DBA does not necessarily make any single malicious update look highly abnormal. The malicious signal appears mainly through the combined effect of multiple weak local-trigger updates after aggregation.

## Mechanism, one sentence

DBA decomposes a global trigger into several local triggers, assigns each local trigger to a different malicious client, and relies on federated aggregation to combine these weak local backdoor associations into a strong global-trigger backdoor.

## The single quantitative claim that survives if everything else is noise

The most memorable quantitative claim is from the single-shot attack setting on MNIST:

After 50 benign training rounds, DBA still achieves about **89% attack success rate**, while the centralized backdoor attack drops to about **21% attack success rate**.

This supports the paper's central claim that DBA is not only effective but also more persistent than centralized backdoor attacks.

A second important number is from the FoolsGold experiment on MNIST:

Under FoolsGold, DBA reaches about **91.55% attack success rate** at round 30, while the centralized attack reaches only about **2.91% attack success rate**.

This supports the claim that DBA can bypass similarity-based robust aggregation better than centralized attacks.

## The bypass / failure mode (the paper's own admission, or what a later paper shows)

### Bypass mode

DBA bypasses RFA because each malicious client only trains on a local trigger. Therefore, each malicious update is smaller and less distant from benign updates than a centralized update trained on the full global trigger. RFA looks for outliers using distance to a geometric median, so DBA weakens the signal that RFA relies on.

DBA bypasses FoolsGold because different malicious clients use different local triggers. Their updates share the same target objective but are not identical in update space. FoolsGold detects repeated similarity among clients, so DBA weakens the similarity signal. Even if individual DBA clients are downweighted, the sum of their aggregation weights can still be large enough to create the backdoor.

In short:

- RFA expects malicious updates to be far away.
- FoolsGold expects malicious clients to look similar.
- DBA makes malicious updates individually closer to benign updates and mutually less identical.

### Failure mode

DBA is not unconditionally effective.

The paper's own analysis shows several limitations:

- Large scaling improves persistence but creates anomalous updates that may be detected by parameter-magnitude checks.
- If local triggers are too small, the model may not learn a usable trigger signal.
- If local triggers are too far apart, the global trigger may fail because the model cannot compose the separated local patterns effectively.
- If the poison ratio is too high, the local model may lose clean accuracy, and scaling such a poor local model can damage the global model.
- If the poisoning interval is too short, several scaled malicious updates may distort the global model too much.
- If the poisoning interval is too long, early local-trigger effects may be forgotten before later local triggers are inserted.
- Stronger Byzantine-style aggregation methods produce mixed results, so the paper does not establish that DBA bypasses all robust aggregation methods.

## What this implies for a general defense

A general defense against federated backdoors should not only search for single-client outliers or highly similar malicious clients.

DBA shows that a backdoor can be distributed across clients such that:

- each individual malicious update appears relatively benign;
- different malicious updates are not necessarily similar;
- the harmful behavior only appears after aggregation;
- the full trigger is never seen by any single malicious client during training;
- local triggers may be weak individually, while the global trigger is strong.

Therefore, a stronger defense should analyze cross-client composition rather than only per-client anomaly.

Possible defense directions:

1. **Cross-client target-logit analysis**

   Check whether multiple clients independently push different input patterns toward the same target class.

2. **Partial-trigger and combined-trigger validation**

   Evaluate not only complete triggers but also whether several weak partial patterns jointly create a strong target-class response.

3. **Update subspace analysis**

   Look for groups of clients whose updates are individually small but collectively move the model in a consistent target-related direction.

4. **Norm clipping plus robust aggregation**

   Since DBA can use scaling, clipping can reduce the effect of large malicious updates before robust aggregation is applied.

5. **Representation-space monitoring**

   Track whether unrelated local patterns become aligned with the same target-class representation after aggregation.

6. **Defense against compositional backdoors**

   The defense should assume that the malicious behavior may not be visible in any single client update, but only in the aggregate effect of several clients.

## My thesis takeaway

DBA is valuable because it changes the way we should think about federated backdoor attacks. The main threat is not a single malicious client with an obvious full trigger, but multiple malicious clients whose weak local attacks are composed by the FL aggregation process.

The key lesson is:

A federated learning defense must reason about the collective effect of client updates, not only whether each individual update is abnormal.

For future work, the most promising direction is to study DBA in representation space: determine whether local triggers create weak target-logit shifts that become additive after aggregation, and design defenses that detect this cross-client compositional behavior before the full global trigger becomes effective.