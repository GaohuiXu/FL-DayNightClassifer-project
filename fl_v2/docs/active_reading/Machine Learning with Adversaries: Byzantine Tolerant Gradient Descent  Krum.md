# Title:

Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent / Krum

## Threat model (attacker knowledge / control / budget):

The system has one reliable parameter server and n workers, up to f of which can be Byzantine. A Byzantine worker can behave arbitrarily: instead of submitting the correct stochastic gradient estimate, it may upload any vector, including random noise, a scaled opposite gradient, or an adaptive vector constructed with knowledge of the aggregation rule and other workers' submitted vectors. Byzantine workers may also collude. The formal guarantee requires 2f + 2 < n. This is a general Byzantine-gradient threat model, not a backdoor-specific threat model: no trigger, target label, poison ratio, stealth metric, or ASR is defined.

## Defender knowledge (clean data? labeled? attack-aware?):

The defender is the parameter server. It does not need clean validation data or labeled auxiliary data. It is attack-aware at the aggregation level because it assumes that up to f workers may be Byzantine and replaces averaging with a robust aggregation rule. However, the server must know or set an upper bound f. The server does not know which specific workers are Byzantine. The theory assumes honest workers submit i.i.d. unbiased stochastic gradient estimates of the same global objective, which is a strong assumption and does not directly match non-IID FedAvg-style FL.

## Mechanism, one sentence:

Krum scores each submitted update by the sum of squared Euclidean distances to its n − f − 2 nearest neighboring updates, selects the lowest-score update, and Multi-Krum averages the m best-scoring updates to trade off robustness and convergence speed.

## The single quantitative claim that survives if everything else is noise:

No linear aggregation rule, including averaging, can tolerate even one Byzantine worker because one malicious vector can force the aggregate to be an arbitrary vector U. Krum is theoretically Byzantine-resilient when 2f + 2 < n and η(n,f)√dσ < ||g||, meaning the true gradient signal must dominate honest-gradient noise. Empirically, with n = 20 on Spambase, averaging fails under 33% Gaussian Byzantine workers while Krum still converges; with sufficiently large mini-batches, Krum under 45% omniscient Byzantine workers can approach averaging with 0% Byzantine workers.

## The bypass / failure mode (the paper's own admission, or what a later paper shows):

Krum relies on honest updates being geometrically concentrated around the true gradient. This assumption can fail in non-IID FL, where honest client updates are naturally dispersed because of label skew, quantity skew, local epochs, client drift, and stochastic data augmentation. In that case, Krum may reject useful honest clients or select an update that only represents a local client cluster. For backdoor attacks, a malicious update may preserve clean-task direction and avoid being a Euclidean outlier while still implanting a trigger-target mapping. Therefore, Krum may fail both by damaging clean accuracy and by missing stealthy backdoor updates.

## What this implies for a general defense:

Krum/Multi-Krum are useful server-side robust aggregation baselines, but they are not sufficient as general FL backdoor defenses. A general defense should not rely only on update-space Euclidean outlierness. It should also measure honest-honest update dispersion, malicious-honest distance, cosine alignment to honest mean, Krum score rank, selected-client label distribution, clean accuracy, ASR, and preferably representation-space or validation-based semantic behavior.