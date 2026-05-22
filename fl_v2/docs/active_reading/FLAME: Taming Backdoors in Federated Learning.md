# Title:

FLAME: Taming Backdoors in Federated Learning

## Threat model (attacker knowledge / control / budget):

White-box FL backdoor attacker. The attacker fully controls \(k < n/2\) clients, including their local data, training process, hyperparameters, and uploaded model updates. The attacker knows the aggregator and the defense, but cannot control the server/aggregator or honest clients. The attack goal is targeted: maintain normal behavior on clean inputs while forcing attacker-chosen outputs on trigger inputs. The honest-majority constraint \(k<n/2\) is a hard assumption, not an implementation detail.

## Defender knowledge (clean data? labeled? attack-aware?):

The aggregator does not need access to raw client training data, clean validation data, poisoned data, trigger samples, or labels. It needs access to individual client model updates. The defense is attack-aware at the level of assuming backdoor/model-poisoning threats, but it does not assume a specific trigger pattern or a fixed number of backdoors. Standard FLAME is incompatible with plain secure aggregation unless modified, because it requires pairwise distances between individual updates.

## Mechanism, one sentence:

FLAME filters angular outliers with HDBSCAN over cosine distances, clips surviving updates to a median L2-norm ball around the previous global model, then adds Gaussian noise calibrated by the clipping bound to suppress remaining stealthy backdoor contributions.

## The single quantitative claim that survives if everything else is noise:

For constrain-and-scale attacks, FLAME reduces BA to 0 on Reddit, CIFAR-10, and IoT-Traffic while keeping MA close to the benign setting: Reddit 22.3 MA vs 22.7 benign, CIFAR-10 91.9 vs 92.2, IoT-Traffic 99.8 vs 100.0. In Table 4, competing baselines either leave BA high or damage MA substantially.

## The bypass / failure mode (the paper's own admission, or what a later paper shows):

The paper's own hard failure mode is malicious majority: if PMR exceeds 50%, the majority cluster or median clipping bound can be controlled by malicious clients, and FLAME fails. A second practical weakness is adaptive stealth: if an attacker can keep malicious updates close in both cosine direction and L2 norm while preserving a strong backdoor under expected Gaussian noise, filtering and clipping provide little signal. The paper tests some adaptive strategies, but not a fully optimized end-to-end white-box attack against all three modules.

## What this implies for a general defense:

A general FL backdoor defense should not only ask whether malicious updates are detectable as outliers. It must handle three regimes: direction outliers, magnitude-amplified updates, and stealthy residual updates. FLAME is valuable because it explicitly decomposes these regimes, but its "general" claim should be read as conditional: honest majority, homogeneous model architecture, visible individual updates, and empirically adequate Gaussian smoothing. A stronger general defense would need representation-level evidence, standardized detection metrics, robustness to stronger adaptive attackers, and compatibility with secure aggregation or personalized/heterogeneous FL.