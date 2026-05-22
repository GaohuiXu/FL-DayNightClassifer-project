# Title:

 *The Limitations of Federated Learning in Sybil Settings* — FoolsGold for sybil-based targeted poisoning defense

## Threat model:

 One adversary can create/control many sybil clients; sybils observe global model state and can submit arbitrary gradients, but cannot see honest clients’ data or individual honest updates; server is honest; no effective identity verification; focus is clone-based targeted poisoning.

## Defender knowledge:

 Server sees individual client updates and stable client histories, but no raw data, no clean labeled validation set, no exact trigger/target knowledge, and no secure aggregation.

## Mechanism, one sentence:

 Penalize clients whose historical updates are abnormally similar to other clients, because sybil clones pursuing the same poisoning objective should produce similar update directions.

## Single quantitative claim:

 In MNIST label-flipping/backdoor with 0–9 sybils, FoolsGold remains robust while mean/Multi-Krum/median/trimmed mean fail as sybil count increases; A-99 further tests 990 sybils vs 10 honest clients.

## Bypass / failure mode:

 Fails or weakens when the attack does not create high inter-malicious similarity: single-client model replacement, A-1 attack, adaptive low-frequency poisoning, intelligent perturbation, and DBA/distributed backdoor.

## Implication for general defense:

 Similarity-based defense is useful but insufficient; a general FL backdoor defense needs multi-signal update analysis: similarity, norm, outlierness, coordinate usage, layer-wise attribution, temporal persistence, and robustness to adaptive/distributed attacks.