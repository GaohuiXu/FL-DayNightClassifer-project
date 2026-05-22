# Title:

BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain

## Threat model (attacker knowledge / control / budget):

The paper studies two related supply-chain threat models. In **outsourced training**, the user specifies the DNN architecture and provides training data, while a potentially malicious trainer returns trained parameters. **The attacker is very strong**: they may poison the training data, modify training hyperparameters such as learning rate and batch size, or even directly set the returned weights, as long as the returned model matches the user-specified architecture and passes clean validation. In **transfer learning**, the attacker provides a malicious pretrained model from an online repository; the user later adapts it to a new task using transfer learning. The attacker’s objective is to preserve high clean accuracy while causing attacker-chosen behavior on inputs containing a secret trigger.

## Defender knowledge (clean data? labeled? attack-aware?):

The defender/user has a held-out clean validation set with labels and accepts the model if its validation accuracy exceeds a target threshold. The user is not assumed to know the trigger, and standard validation contains only clean images. The paper mainly assumes the defender is not attack-aware beyond checking clean validation accuracy. No dedicated backdoor detection, trigger search, activation inspection, pruning, or robust fine-tuning defense is evaluated.

## Mechanism, one sentence:

BadNets injects a backdoor by adding triggered versions of selected training samples with attacker-chosen labels, so the same fixed architecture learns normal behavior on clean inputs and malicious behavior on triggered inputs.

## The single quantitative claim that survives if everything else is noise:

A BadNet can maintain nearly the same clean accuracy as the baseline while achieving very high attack success on triggered inputs; for MNIST, the all-to-all attack reports 0.48% clean error and 0.56% backdoor error against poisoned labels, meaning triggered inputs are classified according to the attacker’s target more than 99% of the time.

## The bypass / failure mode (the paper's own admission, or what a later paper shows):

The attack bypasses standard clean validation because the validation set does not contain triggered inputs. However, the paper does not show that BadNets evade active backdoor defenses. Its evidence is limited by missing random seeds/variance, simple visible triggers, weak physical-world validation, and detection experiments that report mostly classification-style accuracy rather than standard object-detection metrics such as mAP, IoU, or proposal recall.

## What this implies for a general defense:

A general backdoor defense cannot rely only on clean validation accuracy. It should inspect model behavior under synthetic or optimized triggers, compare clean and triggered representations, test fine-tuning/pruning sensitivity, and evaluate both clean utility and attack success rate. For detection models, defense evaluation should also include localization metrics, not only classification accuracy.