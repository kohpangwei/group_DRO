# Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization

This code implements the group DRO algorithm from the following paper:

> Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori Hashimoto, and Percy Liang
>
> [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization]

The experiments use the following datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Waterbirds, formed from [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) + [Places](http://places2.csail.mit.edu/)
- [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)

We are working on releasing the Waterbirds dataset as well as scripts that will replicate the experiments from the above paper. Please stay tuned!

## Abstract

Overparameterized neural networks
can be highly accurate _on average_ on an i.i.d. test set
yet consistently fail on atypical groups of the data
(e.g., by learning spurious correlations that hold on average but not in such groups).
Distributionally robust optimization (DRO) allows us to learn models that instead
minimize the _worst-case_ training loss over a set of pre-defined groups.
However, we find that naively applying group DRO to overparameterized neural networks fails:
these models can perfectly fit the training data,
and any model with vanishing average training loss
also already has vanishing worst-case training loss.
Instead, their poor worst-case performance arises from poor _generalization_ on some groups.
By coupling group DRO models with increased regularization---stronger-than-typical L2 regularization or early stopping---we achieve substantially higher worst-group accuracies,
with 10-40 percentage point improvements
on a natural language inference task and two image tasks, while maintaining high average accuracies.
Our results suggest that regularization is critical for worst-group generalization in the overparameterized regime, even if it is not needed for average generalization.
Finally, we introduce and give convergence guarantees for a stochastic optimizer for the group DRO setting, underpinning the empirical study above.

## Prerequisites
- python 3.6.8
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- pillow 5.4.1
- pytorch 1.1.0
- pytorch_transformers 1.2.0
- torchvision 0.5.0a0+19315e3
- tqdm 4.32.2
