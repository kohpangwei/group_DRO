# Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization

This code implements the group DRO algorithm from the following paper:

> Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori Hashimoto, and Percy Liang
>
> [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization]

The experiments use the following datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Waterbirds, formed from [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) + [Places](http://places2.csail.mit.edu/)
- [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)

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

## Datasets and code

To run our code, you will need to change the `root_dir` variable in `data/data.py`.
The main point of entry to the code is `run_expt.py`. Below, we provide sample commands for each dataset.

### CelebA

Our code expects the following files/folders in the `[root_dir]/celebA` directory:

- `data/list_eval_partition.csv`
- `data/list_attr_celeba.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). The original dataset, due to Liu et al. (2015), can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

A sample command to run group DRO on CelebA is:
`python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 128 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0`


### Waterbirds

The Waterbirds dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).

Our code expects the following files/folders in the `[root_dir]/cub` directory:

- `data/waterbird_complete95_forest2water2/`

You can download a tarball of this dataset [here](https://nlp.stanford.edu/data/waterbird_complete95_forest2water2.tar.gz).

A sample command to run group DRO on Waterbirds is:
`python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 128 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0`

Note that compared to the training set, the validation and test sets are constructed with different proportions of each group. We describe this in more detail in Appendix C.1 of our paper, which we reproduce here for convenience:

> We use the official train-test split of the CUB dataset, randomly choosing 20% of the training data to serve as a validation set. For the validation and test sets, we allocate distribute landbirds and waterbirds equally to land and water backgrounds (i.e., there are the same number of landbirds on land vs. water backgrounds, and separately, the same number of waterbirds on land vs. water backgrounds). This allows us to more accurately measure the performance of the rare groups, and it is particularly important for the Waterbirds dataset because of its relatively small size; otherwise, the smaller groups (waterbirds on land and landbirds on water) would have too few samples to accurately estimate performance on. We note that we can only do this for the Waterbirds dataset because we control the generation process; for the other datasets, we cannot generate more samples from the rare groups.

> In a typical application, the validation set might be constructed by randomly dividing up the available training data. We emphasize that this is not the case here: the training set is skewed, whereas the validation set is more balanced. We followed this construction so that we could better compare ERM vs. reweighting vs. group DRO techniques using a stable set of hyperparameters. In practice, if the validation set were also skewed, we might expect hyperparameter tuning based on worst-group accuracy to be more challenging and noisy.

> Due to the above procedure, when reporting average test accuracy in our experiments,
we calculate the average test accuracy over each group and then report a weighted average, with weights corresponding to the relative proportion of each group in the (skewed) training dataset.

If you'd like to generate variants of this dataset, we have included the script we used to generate this dataset (from the CUB and Places datasets) in `dataset_scripts/generate_waterbirds.py`. Note that running this script will not create the exact dataset we provide above, due to random seed differences. You will need to download the [CUB dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) as well as the [Places dataset](http://places2.csail.mit.edu/download.html). We use the high-resolution training images (MD5: 67e186b496a84c929568076ed01a8aa1) from Places. Once you have downloaded and extracted these datasets, edit the corresponding paths in `generate_waterbirds.py`.


### MultiNLI with annotated negations

Our code expects the following files/folders in the `[root_dir]/multinli` directory:

- `data/metadata_random.csv`
- `glue_data/MNLI/`

We have included the metadata file in `dataset_metadata/multinli` in this repository. The GLUE data can be downloaded with [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e); please note that you only need to download MNLI and not the other datasets. The metadata file records whether each example belongs to the train/val/test dataset as well as whether it contains a negation word.

A sample command to run group DRO on MultiNLI is:
`python run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --lr 2e-05 --batch_size 32 --weight_decay 0 --model bert --n_epochs 20`

We created our own train/val/test split of the MultiNLI dataset, as described in Appendix C.1 of our paper:

> The standard MultiNLI train-test split allocates most examples (approximately 90%) to the training set, with another 5% as a publicly-available development set and the last 5% as a held-out test set that is only accessible through online competition leaderboards (Williams et al., 2018). To accurately estimate performance on rare groups in the validation and test sets, we combine the training set and development set and then randomly resplit it to a 50-20-30 train-val-test split that allocates more examples to the validation and test sets than the standard split.

If you'd like to modify the metadata file (e.g., considering other confounders than the presence of negation words), we have included the script we used to generate the metadata file in `dataset_scripts/generate_multinli.py`. Note that running this script will not create the exact dataset we provide above, due to random seed differences. You will need to download the [MultiNLI dataset](https://www.nyu.edu/projects/bowman/multinli/) and edit the paths in that script accordingly.
