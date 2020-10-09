import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np


def process_df(train_df, val_df, test_df, params):
    loss_metrics = []
    acc_metrics = []
    for group_idx in range(params['n_groups']):
        loss_metrics.append(f'avg_loss_group:{group_idx}')
        acc_metrics.append(f'avg_acc_group:{group_idx}')
    # robust acc
    for df in [train_df, val_df, test_df]:
        df['robust_loss'] = np.max(df.loc[:, loss_metrics], axis=1)
        df['robust_acc'] = np.min(df.loc[:, acc_metrics], axis=1)


def process_df_waterbird9(train_df, val_df, test_df, params):
    process_df(train_df, val_df, test_df, params)
    loss_metrics = []
    acc_metrics = []
    for group_idx in range(params['n_groups']):
        loss_metrics.append(f'avg_loss_group:{group_idx}')
        acc_metrics.append(f'avg_acc_group:{group_idx}')

    ratio = params['n_train'] / np.sum(params['n_train'])
    val_df['avg_acc'] = val_df.loc[:, acc_metrics] @ ratio
    val_df['avg_loss'] = val_df.loc[:, loss_metrics] @ ratio
    test_df['avg_acc'] = test_df.loc[:, acc_metrics] @ ratio
    test_df['avg_loss'] = test_df.loc[:, loss_metrics] @ ratio


def sanitize_df(df):
    """
    Fix a results df for problems arising from resuming.
    """
    # Remove stray epoch/batches
    duplicates = df.duplicated(
        subset=['epoch', 'batch'],
        keep='last')
    df = df.loc[~duplicates, :]
    df.index = np.arange(len(df))

    if np.sum(duplicates) > 0:
        print(f"Removed {np.sum(duplicates)} duplicates from epochs {np.unique(df.loc[duplicates, 'epoch'])}")

    # Make sure epoch/batch is increasing monotonically
    prev_epoch = -1
    prev_batch = -1
    last_batch_in_epoch = -1
    for i in range(len(df)):
        try:
            epoch, batch = df.loc[i, ['epoch', 'batch']].astype(int)
        except:
            print (i, epoch, batch, len(df))
        assert (
            ((prev_epoch == epoch) and (prev_batch < batch)) or
            ((prev_epoch == epoch - 1))
        )
        if prev_epoch == epoch - 1:
            assert ((last_batch_in_epoch == -1) or (last_batch_in_epoch == prev_batch))
            last_batch_in_epoch = prev_batch
        prev_epoch = epoch
        prev_batch = batch

    return df


def load_log(run_dir):
    dfs = []
    for split in ['train', 'val', 'test']:
        log_path = os.path.join(run_dir, 'log', f'{split}.csv')
        if os.path.exists(log_path):
            df = sanitize_df(
                pd.read_csv(log_path))
            dfs.append(df)
        else:
            print(f'Could not find {log_path}')
            dfs.append(None)
    return tuple(dfs)


def get_accs_for_epoch_across_batches(df, epoch):
    n_groups = 1 + np.max([int(col.split(':')[1]) for col in df.columns if col.startswith('avg_acc_group')])

    indices = df['epoch'] == epoch

    accs = np.zeros(n_groups)
    total_counts = np.zeros(n_groups)
    correct_counts = np.zeros(n_groups)

    for i in np.where(indices)[0]:
        for group in range(n_groups):
            total_counts[group] += df.loc[i, f'processed_data_count_group:{group}']
            correct_counts[group] += np.round(
                df.loc[i, f'avg_acc_group:{group}'] * df.loc[i, f'processed_data_count_group:{group}'])

    accs = correct_counts / total_counts

    robust_acc = np.min(accs)
    avg_acc = accs @ total_counts / np.sum(total_counts)

    return avg_acc, robust_acc


def print_accs(dfs, params=None,
               epoch_to_eval=None, print_avg=False, output=True,
               splits=['train', 'val', 'test'],
               early_stop=True):
    """
    Input: dictionary of dfs with keys 'val', 'test'
    This takes the minority group 'n' for calculating stdev,
    which is conservative.
    Since clean val/test acc for waterbirds is estimated from a val/test set with a different distribution, there's probably a bit more variability,
    but this is minor since the overall n is high.
    """
    for split in splits:
        assert split in dfs

    early_stopping_epoch = np.argmax(dfs['val']['robust_acc'].values)

    epochs = []
    assert early_stop or (epoch_to_eval is not None)
    if early_stop:
        epochs += [('early stop at epoch', 'early_stopping', early_stopping_epoch)]
    if epoch_to_eval is not None:
        epochs += [('epoch', 'epoch_to_eval', epoch_to_eval)]

    metrics = [('Robust', 'robust_acc')]
    if print_avg:
        metrics += [('Avg', 'avg_acc')]

    results = {}
    for metric_str, metric in metrics:
        results[metric] = {}

        for split in splits:
            for epoch_print_str, epoch_save_str, epoch in epochs:
                if epoch not in dfs[split]['epoch'].values:
                    if output:
                        print(f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch_to_eval}):               Not yet run")
                else:
                    if split == 'train':
                        avg_acc, robust_acc = get_accs_for_epoch_across_batches(dfs[split], epoch)
                        if metric == 'avg_acc':
                            acc = avg_acc
                        elif metric == 'robust_acc':
                            acc = robust_acc
                    else:
                        idx = np.where(dfs[split]['epoch'] == epoch)[0][-1] # Take the last batch in this epoch
                        acc = dfs[split].loc[idx, metric]

                    if split not in results[metric]:
                        results[metric][split] = {}

                    if params is None:
                        if output:
                            print(f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch}): "
                              f"{acc*100:.1f}")
                    else:
                        n_str = f'n_{split}'
                        minority_n = np.min(params[n_str])
                        total_n = np.sum(params[n_str])
                        if metric == 'robust_acc':
                            n = minority_n
                        elif metric == 'avg_acc':
                            n = total_n

                        stddev = np.sqrt(acc * (1 - acc) / n)
                        results[metric][split][epoch_save_str] = (acc, stddev)

                        if output:
                            print(f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch}): "
                              f"{acc*100:.1f} ({stddev*100:.1f})")

    return results


def print_best_adj_wd_accs(dfs, params, epoch_to_eval=None, print_avg=False,
                        splits=['train', 'val', 'test']):
    robust_accs = []
    wd = params['adjusted_wd']
    for adj in params['adj_list']:
        adj_dfs = dfs[adj][wd]
        if epoch_to_eval is None:
            epoch = np.argmax(adj_dfs['val']['robust_acc'].values)
        else:
            epoch = epoch_to_eval
        robust_accs.append(adj_dfs['val'].loc[epoch,'robust_acc'])
    best_adj = params['adj_list'][np.argmax(robust_accs)]
    print(f'==================  DRO, adj={best_adj} ================== ')
    return print_accs(
        dfs[best_adj][wd],
        params,
        epoch_to_eval=epoch_to_eval,
        print_avg=print_avg,
        splits=splits)


def print_best_adj_accs(dfs, params, epoch_to_eval=None, print_avg=False,
                        splits=['train', 'val', 'test']):
    robust_accs = []
    wd = params['adjusted_wd']
    for adj in params['adj_list']:
        adj_dfs = dfs[adj][wd]
        if epoch_to_eval is None:
            epoch = np.argmax(adj_dfs['val']['robust_acc'].values)
        else:
            epoch = epoch_to_eval
        robust_accs.append(adj_dfs['val'].loc[epoch,'robust_acc'])
    best_adj = params['adj_list'][np.argmax(robust_accs)]
    print(f'==================  DRO, adj={best_adj} ================== ')
    return print_accs(
        dfs[best_adj][wd],
        params,
        epoch_to_eval=epoch_to_eval,
        print_avg=print_avg,
        splits=splits)


def print_best_wd_accs(dfs, params, epoch_to_eval=None, print_avg=False,
                       splits=['train', 'val', 'test']):
    robust_accs = []
    for wd in params['wd']:
        if epoch_to_eval is None:
            epoch = np.argmax(dfs[wd]['val']['robust_acc'].values)
        else:
            epoch = epoch_to_eval
        robust_accs.append(dfs[wd]['val'].loc[epoch,'robust_acc'])
    best_wd = params['wd'][np.argmax(robust_accs)]
    print(f'=== wd={best_wd}')
    return print_accs(
        dfs[best_wd],
        params,
        epoch_to_eval=epoch_to_eval,
        print_avg=print_avg,
        splits=splits)


def plot_adj_sweep(dfs, params, acc=False, ylim=None, plot_train=True, plot_val=True):
    fig, ax = plt.subplots(1, len(params['adj_list']),
                           figsize=(20,4),
                           sharey=True, sharex=True)
    for i_adj,adj in enumerate(params['adj_list']):
        if acc:
            plotted_col='avg_acc'
        else:
            plotted_col='avg_loss'
        wd = params['adjusted_wd']
        legend = []
        for group_idx in range(params['n_groups']):
            df = dfs[adj][wd]
            if df is None:
                continue
            plot_train_val_losses(ax[i_adj], df['train'], df['val'],
                                  f'{plotted_col}_group:{group_idx}', f'C{group_idx}',
                                  title=f'adj={adj}', plot_train=plot_train, plot_val=plot_val)
            legend.append(f'group {group_idx}')
            legend.append('_no_legend')
        ax[i_adj].legend(legend)
        ax[i_adj].set_xlabel(plotted_col)
        fig.tight_layout()
        ax[i_adj].set_ylim(ylim)


def plot_train_val_losses(ax, train_df, val_df, y_cols, color, title, x_column=None, x_cumsum=False,
    plot_train=True, plot_val=True):

    assert plot_train or plot_val

    df = train_df.merge(val_df, on='epoch', suffixes=['_train','_val'])

    if isinstance(y_cols, tuple):
        assert(len(y_cols) == 2)
    else:
        y_cols = (y_cols,)

    val_col = y_cols[0] + '_val'
    train_col = y_cols[0] + '_train'
    if x_column is None:
        x = np.arange(df.shape[0])
        xlabel = 'batch'
    else:
        x = df[x_column].values
        if x_cumsum:
            x = np.cumsum(x)
        xlabel = x_column
    if plot_val: ax.plot(x, df[val_col], color=color, label=val_col)
    if plot_train: ax.plot(x, df[train_col], linestyle='--', color=color, label=train_col, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_cols[0])
    ax.grid(linestyle='--')
    ax.set_title(title)

    if len(y_cols) > 1:
        ax2 = ax.twinx()
        val_col = y_cols[1] + '_val'
        train_col = y_cols[1] + '_train'
        color = 'C' + str(int(color[1]) + 2)
        if plot_val: ax2.plot(x, df[val_col], color=color, label=val_col)
        if plot_train: ax2.plot(x, df[train_col], linestyle='--', color=color, label=train_col, alpha=0.5)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(y_cols[1])
        ax2.set_ylim((0, 1))
        ax2.set_title(title)


def scatter_train_vs_val(ax, train_df, val_df, train_column, val_column, train_cumsum=False, val_xumsum=False,
                         color='C0', title=''):
    train_df = train_df.groupby('epoch').mean().reset_index()
    df = train_df.merge(val_df, on='epoch', suffixes=['_train','_val'])
    ax.scatter(df[train_column+"_train"], df[val_column+'_val'], color=color, alpha=0.5)
    ax.set_xlabel(train_column+"_train")
    ax.set_ylabel(val_column+"_val")


def compute_stats_last_epoch(train_df, val_df, column, epoch_column='epoch'):
    last_epoch = max(val_df[epoch_column])
    train_loss = train_df[train_df[epoch_column]==last_epoch][column].mean()
    val_loss = val_df[val_df[epoch_column]==last_epoch][column].values
    return train_loss, val_loss


def scatter_train_vs_val_last_epoch(ax, train_df, val_df, train_column, val_column,
                                    epoch_column='epoch', color='C0'):
    last_epoch = max(val_df[epoch_column])
    train_loss = train_df[train_df[epoch_column]==last_epoch][train_column].mean()
    val_loss = val_df[val_df[epoch_column]==last_epoch][val_column].values
    ax.scatter(train_loss, val_loss, color=color, alpha=0.5)
    ax.set_xlabel(train_column+"_train")
    ax.set_ylabel(val_column+"_val")


def scatter_gen_gap_last_epoch(ax, x, train_df, val_df, column,
                               epoch_column='epoch', color='C0'):
    last_epoch = max(val_df[epoch_column])
    train_loss = train_df[train_df[epoch_column]==last_epoch][column].mean()
    val_loss = val_df[val_df[epoch_column]==last_epoch][column].values
    ax.scatter(x, val_loss - train_loss, color=color, alpha=0.5)
    ax.set_ylabel("generalization gap")


def scatter_train_and_val_last_epoch(ax, x, train_df, val_df, column,
                                     epoch_column='epoch', color='C0'):
    last_epoch = max(val_df[epoch_column])
    train_loss = train_df[train_df[epoch_column]==last_epoch][column].mean()
    val_loss = val_df[val_df[epoch_column]==last_epoch][column].values
    ax.scatter(x, train_loss, color=color, facecolors='none')
    ax.scatter(x, val_loss, color=color, alpha=0.5)
    ax.set_xlabel(column)


def load_log_old(run_dir):
    names = ['train_loss', 'train_acc',
             'train_loss_0', 'train_loss_1', 'train_loss_2', 'train_loss_3',
             'val_loss', 'val_acc',
             'val_loss_0', 'val_loss_1', 'val_loss_2', 'val_loss_3',
             'val_acc_0', 'val_acc_1', 'val_acc_2', 'val_acc_3']
    log_path = os.path.join(run_dir, 'log', 'log.csv')
    try:
        df = pd.read_csv(log_path, names=names, header=0)
    except pd.errors.ParserError:
        df = pd.read_csv(log_path, names=names[:-4], header=0)
    return df


def plot_train_val_losses_old(ax, df, group_idx, color, title):
    if group_idx is None:
        val_col='val_loss'
        train_col='train_loss'
    else:
        val_col = f'val_loss_{group_idx}'
        train_col = f'train_loss_{group_idx}'
    ax.plot(np.arange(df.shape[0]), df[val_col], color=color, label=val_col)
    ax.plot(np.arange(df.shape[0]), df[train_col], linestyle='--', color=color, label=train_col)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(linestyle='--')
    ax.set_title(title)
