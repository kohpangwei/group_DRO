import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class MultiNLIDataset(ConfounderDataset):
    """
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        assert len(confounder_names) == 1
        assert confounder_names[0] == 'sentence2_has_negation'
        assert target_name in ['gold_label_preset', 'gold_label_random']
        assert augment_data == False
        assert model_type == 'bert'

        self.data_dir = os.path.join(
            self.root_dir,
            'data')
        self.glue_dir = os.path.join(
            self.root_dir,
            'glue_data',
            'MNLI')
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')
        if not os.path.exists(self.glue_dir):
            raise ValueError(
                f'{self.glue_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        type_of_split = target_name.split('_')[-1]
        self.metadata_df = pd.read_csv(
            os.path.join(
                self.data_dir,
                f'metadata_{type_of_split}.csv'),
            index_col=0)

        # Get the y values
        # gold_label is hardcoded
        self.y_array = self.metadata_df['gold_label'].values
        self.n_classes = len(np.unique(self.y_array))

        self.confounder_array = self.metadata_df[confounder_names[0]].values
        self.n_confounders = len(confounder_names)

        # Map to groups
        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (self.y_array*(self.n_groups/self.n_classes) + self.confounder_array).astype('int')

        # Extract splits
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Load features
        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli',  
            'cached_dev_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli-mm'
            ]:

            features = torch.load(
                os.path.join(
                    self.glue_dir,
                    feature_file))

            self.features_array += features

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)

        self.x_array = torch.stack((
            self.all_input_ids,
            self.all_input_masks,
            self.all_segment_ids), dim=2)

        assert np.all(np.array(self.all_label_ids) == self.y_array)


    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        x = self.x_array[idx, ...]
        return x, y, g

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        attr_name = self.confounder_names[0]
        group_name = f'{self.target_name} = {int(y)}, {attr_name} = {int(c)}'
        return group_name
