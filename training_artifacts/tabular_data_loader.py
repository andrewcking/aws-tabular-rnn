import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Data is expected as a pandas dataframe saved/loaded in pickle format
    Series on y axis, time on x with multiindex of feature vectors
    """

    def __init__(self, data_path, data_path_norms_csv, num_inputs, out_size):
        self.df = pd.read_csv(data_path, header=[0, 1], index_col=[0])  # dataset dataframe
        self.funnel_norms = pd.read_csv(data_path_norms_csv, index_col=0)  # dataframe for normalization metrics
        self.num_inputs = num_inputs  # number of inputs
        self.feature_size = out_size  # number of outputs at each time step

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx, :]  # get row
        data = data[~pd.isnull(data)]  # remove null years
        data = data.values.reshape(-1, self.num_inputs)
        data = torch.Tensor(data)  # convert to Torch Tensor

        # normalize data on the fly
        for idx, col in enumerate(self.funnel_norms.columns.values):
            data[:, idx:idx + 1] = (data[:, idx:idx + 1] - self.funnel_norms[col]['min']) / (self.funnel_norms[col]['max'] - self.funnel_norms[col]['min'])

        # Shape is [number of series, number of features]
        row_x = data[:-1]  # get all but last year (X)

        # Shape is [number of series, number of features]
        row_y = data[1:, :self.feature_size]  # get all but first year (Y) and remove external features as we are not predicting them

        return {'X': row_x, 'Y': row_y}
