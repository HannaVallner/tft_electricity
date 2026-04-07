import numpy as np
import torch
from torch.utils.data import Dataset


class TFTDataset(Dataset):
    """
    PyTorch-style dataset for Temporal Fusion Transformer.

    This class converts a formatted dataframe into sliding-window samples, where
    - data is grouped by entity ID
    - each entity is sorted by time
    - overlapping windows of length total_time_steps are created
    - model inputs use all formatter-defined columns except ID and TIME
    - targets keep only the decoder part
    - time and identifier are also stored for debugging / prediction formatting
    """

    def __init__(self, df, formatter):
        """
        Parameters
        ----------
        df : pd.DataFrame
            A formatted dataframe, e.g. train / valid / test from the formatter.

        formatter : ElectricityFormatter
            The formatter object used to define columns and fixed params.
        """
        self.df = df.copy()
        self.formatter = formatter

        fixed_params = formatter.get_fixed_params()
        self.total_time_steps = fixed_params["total_time_steps"]
        self.num_encoder_steps = fixed_params["num_encoder_steps"]
        self.decoder_steps = self.total_time_steps - self.num_encoder_steps

        self.column_definition = formatter.get_column_definition()

        self.id_col = self._get_single_col_by_type("ID")
        self.time_col = self._get_single_col_by_type("TIME")
        self.target_col = self._get_single_col_by_type("TARGET")

        # use every formatter-defined column except ID and TIME as model inputs
        self.input_cols = [
            name
            for name, _, role in self.column_definition
            if role.name not in {"ID", "TIME"}
        ]

        # Build all samples once during initialization
        self.samples = self._build_samples()

    def _get_single_col_by_type(self, input_type_name):
        """
        Returns the column name matching the requested input type.

        Example:
            "ID" -> "id"
            "TIME" -> "hours_from_start"
            "TARGET" -> "power_usage"
        """
        matches = [
            name
            for name, _, role in self.column_definition
            if role.name == input_type_name
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one column with type {input_type_name}, found {len(matches)}."
            )

        return matches[0]

    def _build_samples(self):
        """
        Builds all sliding-window samples.

        For each entity:
        1. sort by time
        2. create overlapping windows of length total_time_steps
        3. store:
            - inputs: full input window
            - outputs: decoder part of target only
            - time: full time window
            - identifier: full id window
            - active_entries: ones like in the original code
        """
        data = self.df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

        samples = []

        for identifier, sliced in data.groupby(self.id_col):
            sliced = sliced.sort_values(self.time_col).reset_index(drop=True)

            num_rows = len(sliced)
            if num_rows < self.total_time_steps:
                continue

            # Convert once for efficiency
            input_array = sliced[self.input_cols].to_numpy(dtype=np.float32)
            target_array = sliced[[self.target_col]].to_numpy(dtype=np.float32)

            # Keep time and identifier as arrays too
            time_array = sliced[[self.time_col]].to_numpy()
            identifier_array = sliced[[self.id_col]].to_numpy()

            # Create overlapping windows
            for start_idx in range(num_rows - self.total_time_steps + 1):
                end_idx = start_idx + self.total_time_steps

                x = input_array[start_idx:end_idx]

                # Keep only decoder part of target
                y = target_array[start_idx:end_idx][self.num_encoder_steps:]

                time_window = time_array[start_idx:end_idx]
                id_window = identifier_array[start_idx:end_idx]

                active_entries = np.ones_like(y, dtype=np.float32)

                samples.append({
                    "inputs": torch.tensor(x.copy(), dtype=torch.float32),
                    "outputs": torch.tensor(y.copy(), dtype=torch.float32),
                    "active_entries": torch.tensor(active_entries.copy(), dtype=torch.float32),
                    "time": time_window.copy(),
                    "identifier": id_window.copy(),
                })

        if not samples:
            raise ValueError("No valid sliding-window samples could be created.")

        return samples

    def __len__(self):
        """ Returns the number of sliding-window samples """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns one training sample.

        Returns a dictionary with:
        - inputs: torch.FloatTensor, shape (total_time_steps, input_size)
        - outputs: torch.FloatTensor, shape (decoder_steps, 1)
        - active_entries: torch.FloatTensor, shape (decoder_steps, 1)
        """
        sample = self.samples[idx]
        return {
            "inputs": sample["inputs"],
            "outputs": sample["outputs"],
            "active_entries": sample["active_entries"],
        }

    def summary(self):
        """ Prints dataset summary (for debugging) """
        first_sample = self.samples[0]

        print("TFTDataset summary")
        print(f"  number of samples:   {len(self.samples)}")
        print(f"  input columns:       {self.input_cols}")
        print(f"  target column:       {self.target_col}")
        print(f"  inputs shape:        {first_sample['inputs'].shape}")
        print(f"  outputs shape:       {first_sample['outputs'].shape}")
        print(f"  active_entries:      {first_sample['active_entries'].shape}")
        print(f"  time shape:          {first_sample['time'].shape}")
        print(f"  identifier shape:    {first_sample['identifier'].shape}")