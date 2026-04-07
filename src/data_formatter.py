from enum import IntEnum
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataTypes(IntEnum):
    """ Describes the type of a column """

    REAL_VALUED = 0     # Numeric values (power usage, hour index, etc.)
    CATEGORICAL = 1     # Categories (entity IDs, categorical time features)
    DATE = 2            # Actual datetime/date columns.


class InputTypes(IntEnum):
    """ Describes how the model should use a column """

    TARGET = 0          # The value we want to predict
    OBSERVED_INPUT = 1  # A feature observed from the past (past measured values)
    KNOWN_INPUT = 2     # A feature known in advance (hour of day etc.)
    STATIC_INPUT = 3    # A feature that remains unchanged for an entity (customer ID)
    ID = 4              # The column used as an entity identifier
    TIME = 5            # The time index column


def get_single_col_by_input_type(input_type, column_definition):
    """ 
    Returns the single column name that matches the given input type.
    Example: InputTypes.ID -> "id"
    """

    matching_columns = [
        column_name
        for column_name, _, role in column_definition
        if role == input_type
    ]

    if len(matching_columns) != 1:
        raise ValueError(
            f"Expected exactly one column with input type {input_type}, "
            f"but found {len(matching_columns)}."
        )
    
    return matching_columns[0]

def extract_cols_from_data_type(data_type, column_definition, excluded_input_types=None):
    """ Returns column names with a given data type, excluding some input types """

    if excluded_input_types is None:
        excluded_input_types = set()

    return [
        column_name
        for column_name, dtype, role in column_definition
        if dtype == data_type and role not in excluded_input_types
    ]

class ElectricityFormatter:
    """ 
    Defines and formats data for the electricity dataset, including:
        1. Describing which columns mean what
        2. Splitting the dataframe into train / validation / test
        3. Fitting scalers on the training data
        4. Transforming real-valued and categorical inputs
        5. Converting predictions back to the original scale
    """

    _column_definition = [
        ("id", DataTypes.CATEGORICAL, InputTypes.ID),
        ("hours_from_start", DataTypes.REAL_VALUED, InputTypes.TIME),
        ("power_usage", DataTypes.REAL_VALUED, InputTypes.TARGET),
        ("hour", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ("day_of_week", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ("month", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ("hours_from_start", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ("categorical_id", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """ Initialises formatter """

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']
    
    def get_column_definition(self):
        """ Returns formatted column definition in order expected by the TFT """

        column_definition = self._column_definition

        def _check_single_column(input_type):
            matches = [tup for tup in column_definition if tup[2] == input_type]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one column with input type {input_type}, "
                    f"found {len(matches)}."
                )
            
        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]

        real_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.REAL_VALUED
            and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        categorical_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.CATEGORICAL
            and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs
    
    @property
    def num_classes_per_cat_input(self):
        """ Returns the number of classes for each categorical input """
        return self._num_classes_per_cat_input
    
    def get_fixed_params(self):
        """
        Defines the fixed parameters used by the model for training:
            'total_time_steps': Total number of time steps used by TFT
            'num_encoder_steps': Length of LSTM encoder (i.e. history)
            'num_epochs': Maximum number of epochs for training
            'early_stopping_patience': Early stopping param for keras
            'multiprocessing_workers': no. of cpus for data processing
        """

        return {
            "total_time_steps": 8 * 24,     # 192 total steps
            "num_encoder_steps": 7 * 24,    # 168 history steps
            "num_epochs": 100,
            "early_stopping_patience": 5,
            "multiprocessing_workers": 5,
        }
    
    def get_default_model_params(self):
        """ Returns default model parameters """

        model_params = {
            'dropout_rate': 0.1,
            'hidden_layer_size': 40,
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 4,
            'stack_size': 1
        }
        return model_params
    
    def get_num_samples_for_calibration(self):
        """
        Returns sample counts used for calibration.
        """
        return 450000, 50000
    
    def split_data(self, df, valid_boundary=1315, test_boundary=1339):
        """ 
        Splits the dataframe into train / validation / test.
            Train: days_from_start < 1315
            Validation: 1308 <= days_from_start < 1339
            Test: days_from_start >= 1332
        """

        print('Formatting train-validation-test splits')

        day_index = df["days_from_start"]

        train = df.loc[day_index < valid_boundary].copy()
        valid = df.loc[(day_index >= valid_boundary - 7) & (day_index < test_boundary)].copy()
        test = df.loc[day_index >= test_boundary - 7].copy()

        # Fit scalers on the training data
        self.set_scalers(train)

        # Transform all splits using the train-fitted scalers
        train = self.transform_inputs(train)
        valid = self.transform_inputs(valid)
        test = self.transform_inputs(test)

        return train, valid, test

    def set_scalers(self, df):
        """ Fits all scalers and encoders using the training data """

        print('Setting scalers with training data...')
        column_definition = self.get_column_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID, column_definition)
        target_column = get_single_col_by_input_type(InputTypes.TARGET, column_definition)

        # Real-valued inputs except the special ID and TIME columns
        real_inputs = extract_cols_from_data_type(DataTypes.REAL_VALUED,
            column_definition,
            {InputTypes.ID, InputTypes.TIME},
        )

        # Initialise scaler caches
        self._real_scalers = {}
        self._target_scaler = {}
        identifiers = []

        # Fit one scaler per entity
        for identifier, sliced in df.groupby(id_column):
            # Keep only series long enough for one full model window
            if len(sliced) >= self._time_steps:
                real_data = sliced[real_inputs].values
                target_data = sliced[[target_column]].values

                self._real_scalers[identifier] = StandardScaler().fit(real_data)
                self._target_scaler[identifier] = StandardScaler().fit(target_data)

                identifiers.append(identifier)

        # Fit categorical encoders
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL,
            column_definition,
            {InputTypes.ID, InputTypes.TIME},
        )

        self._cat_scalers = {}
        num_classes = []

        for col in categorical_inputs:
            # Set all to str to avoid mixed integer/string columns
            as_string = df[col].astype(str)

            encoder = LabelEncoder()
            encoder.fit(as_string.values)

            self._cat_scalers[col] = encoder
            num_classes.append(as_string.nunique())

        self._num_classes_per_cat_input = num_classes
        self.identifiers = identifiers

    def transform_inputs(self, df):
        """
        Transforms a dataframe using the already-fitted scalers, 
        including preprocessing and normalisation.
        """

        if self._real_scalers is None or self._cat_scalers is None:
            raise ValueError("Scalers have not been set.")

        # Extract relevant columns
        column_definition = self.get_column_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID, column_definition)

        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definition,
            {InputTypes.ID, InputTypes.TIME},
        )
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL,
            column_definition,
            {InputTypes.ID, InputTypes.TIME},
        )

        # Transform real inputs per entity
        transformed_parts = []
        for identifier, sliced in df.groupby(id_column):
             # Skip entities that did not get a scaler during training calibration
            if identifier not in self._real_scalers:
                continue
            # Skip sequences that are too short
            if len(sliced) < self._time_steps:
                continue

            sliced_copy = sliced.copy()
            sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
                sliced_copy[real_inputs].values
            )

            transformed_parts.append(sliced_copy)
        
        if not transformed_parts:
            raise ValueError("No valid entity slices remained after transformation.")
        
        output = pd.concat(transformed_parts, axis=0).reset_index(drop=True)

        # Format categorical inputs
        for col in categorical_inputs:
            output[col] = self._cat_scalers[col].transform(output[col].astype(str))

        return output
    
    def format_predictions(self, predictions):
        """ Converts normalized predictions back to the original scale """

        if self._target_scaler is None:
            raise ValueError("Target scalers have not been set.")
        
        restored_parts = []

        for identifier, sliced in predictions.groupby("id"):
            if identifier not in self._target_scaler:
                raise ValueError(f"No target scaler found for identifier: {identifier}")

            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]

            for col in sliced_copy.columns:
                if col not in {"id", "forecast_time"}:
                    # Turn into a 2D shape
                    sliced_copy[col] = target_scaler.inverse_transform(
                        sliced_copy[[col]]
                    )

            restored_parts.append(sliced_copy)

        output = pd.concat(restored_parts, axis=0).reset_index(drop=True)
        return output


















