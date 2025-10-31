import pandas as pd
import subprocess
import os
from typing import Optional, Union
from itertools import tee, islice
class AutoTabularLoader:
    common_delimiters = [',', '|', '\t', ';']

    def __init__(self, file_path: str, chunk_threshold: int = 1_000_000, chunksize: int = 100_000, sample_lines: int = 100):
        """
        Args:
            file_path: Path to the dataset.
            chunk_threshold: Number of rows above which data will be loaded in chunks.
            chunksize: Number of rows per chunk if chunked.
            sample_lines: Lines to sample for delimiter detection.
        Note: This assumes there are at least 1000 rows 
        """
        self.file_path = file_path
        self.chunk_threshold = chunk_threshold
        self.chunksize = chunksize
        self.sample_lines = sample_lines

        # Auto-detected / derived
        self.delimiter: Optional[str] = None
        self.num_rows: Optional[int] = None
        self.num_columns: Optional[int] = None
        self.col_names: Optional[list] = None
        self.numeric_cols: list = []
        self.categorical_cols: list = []

        # Data holder (DataFrame or TextFileReader)
        self.df: Optional[Union[pd.DataFrame, pd.io.parsers.TextFileReader]] = None

        # Run auto-load pipeline
        self._auto_load()


    def print_info(self):
        print("=== AutoTabularLoader Info ===")
        print(f"File path       : {self.file_path}")
        print(f"Detected delimiter : {self.delimiter}")
        print(f"Number of rows  : {self.num_rows}")
        print(f"Number of columns : {self.num_columns}")
        print(f"Column names    : {self.col_names}")
        print(f"Numeric columns : {self.numeric_cols}")
        print(f"Categorical columns : {self.categorical_cols}")
        print(f"Chunksize       : {getattr(self, 'chunksize', 'N/A')}")
        if isinstance(self.df, pd.io.parsers.TextFileReader):
            print("Data type       : Chunked iterator")
        elif isinstance(self.df, pd.DataFrame):
            print("Data type       : Full DataFrame")
        else:
            print("Data type       : Not loaded")
        print("==============================")
    def apply(self, fn):
        """
        Apply a function to the dataset in a non-destructive way.
        Returns a list of results.
        """
        if isinstance(self.df, pd.DataFrame):
            return fn(self.df)
        else:
            iter_copy, __ = tee(self.df)
            results = []
            for chunk in iter_copy:
                results.append(fn(chunk.copy()))
            return results

    # -----------------------------
    # Main pipeline
    def _auto_load(self):
        self._detect_delimiter()
        self._count_rows()
        self._count_columns()
        self._load_dataframe()
        self._identify_column_types()

    # Detect delimiter
    def _detect_delimiter(self):
        max_lines = min(self.sample_lines, 500)
        with open(self.file_path, 'r', encoding='utf-8') as f:
            sample_lines = [next(f) for _ in range(max_lines)]
    
        for d in self.common_delimiters:
            # Use first line as reference
            expected_count = sample_lines[0].count(d)
            print(d, expected_count)
            if expected_count==0:
                continue
            consistent = True

            for line in sample_lines[1:]:
                if line.count(d) != expected_count:
                    consistent = False
                    break  # stop immediately if inconsistent

            if consistent:
                self.delimiter = d
                return d
        self.delimiter = None
        raise ValueError("Could not detect delimiter. Please set manually using set_delimiter().")

    def set_delimiter(self, delimiter: str):
        self.delimiter = delimiter

    # Count rows using wc
    def _count_rows(self):
        try:
            result = subprocess.run(['wc', '-l', self.file_path], capture_output=True, text=True, check=True)
            self.num_rows = int(result.stdout.strip().split()[0])
        except Exception as e:
            print(f"Error counting rows using wc: {e}")
            self.num_rows = None

    # Count columns
    def _count_columns(self):
        if self.delimiter is None:
            raise ValueError("Delimiter not set.")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            self.col_names = first_line.split(self.delimiter)
            self.num_columns = len(self.col_names)

    # Load dataframe (full or chunked)
    def _load_dataframe(self):
        if self.delimiter is None:
            raise ValueError("Delimiter not set.")

        if self.num_rows is None:
            self._count_rows()

        if self.num_rows is not None and self.num_rows > self.chunk_threshold:
            self.df = pd.read_csv(
                self.file_path,
                delimiter=self.delimiter,
                chunksize=self.chunksize,
                names=self.col_names,
                iterator=True,
                header=0
            )
        else:
            self.df = pd.read_csv(
                self.file_path,
                delimiter=self.delimiter,
                names=self.col_names,
                header=0
            )

    # Identify numeric vs categorical columns
    def _identify_column_types(self, samplerows=100):
        
        df_sample = None

        if self.df is None:
            raise ValueError("Data not loaded.")

        if isinstance(self.df, pd.io.parsers.TextFileReader):
            # chunked: take first chunk
            chunks = self._peek_chunk(n_chunks=1)
            df_sample = chunks[0].head(samplerows)
        else:
            # full DataFrame: sample top rows
            df_sample = self.df.head(samplerows)

        self.numeric_cols = df_sample.select_dtypes(include='number').columns.tolist()
        self.categorical_cols = df_sample.select_dtypes(include='object').columns.tolist()


    # -----------------------------
    # Magic method to forward calls to underlying pandas DataFrame
    def __getattr__(self, name):
        """
        Forward any attribute/method calls not defined in this class to self.df.
        Works for both full DataFrame and chunked TextFileReader.
        """
        if self.df is None:
            raise AttributeError(f"'AutoTabularLoader' has no attribute '{name}' and data is not loaded yet.")

        # If _df is a chunked iterator, always use the first chunk for attribute access
        df_target = self.df
        if isinstance(self.df, pd.io.parsers.TextFileReader):
            chunk_copy, __ = tee(self.df)
            df_target = next(chunk_copy)
        attr = getattr(df_target, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)
            return wrapper
        else:
            return attr
    
    def __getitem__(self, key):
        if self.df is None:
            raise AttributeError(f"'AutoTabularLoader' has no attribute '{name}' and data is not loaded yet.")

        if isinstance(self.df, pd.io.parsers.TextFileReader):
            df_target = self._peek_chunk(1)[0]
        else:
            df_target = self.df

        if isinstance(key, (list, tuple)):
            missing = [k for k in key if k not in df_target.columns]
            if missing:
                raise KeyError(f"Columns {missing} not found in the dataset.")
            return df_target[list(key)]  # Return as DataFrame subset
        # Single column access
        if key not in df_target.columns:
            raise KeyError(f"Column '{key}' not found in the dataset.")
        return df_target[key]
    #Utility Function for peaking:
    def _peek_chunk(self, n_chunks=1):
        """
        Peek first n_chunks of a chunked TextFileReader without consuming the original iterator.
        Returns a list of DataFrames (chunks).
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        if not isinstance(self.df, pd.io.parsers.TextFileReader):
            return [self.df]  # not chunked

        # Duplicate iterator
        peek_iter, _ = tee(self.df)
        chunks = list(islice(peek_iter, n_chunks))
        return chunks

