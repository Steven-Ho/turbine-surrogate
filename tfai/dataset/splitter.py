import math
import os
from typing import Tuple

# Imports `pandas` for reuse in this module.
import pandas as pd
from sklearn.model_selection import train_test_split


class Splitter:
    """数据集划分器。

    提供两类基础能力：
    - 将样本表划分为训练集和验证集
    - 将样本表划分为训练集和测试集

    兼容保留 `split_dataset`，用于一次性划分训练集、验证集和测试集。
    """

    @staticmethod
    # Defines `_validate_ratio` for  validate ratio logic.
    def _validate_ratio(name: str, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")

    @staticmethod
    # Defines `_write_split_csvs` for  write split csvs logic.
    def _write_split_csvs(
        output_dir: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        # Continues the surrounding statement or collection definition.
        test_df: pd.DataFrame,
    ) -> Tuple[str, str, str]:
        train_csv = os.path.join(output_dir, "train.csv")
        valid_csv = os.path.join(output_dir, "valid.csv")
        # Stores the `test_csv` for use by the remaining steps in this flow.
        test_csv = os.path.join(output_dir, "test.csv")

        train_df.to_csv(train_csv, index=False)
        valid_df.to_csv(valid_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        # Returns `train_csv, valid_csv, test_csv` to the caller for the next stage.
        return train_csv, valid_csv, test_csv

    @staticmethod
    def _write_pair_csvs(
        first_df: pd.DataFrame,
        # Continues the surrounding statement or collection definition.
        second_df: pd.DataFrame,
        first_csv: str,
        second_csv: str,
    ) -> Tuple[str, str]:
        # Stores the `first_dir` for use by the remaining steps in this flow.
        first_dir = os.path.dirname(first_csv)
        second_dir = os.path.dirname(second_csv)
        if first_dir:
            os.makedirs(first_dir, exist_ok=True)
        # Checks `second_dir` before entering this conditional branch.
        if second_dir:
            os.makedirs(second_dir, exist_ok=True)
        first_df.to_csv(first_csv, index=False)
        second_df.to_csv(second_csv, index=False)
        # Returns `first_csv, second_csv` to the caller for the next stage.
        return first_csv, second_csv

    @staticmethod
    def _load_samples(samples_csv: str) -> pd.DataFrame:
        return pd.read_csv(samples_csv)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _resolve_split_count(sample_count: int, ratio: float, name: str) -> int:
        Splitter._validate_ratio(name, ratio)
        return max(1, math.ceil(sample_count * ratio))

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def split_train_valid(
        samples_csv: str,
        output_dir: str,
        # Continues the surrounding statement or collection definition.
        valid_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple[str, str]:
        """划分 `samples.csv` 为训练集和验证集。"""
        os.makedirs(output_dir, exist_ok=True)
        # Stores the `df` for use by the remaining steps in this flow.
        df = Splitter._load_samples(samples_csv)
        train_csv = os.path.join(output_dir, "train.csv")
        valid_csv = os.path.join(output_dir, "valid.csv")

        if len(df) < 2:
            # Returns `Splitter._write_pair_csvs(df, df, train_csv, valid_csv)` to the caller for the next stage.
            return Splitter._write_pair_csvs(df, df, train_csv, valid_csv)

        sample_count = len(df)
        valid_count = Splitter._resolve_split_count(sample_count, valid_size, "valid_size")
        if valid_count >= sample_count:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"valid_size={valid_size} is too large for {sample_count} samples; "
                "at least 1 sample must remain in train"
            )

        # Stores the `(train_df, valid_df)` for use by the remaining steps in this flow.
        train_df, valid_df = train_test_split(
            df,
            test_size=valid_count,
            random_state=seed,
        # Completes the multi-line `train_test_split` arguments or elements.
        )
        return Splitter._write_pair_csvs(train_df, valid_df, train_csv, valid_csv)

    @staticmethod
    def split_train_test(
        # Continues the surrounding statement or collection definition.
        samples_csv: str,
        output_dir: str,
        test_size: float = 0.1,
        seed: int = 42,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[str, str]:
        """划分 `samples.csv` 为训练集和测试集。"""
        os.makedirs(output_dir, exist_ok=True)
        df = Splitter._load_samples(samples_csv)
        train_csv = os.path.join(output_dir, "train.csv")
        # Stores the `test_csv` for use by the remaining steps in this flow.
        test_csv = os.path.join(output_dir, "test.csv")

        if len(df) < 2:
            return Splitter._write_pair_csvs(df, df, train_csv, test_csv)

        sample_count = len(df)
        # Stores the `test_count` for use by the remaining steps in this flow.
        test_count = Splitter._resolve_split_count(sample_count, test_size, "test_size")
        if test_count >= sample_count:
            raise ValueError(
                f"test_size={test_size} is too large for {sample_count} samples; "
                # Completes the multi-line `ValueError` arguments or elements.
                "at least 1 sample must remain in train"
            )

        train_df, test_df = train_test_split(
            df,
            # Completes the multi-line `train_test_split` arguments or elements.
            test_size=test_count,
            random_state=seed,
        )
        return Splitter._write_pair_csvs(train_df, test_df, train_csv, test_csv)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def split_dataset(
        samples_csv: str,
        output_dir: str,
        # Continues the surrounding statement or collection definition.
        test_size: float = 0.1,
        valid_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple[str, str, str]:
        """划分 `samples.csv` 为训练集、验证集和测试集。

        Args:
            samples_csv: 样本表路径。
            output_dir: 划分结果输出目录。
            test_size: 测试集比例，相对于全量样本。
            valid_size: 验证集比例，相对于全量样本。
            seed: 随机种子。

        Returns:
            `(train_csv, valid_csv, test_csv)` 三个文件路径。
        """
        # Calls `os.makedirs` to perform the operation required at this stage.
        os.makedirs(output_dir, exist_ok=True)
        df = Splitter._load_samples(samples_csv)

        if len(df) < 3:
            # 样本过少时，保守地让三份切分都指向同一批样本，避免后续流程因空集失败。
            return Splitter._write_split_csvs(output_dir, df, df, df)

        # Stores the `sample_count` for use by the remaining steps in this flow.
        sample_count = len(df)
        if test_size + valid_size >= 1:
            raise ValueError(
                "test_size + valid_size must be less than 1 so that the train split remains non-empty"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        test_count = Splitter._resolve_split_count(sample_count, test_size, "test_size")
        valid_count = Splitter._resolve_split_count(sample_count, valid_size, "valid_size")
        if test_count + valid_count >= sample_count:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"test_size={test_size} and valid_size={valid_size} are too large for "
                f"{sample_count} samples; at least 1 sample must remain in train"
            )

        # Stores the `(train_val_df, test_df)` for use by the remaining steps in this flow.
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_count,
            random_state=seed,
        # Completes the multi-line `train_test_split` arguments or elements.
        )

        train_df, valid_df = train_test_split(
            train_val_df,
            test_size=valid_count,
            # Completes the multi-line `train_test_split` arguments or elements.
            random_state=seed,
        )

        return Splitter._write_split_csvs(output_dir, train_df, valid_df, test_df)
