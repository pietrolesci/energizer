"""Top-level package for Pytorch-Energizer."""

from energizer.data.datamodule import ActiveDataModule
from energizer.trainer import Trainer

__author__ = """Pietro Lesci"""
__email__ = 'pietrolesci@outlook.com'
__version__ = '0.2.0'
__all__ = ["Trainer", "ActiveDataModule"]
