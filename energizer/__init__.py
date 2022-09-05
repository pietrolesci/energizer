"""Top-level package for Pytorch-Energizer."""

from energizer.data.datamodule import ActiveDataModule
from energizer.query_strategies.base import AccumulatorStrategy, NoAccumulatorStrategy, MCAccumulatorStrategy
from energizer.trainer import Trainer

__author__ = """Pietro Lesci"""
__email__ = 'pietrolesci@outlook.com'
__version__ = '0.1.0'
__all__ = ["AccumulatorStrategy", "NoAccumulatorStrategy", "Trainer", "ActiveDataModule", "MCAccumulatorStrategy"]
