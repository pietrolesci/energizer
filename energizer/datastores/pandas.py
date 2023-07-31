class PandasDataStoreForSequenceClassification(PandasDataStoreWithIndex):
    _tokenizer: Optional[PreTrainedTokenizerBase]
    _labels: List[str]
    _label_distribution: Dict[str, int]

    def prepare_for_loading(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        replacement: bool = False,
        max_length: int = 512,
    ) -> None:
        super().prepare_for_loading(
            batch_size,
            eval_batch_size,
            num_workers,
            pin_memory,
            drop_last,
            persistent_workers,
            shuffle,
            seed,
            replacement,
        )
        self.max_length = max_length

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._tokenizer

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    def label_distribution(self, normalized: bool = False) -> Dict[str, Union[float, int]]:
        if normalized:
            total = sum(self._label_distribution.values())
            return {k: self._label_distribution[k] / total for k in self._label_distribution}
        return dict(self._label_distribution)

    def from_datasets(
        self,
        input_names: Union[str, List[str]],
        target_name: str,
        train_dataset: Dataset,
        _validation_dataset: Optional[Dataset] = None,
        _test_dataset: Optional[Dataset] = None,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._labels = train_dataset.features[target_name].names
        self._label_distribution = Counter(train_dataset[target_name])
        return super().from_datasets(
            train_dataset=train_dataset,
            _validation_dataset=_validation_dataset,
            _test_dataset=_test_dataset,
            input_names=input_names,
            target_name=target_name,
            uid_name=uid_name,
            on_cpu=on_cpu,
        )

    def from_dataset_dict(
        self,
        dataset_dict: DatasetDict,
        input_names: Union[str, List[str]],
        target_name: str,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.from_datasets(
            train_dataset=dataset_dict[RunningStage.TRAIN],
            _validation_dataset=dataset_dict.get(RunningStage.VALIDATION, None),
            _test_dataset=dataset_dict.get(RunningStage.TEST, None),
            input_names=input_names,
            target_name=target_name,
            uid_name=uid_name,
            on_cpu=on_cpu,
            tokenizer=tokenizer,
        )

    def get_collate_fn(self, stage: Optional[RunningStage] = None) -> Optional[Callable]:
        on_cpu = self.on_cpu if self.on_cpu is not None else []
        return partial(
            collate_fn,
            input_names=self.input_names,
            target_name=self.target_name,
            on_cpu=on_cpu + [SpecialKeys.ID],
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            pad_fn=_pad,
        )

    def save(self, dir: Union[str, Path]) -> None:
        dir = Path(dir)

        datasets = {
            RunningStage.TRAIN: Dataset.from_pandas(self._train_data, preserve_index=False, features=self._features)
        }
        if self._validation_data:
            datasets[RunningStage.VALIDATION] = self._validation_data
        if self._test_data:
            datasets[RunningStage.TEST] = self._test_data

        for split, dataset in datasets.items():
            dataset.save_to_disk(dir / split)

        meta = {
            "input_names": self.input_names,
            "target_name": self.target_name,
            "on_cpu": self.on_cpu,
            "name_or_path": self.tokenizer.name_or_path if self.tokenizer else None,
            "seed": self.seed,
        }
        srsly.write_json(dir / "metadata.json", meta)
        self.save_index(dir)

    @classmethod
    def load(cls, dir: Union[str, Path]) -> "PandasDataStoreForSequenceClassification":
        dir = Path(dir)
        datasets = {split: load_from_disk(dir / split) for split in RunningStage if (dir / split).exists()}
        meta: Dict = srsly.read_json(dir / "metadata.json")  # type: ignore
        tokenizer = None
        if meta["name_or_path"] is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(meta["name_or_path"])

        out = cls(meta["seed"])
        out.from_datasets(
            train_dataset=datasets.get(RunningStage.TRAIN, None),  # type: ignore
            _validation_dataset=datasets.get(RunningStage.VALIDATION, None),  # type: ignore
            _test_dataset=datasets.get(RunningStage.TEST, None),  # type: ignore
            input_names=meta["input_names"],
            target_name=meta["target_name"],
            on_cpu=meta["on_cpu"],
            tokenizer=tokenizer,
        )
        out.load_index(dir)

        return out


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    input_names: List[str],
    target_name: str,
    on_cpu: List[str],
    max_length: Optional[int],
    pad_token_id: Optional[int],
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:
    new_batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    values_on_cpu = {col: new_batch.pop(col, None) for col in on_cpu if col in new_batch}

    labels = new_batch.pop(target_name, None)

    # input_ids and attention_mask to tensor: truncate -> convert to tensor -> pad
    new_batch = {
        k: pad_fn(
            inputs=new_batch[k],
            padding_value=pad_token_id,
            max_length=max_length,
        )
        for k in input_names
    }

    if labels is not None:
        new_batch[target_name] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch
