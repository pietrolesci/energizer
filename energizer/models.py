from abc import ABC, abstractmethod
from pathlib import Path

import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from peft import get_peft_model  # type: ignore
from peft.config import PeftConfig
from peft.peft_model import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# from transformers.generation import GenerationConfig
# from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils import PreTrainedTokenizerBase

from energizer.utilities import local_seed
from energizer.utilities.model_summary import summarize


class Model(ABC):
    _model_instance: torch.nn.Module | None = None

    @property
    def model_instance(self) -> torch.nn.Module:
        assert self._model_instance is not None, RuntimeError("Model needs to be initialised by the Estimator")
        return self._model_instance

    @property
    def summary(self) -> str:
        return summarize(self.model_instance)

    @abstractmethod
    def configure_model(self, *args, **kwargs) -> None:
        ...


class TorchModel(Model):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model_instance = model

    def configure_model(self, *args, **kwargs) -> None:
        pass


class HFModel(Model):
    """This class allows to load any Huggingface model and adapters."""

    AUTO_CONFIG_CLASS: type[AutoConfig] = AutoConfig
    AUTO_MODEL_CLASS: type[AutoModel] = AutoModel
    AUTO_TOKENIZER_CLASS: type[AutoTokenizer] = AutoTokenizer

    _model: PreTrainedModel | PeftModel
    _config: PretrainedConfig | None = None
    _tokenizer: PreTrainedTokenizerBase | None = None
    _model_kwargs: dict | None = None

    def __init__(
        self,
        model_name_or_config: str | PretrainedConfig,
        revision: str | None = None,
        subfolder: str | None = None,
        adapters: dict[str, PeftConfig] | None = None,
        seed: int = 42,
        cache_dir: str | Path | None = None,
        convert_to_bettertransformer: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(model_name_or_config, PretrainedConfig):
            rank_zero_info(
                "You passed a `PretrainedConfig` which means the model will be re-initialised from scratch."
                "The tokenizer is not created automatically in this case: remember to `attach_tokenizer`."
            )
            self._config = model_name_or_config
        else:
            rank_zero_info(
                "You passed a model name (`str`) which means the model will be initialised from "
                f"the {model_name_or_config} pre-trained checkpoint."
            )
            if revision is not None and subfolder is not None:
                revision += "/" + subfolder
            self._model_kwargs = {
                "pretrained_model_name_or_path": model_name_or_config,
                "revision": revision,
                "cache_dir": cache_dir,
            }

        self._adapters = adapters
        self._seed = seed
        self._convert_to_bettertransformer = convert_to_bettertransformer

    @property
    def config(self) -> PretrainedConfig | None:
        return self._config

    @property
    def model_kwargs(self) -> dict | None:
        return self._model_kwargs

    @property
    def adapters(self) -> dict[str, PeftConfig] | None:
        return self._adapters

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        return self._tokenizer

    def attach_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # NOTE: we do not set the tokenizer by default
        self._tokenizer = tokenizer

    @property
    def summary(self) -> str:
        max_depth = 1 if self.adapters is None else 2
        rank_zero_info(
            f"Total num params: {self._model.num_parameters(only_trainable=False) / 1e6:.01f}M\n"
            f"Of which trainable: {self._model.num_parameters(only_trainable=True) / 1e6:.01f}M\n"
            f"With a memory footprint of {self._model.get_memory_footprint() / 1e9:.02f}GB\n"
            f"Total memory allocated {torch.cuda.max_memory_allocated() / 1e9:.02f}GB"
        )
        return summarize(self, max_depth)

    def configure_model(self) -> None:
        with local_seed(self._seed):
            if self.model_kwargs is None:
                self._model_instance = self.AUTO_MODEL_CLASS.from_config(self.config)
            else:
                self._model_instance = self.AUTO_MODEL_CLASS.from_pretrained(**self.model_kwargs)
                self._config = self.AUTO_CONFIG_CLASS.from_pretrained(**self.model_kwargs)
                self._tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
                    self.model_kwargs["pretrained_model_name_or_path"], revision=self.model_kwargs["revision"]
                )

            if self.adapters is not None:
                for adapter_name, peft_config in self.adapters.items():
                    if isinstance(self._model_instance, PeftModel):
                        self._model_instance.add_adapter(adapter_name=adapter_name, peft_config=peft_config)
                    else:
                        self._model_instance = get_peft_model(
                            self._model_instance,  # type: ignore
                            adapter_name=adapter_name,
                            peft_config=peft_config,
                        )

                    rank_zero_info(f"Adapter `{adapter_name}` loaded successfully")

            if self._convert_to_bettertransformer:
                self.convert_to_bettertransformer()

    def convert_to_bettertransformer(self) -> None:
        assert self._model_instance is not None
        self._model_instance = self._model_instance.to_bettertransformer()

    def reverse_bettertransformer(self) -> None:
        assert self._model_instance is not None
        self._model_instance = self._model_instance.reverse_bettertransformer()


class HFModelForSequenceClassification(HFModel):
    """Instantiates language models with Classification head"""

    AUTO_MODEL_CLASS: type[AutoModelForSequenceClassification] = AutoModelForSequenceClassification

    def __init__(
        self,
        model_name_or_config: str | PretrainedConfig,
        num_classes: int,
        revision: str | None = None,
        subfolder: str | None = None,
        adapters: dict[str, PeftConfig] | None = None,
        seed: int = 42,
        cache_dir: str | Path | None = None,
        convert_to_bettertransformer: bool = False,
    ) -> None:
        super().__init__(
            model_name_or_config, revision, subfolder, adapters, seed, cache_dir, convert_to_bettertransformer
        )
        if self._model_kwargs is not None:
            self._model_kwargs["num_labels"] = num_classes


# class HFModelForGeneration(HFModel):
#     _generation_config: GenerationConfig | None = None

#     @property
#     def generation_config(self) -> GenerationConfig | None:
#         return self._generation_config

#     def prepare_for_generation(self, generation_config: dict | GenerationConfig) -> None:
#         """Use this method to set the default generation configuration.

#         You can override by passing `generation_config` to the `generation_step` method.
#         """
#         if isinstance(generation_config, dict):
#             generation_config = GenerationConfig(**generation_config)
#         self._generation_config = generation_config

#     def generate(
#         self, inputs: torch.Tensor, generation_config: GenerationConfig | None = None, **kwargs
#     ) -> GenerateOutput | torch.LongTensor:
#         generation_config = generation_config or self.generation_config
#         return self.model_instance.generate(inputs, generation_config=generation_config, **kwargs)

#     def generate_from_texts(
#         self,
#         texts: str | list[str],
#         generation_config: GenerationConfig | None = None,
#         skip_special_tokens: bool = True,
#         **kwargs,
#     ) -> list[str]:
#         assert self.tokenizer is not None, "To `generate_from_texts` you need to `attach_tokenizer`."
#         if isinstance(texts, str):
#             texts = [texts]

#         inputs = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=False,
#             return_token_type_ids=False,
#         ).input_ids

#         return self.generate_from_ids(inputs, generation_config, skip_special_tokens, **kwargs)

#     def generate_from_ids(
#         self,
#         input_ids: torch.LongTensor,
#         generation_config: GenerationConfig | None = None,
#         skip_special_tokens: bool = True,
#         **kwargs,
#     ) -> list[str]:
#         assert self.tokenizer is not None, "To `generate_from_texts` you need to `attach_tokenizer`."
#         generated_ids = self.generate(input_ids, generation_config, **kwargs)
#         generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
#         return generated_texts


# class HFModelForCausalLM(HFModelForGeneration):
#     """Instantiates language models with LM head"""

#     # NOTE: for cause language models AutoModelForCausalLM == AutoModelForPreTraining
#     AUTO_MODEL_CLASS: type[AutoModelForCausalLM] = AutoModelForCausalLM

#     def tokenizer_set_pad_to_eos(self) -> None:
#         if self.tokenizer is not None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         rank_zero_info("No tokenizer set. Doing nothing.")
