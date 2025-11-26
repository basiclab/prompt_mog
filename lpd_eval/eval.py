import os
from typing import Literal

from accelerate import Accelerator
from tyro import conf

from lpd_eval.eval_diversity import eval_diversity
from lpd_eval.eval_semantics import eval_semantics, setup_logging

# avoid the warning of tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def eval_(
    gen_root_dir: str,
    dataset_name: str = "Justin900/LPD-Bench",
    partial_num: int | None = None,
    dtype: Literal["none", "fp16", "bf16"] = "bf16",
    num_workers: int = 4,
    overwrite: bool = False,
    batch_size: conf.Fixed[int] = 1,
):
    setup_logging()
    accelerator = Accelerator(mixed_precision=dtype)
    eval_semantics(
        accelerator=accelerator,
        gen_root_dir=gen_root_dir,
        dataset_name=dataset_name,
        partial_num=partial_num,
        dtype=dtype,
        num_workers=num_workers,
        overwrite=overwrite,
        batch_size=batch_size,
    )
    eval_diversity(
        accelerator=accelerator,
        gen_root_dir=gen_root_dir,
        dtype=dtype,
        num_workers=num_workers,
        overwrite=overwrite,
        batch_size=batch_size,
    )
    accelerator.end_training()
