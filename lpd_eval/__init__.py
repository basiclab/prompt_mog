"""LPD-Eval: Evaluation Toolkit for Long-Prompt-Diversity Benchmark (LPD-Bench)"""

from lpd_eval.eval import eval_
from lpd_eval.print_score import print_score

__version__ = "0.1.0"
__all__ = ["eval_", "print_score", "__version__"]
