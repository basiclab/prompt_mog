"""Command-line interface for lpd_eval."""

import os

import tyro

from lpd_eval.eval import eval_
from lpd_eval.print_score import print_score


def cli_print():
    tyro.cli(print_score)


def cli():
    rank = int(os.environ.get("RANK", 0))
    tyro.cli(eval_, console_outputs=rank == 0)


if __name__ == "__main__":
    cli()
