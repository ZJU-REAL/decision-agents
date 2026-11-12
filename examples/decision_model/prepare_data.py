"""Create local batch placeholders; actual tasks are sampled by the environments."""
import argparse
from pathlib import Path


def prepare(output, train_size=16, val_size=128):
    from datasets import Dataset

    if train_size < 1 or val_size < 1:
        raise ValueError("batch sizes must be positive")
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    for split, size in (("train", train_size), ("test", val_size)):
        rows = [{"data_source": "text", "prompt": [{"role": "user", "content": ""}],
                 "ability": "agent", "extra_info": {"split": split, "index": index}}
                for index in range(size)]
        Dataset.from_list(rows).to_parquet(str(output / (split + ".parquet")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=128)
    args = parser.parse_args()
    prepare(args.output, args.train_size, args.val_size)
