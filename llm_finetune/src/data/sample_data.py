import json

import click
import numpy as np


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True),
    help="Path to messages data in json format",
)
@click.option(
    "-o", "--output-path", type=click.Path(), help="Path to save sampled json dataset"
)
@click.option(
    "--sample_n_fc",
    type=click.INT,
    default=1300,
    help="Number of messages with Function calls to save",
)
@click.option(
    "--sample_n_no_fc",
    type=click.INT,
    default=700,
    help="Number of messages without Function calls to save",
)
@click.option("--seed", type=click.INT, default=12345, help="Random seed")
def sample_data(
    input_path: str,
    output_path: str,
    sample_n_fc: int = 700,
    sample_n_no_fc: int = 300,
    seed=12345,
):
    with open(input_path, "rb") as f:
        data = json.load(f)

    def contains_fc(msg):
        for m in msg:
            if m["role"] == "function":
                return True
        return False

    bool_fc = np.array([contains_fc(x) for x in data])

    rng = np.random.default_rng(seed)
    sampled_ind_fc = rng.choice(sum(bool_fc), size=sample_n_fc, replace=False)
    sampled_ind_no_fc = rng.choice(
        len(data) - sum(bool_fc), size=sample_n_no_fc, replace=False
    )

    sampled_fc = set(np.arange(len(data))[bool_fc][sampled_ind_fc])
    sampled_no_fc = set(np.arange(len(data))[~bool_fc][sampled_ind_no_fc])
    data_sampled = [
        x for i, x in enumerate(data) if i in set(sampled_fc) or i in set(sampled_no_fc)
    ]

    with open(output_path, "w") as f:
        json.dump(data_sampled, f)


if __name__ == "__main__":
    sample_data()
