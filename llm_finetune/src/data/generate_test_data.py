import datetime
import random

import click
import pandas as pd


def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())),
    )


def date_to_string(date, pattern="%m.%d.%Y"):
    return date.strftime(pattern)


@click.command()
@click.option(
    "-o", "--output-path", type=click.Path(), help="Path to save generated data"
)
def generate_test_data(output_path: str):
    random.seed(42)

    NUM_DATES = 5
    START_DATE = datetime.datetime(year=2020, month=1, day=1)
    END_DATE = datetime.datetime(year=2023, month=12, day=29)
    cities = [
        "Moscow",
        "Novosibirsk",
        "Saint-Petersburg",
        "Yekaterinburg",
    ]

    data = []

    for city_from in cities:
        for city_to in cities:
            if city_to == city_from:
                continue
            for i in range(NUM_DATES):
                date = random_date(START_DATE, END_DATE)
                date_str = date_to_string(date)
                date_back = date_to_string(random_date(date, END_DATE))
                for lug in [True, False]:
                    data.append([city_from, city_to, date_str, date_back, lug])
                    data.append([city_from, city_to, date_str, date_back, None])
                    data.append([city_from, city_to, date_str, None, lug])
                    data.append([city_from, city_to, date_str, None, None])
                    data.append(
                        [city_from, None, date_str, None, None]
                    )  # error in Pydantic
                    data.append(
                        [city_from, city_to, None, None, None]
                    )  # error in Pydantic

    df = pd.DataFrame(data)
    df.columns = ["city_from", "city_to", "date", "date_back", "need_luggage"]
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    generate_test_data()
