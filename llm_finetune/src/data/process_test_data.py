import json

import numpy as np
import pandas as pd


def get_test_csv_data(path: str):
    df = pd.read_csv(
        path,
    )
    df = df.fillna(np.nan).replace([np.nan], [None])

    test_data = []
    for i, row in df.iterrows():
        city_from, city_to = row["city_from"], row["city_to"]
        date, date_back = row["date"], row["date_back"]
        need_luggage = (
            None
            if row["need_luggage"] is None
            else True
            if str(row["need_luggage"]) == "True"
            else False
        )
        d = {
            "city_from": city_from,
            "city_to": city_to,
            "date": date,
            "date_back": date_back,
            "need_luggage": need_luggage,
        }
        if row["zephyr"]:
            question = row["zephyr_question"]
            data = {"question": question, "answer_dict": d}
            test_data.append(data)
        if row["llama"]:
            question = row["llama_question"]
            data = {"question": question, "answer_dict": d}
            test_data.append(data)
    return test_data


def main():
    flights_q_path = "../../data/test/raw/test_with_questions_fixed_part.csv"
    general_json = "../../data/test/raw/random_questions.json"
    flights_general_json = "../../data/test/raw/plains_flights_questions.json"
    output_path = "../../data/test/final/function_calling_test.json"
    test_main_data = get_test_csv_data(flights_q_path)

    with open(general_json, "r") as f:
        data = json.load(f)
    general_data = [
        {"question": q, "answer_dict": None} for q in data if len(q.strip()) > 0
    ]

    with open(flights_general_json, "r") as f:
        data = json.load(f)
    flights_general_data = [
        {"question": q, "answer_dict": None} for q in data if len(q.strip()) > 0
    ]
    print(len(test_main_data), len(general_data), len(flights_general_data))

    test_main_data.extend(general_data)
    test_main_data.extend(flights_general_data)

    for i, item in enumerate(test_main_data):
        item["index"] = i

    with open(output_path, "w") as f:
        json.dump(test_main_data, f)


if __name__ == "__main__":
    main()
