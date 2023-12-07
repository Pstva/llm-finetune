# for item in test_data:
#     user_question = item["question"]
#     index = item["index"]

import json

import click
import pandas as pd
import yaml


def parse_answer_json(text: str) -> tuple[bool, dict | None]:
    try:
        output_dict = yaml.safe_load(text)
        if isinstance(output_dict, dict):
            return True, output_dict
        return False, None
    except:
        return False, None


def parse_argumets(text: str) -> tuple[bool, dict | None]:
    try:
        output_dict = yaml.safe_load(text)
        if isinstance(output_dict, dict):
            return True, output_dict
        return False, None
    except:
        return False, None


def allign_results(test_data_path: str, model_outputs_path: str, output_path: str):
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    with open(model_outputs_path, "r") as f:
        model_outputs = json.load(f)

    result_data = {
        "question": [],
        "is_fc": [],
        "city_from": [],
        "city_to": [],
        "date": [],
        "date_back": [],
        "need_luggage": [],
        "answer": [],
        "is_fc_answer": [],
        "parsed_fc": [],
        "function_name": [],
        "parsed_args": [],
        "parsed_city_from": [],
        "parsed_city_to": [],
        "parsed_date": [],
        "parsed_date_back": [],
        "parsed_need_luggage": [],
        "all_parsed_arguments": [],
    }

    for data, output in zip(test_data[: len(model_outputs)], model_outputs):
        assert data["index"] == output["index"]
        question, true_answer_dict = data["question"], data["answer_dict"]
        # заполняем инфо о Ground truth
        result_data["question"].append(question)
        is_fc = True if true_answer_dict is not None else False
        result_data["is_fc"].append(is_fc)
        if is_fc:
            result_data["city_from"].append(true_answer_dict["city_from"])
            result_data["city_to"].append(true_answer_dict["city_to"])
            result_data["date"].append(true_answer_dict["date"])
            result_data["date_back"].append(true_answer_dict["date_back"])
            result_data["need_luggage"].append(true_answer_dict["need_luggage"])
        else:
            result_data["city_from"].append(None)
            result_data["city_to"].append(None)
            result_data["date"].append(None)
            result_data["date_back"].append(None)
            result_data["need_luggage"].append(None)

        # заполняем инфо о predictions
        pred_is_functioncall, answer_text = (
            output["answer"]["functioncall"],
            output["answer"]["text"],
        )
        result_data["answer"].append(answer_text)
        result_data["is_fc_answer"].append(pred_is_functioncall)

        if pred_is_functioncall:
            parsed, parsed_dict = parse_answer_json(answer_text)
            if parsed:
                result_data["parsed_fc"].append(True)
                result_data["function_name"].append(
                    parsed_dict["name"] if "name" in parsed_dict else None
                )
                parsed_args = False
                if "arguments" in parsed_dict:
                    if isinstance(parsed_dict["arguments"], str):
                        parsed_args, args = parse_argumets(parsed_dict["arguments"])
                    elif isinstance(parsed_dict["arguments"], dict):
                        parsed_args = True
                        args = parsed_dict["arguments"]
                    else:
                        parsed_args = False
                if parsed_args is False:
                    result_data["parsed_args"].append(False)
                    result_data["parsed_city_from"].append(None)
                    result_data["parsed_city_to"].append(None)
                    result_data["parsed_date"].append(None)
                    result_data["parsed_date_back"].append(None)
                    result_data["parsed_need_luggage"].append(None)
                    result_data["all_parsed_arguments"].append(None)
                else:
                    result_data["parsed_args"].append(True)
                    result_data["parsed_city_from"].append(
                        args["city_from"] if "city_from" in args else None
                    )
                    result_data["parsed_city_to"].append(
                        args["city_to"] if "city_to" in args else None
                    )
                    result_data["parsed_date"].append(
                        args["date"] if "date" in args else None
                    )
                    result_data["parsed_date_back"].append(
                        args["date_back"] if "date_back" in args else None
                    )
                    result_data["parsed_need_luggage"].append(
                        args["need_luggage"] if "need_luggage" in args else None
                    )
                    result_data["all_parsed_arguments"].append([x for x in args])

            else:
                result_data["parsed_fc"].append(False)
                result_data["function_name"].append(None)
                result_data["parsed_args"].append(False)
                result_data["parsed_city_from"].append(None)
                result_data["parsed_city_to"].append(None)
                result_data["parsed_date"].append(None)
                result_data["parsed_date_back"].append(None)
                result_data["parsed_need_luggage"].append(None)
                result_data["all_parsed_arguments"].append(None)
        else:
            result_data["parsed_fc"].append(False)
            result_data["function_name"].append(None)
            result_data["parsed_args"].append(False)
            result_data["parsed_city_from"].append(None)
            result_data["parsed_city_to"].append(None)
            result_data["parsed_date"].append(None)
            result_data["parsed_date_back"].append(None)
            result_data["parsed_need_luggage"].append(None)
            result_data["all_parsed_arguments"].append(None)

    df = pd.DataFrame.from_dict(result_data)
    df.to_csv(output_path)


@click.command()
@click.option(
    "--test-data-path",
    type=click.Path(exists=True),
    help="Path to the data in json format",
)
@click.option(
    "--model-outputs-path",
    type=click.Path(exists=True),
    help="Path of model answers in json format",
)
@click.option(
    "--result-df-path",
    type=click.Path(),
    help="Path to save csv df for metrics calculations",
)
def main(test_data_path: str, model_outputs_path: str, result_df_path: str):
    allign_results(
        test_data_path=test_data_path,
        model_outputs_path=model_outputs_path,
        output_path=result_df_path,
    )


if __name__ == "__main__":
    main()
