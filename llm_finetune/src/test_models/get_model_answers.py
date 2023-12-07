import json

import click

from llm_finetune.src.inference import ModelInference


def test_flights_get_info(
    model_path: str, model_name: str, test_data_path: str, output_path: str
):
    system_prompt = """You are a helpful assistant with access to the following functions. Use them if required - 
    {
        "name": "get_flight_info",
        "description": "Get information about flight and plain tickets.",
        "parameters": {
            "type": "object",
            "properties": {
                "city_from": {
                    "type": "string",
                    "description": "Departure city"
                },
                "city_to": {
                    "type": "string",
                    "description": "Arrival city"
                },
                "date" : {
                    "type": "string",
                    "format": "date",
                    "description": "Date of Departure in MM-DD-YYYY format"
                },
                "date_back: {
                    "type": "string",
                    "format": "date",
                    "description": "Date of Arrival in MM-DD-YYYY format"
                },
                "need_luggage: {
                    "type": "boolean",
                    "description": "Whether a luggage is necessary, True or False (if information is provided in request)"
                },
            },
        "required": [
            "city_from",
            "city_to",
            "date",
            "date_back",
            "need_luggage"
        ]
    }
    """

    model = ModelInference(model_path=model_path, model_name=model_name)

    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    model_outputs = []

    for item in test_data:
        user_question = item["question"]
        index = item["index"]

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_question,
            },
        ]

        output = model.generate(messages=messages)
        model_outputs.append({"index": index, "answer": output})

    with open(output_path, "w") as f:
        json.dump(model_outputs, f)


@click.command()
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    help="Path to model in gguf format",
)
@click.option(
    "--model-name",
    type=str,
    help=f"Name of the model, one of {ModelInference._model_names}",
)
@click.option(
    "--test-data-path",
    type=click.Path(exists=True),
    help="Path to the data in json format",
)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Path to save model answers in json format",
)
def main(model_path: str, model_name: str, test_data_path: str, output_path: str):
    test_flights_get_info(
        model_path=model_path,
        model_name=model_name,
        test_data_path=test_data_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
