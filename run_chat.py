from llm_finetune.src.inference import ModelInference
from llm_finetune.src.parse_json_output import parse_fc


def API_mock(request: dict[str, str]):
    """
    Глупая mock-функция API какого-то сервиса по покупке билетов просто для примера
    """

    if request["name"] != "get_flight_info":
        response = f"'API response': 'Error: cannot run function {request['name']}, provide valid function'"
    elif (
        request["arguments"] is None
        or "city_from" not in request["arguments"]
        or "city_to" not in request["arguments"]
    ):
        response = """{'API response': 'Error: cannot run function get_flight_info without arguments ['city_from', 'city_to'], 
        provide valid arguments'"""
    else:
        response = (
            "{'API response': 'response is sent, flight information will be sent to your e-mail'}"
            ""
        )

    return response


def chat(model, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    while True:
        user_question = input("User (print 'exit' for stop): ")
        if user_question.strip() == "exit":
            print("Buy!")
            return

        messages.append({"role": "user", "content": user_question})
        model_output = model.generate(messages)
        model_answer = model_output["text"]
        messages.append({"role": "assistant", "content": model_answer})
        print("Assistant: ", model_answer)
        if model_output["functioncall"]:
            function_answer = API_mock(parse_fc(model_answer))
            messages.append({"role": "function", "content": function_answer})
            print("Function: ", function_answer)
            second_model_answer = model.generate(messages)["text"]
            messages.append({"role": "assistant", "content": second_model_answer})
            print("Assistant: ", second_model_answer)


def main(model_path: str, system_prompt: str):
    model = ModelInference(model_path=model_path, model_name="deepseek")
    chat(model, system_prompt)


if __name__ == "__main__":
    MODEL_PATH = "models/deepseek-coder-1.3b-function-calling-v1/ggml-model-f16.gguf"
    SYSTEM_PROMPT = """You are a helpful assistant with access to the following functions. Use them if required -
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
    main(model_path=MODEL_PATH, system_prompt=SYSTEM_PROMPT)
