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


def parse_fc(text: str):
    return_dict = {"name": None, "arguments": {}}
    parsed, parsed_dict = parse_answer_json(text)
    if parsed:
        return_dict["name"] = parsed_dict["name"] if "name" in parsed_dict else None
        parsed_args = False
        if "arguments" in parsed_dict:
            if isinstance(parsed_dict["arguments"], dict):
                parsed_args = True
                args = parsed_dict["arguments"]
            elif isinstance(parsed_dict["arguments"], str):
                parsed_args, args = parse_argumets(parsed_dict["arguments"])
        if parsed_args:
            return_dict["arguments"] = args

    return return_dict
