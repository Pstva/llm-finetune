import json

import click


def check_function_call_consistency(msg: list[dict[str, str]]) -> bool:
    ch1, ch2 = False, False
    for m in msg:
        ch1 = ch1 or "<functioncall>" in m["content"]
        ch2 = ch2 or m["role"] == "function"
    if ch1 != ch2:
        return False
    return True


def process_msg(msg: str) -> list[dict[str, str]]:
    messages = []
    cur_content = []
    cur_role = None
    for line in msg.split("\n"):
        line = line.replace("<|endoftext|>", "").strip()
        if line.startswith("SYSTEM:"):
            if cur_content:
                messages.append(
                    {"role": cur_role, "content": "\n".join(cur_content).strip()}
                )
            cur_content = []
            cur_role = "system"
            cur_content.append(line[len("SYSTEM: ") :])
        elif line.startswith("USER:"):
            if cur_content:
                messages.append(
                    {"role": cur_role, "content": "\n".join(cur_content).strip()}
                )
            cur_content = []
            cur_role = "user"
            cur_content.append(line[len("USER: ") :])
        elif line.startswith("ASSISTANT"):
            if cur_content:
                messages.append(
                    {"role": cur_role, "content": "\n".join(cur_content).strip()}
                )
            cur_content = []
            cur_role = "assistant"
            cur_content.append(line[len("ASSISTANT: ") :])
        elif line.startswith("FUNCTION RESPONSE:"):
            if cur_content:
                messages.append(
                    {"role": cur_role, "content": "\n".join(cur_content).strip()}
                )
            cur_content = []
            cur_role = "function"
            cur_content.append(line[len("FUNCTION RESPONSE: ") :])
        else:
            cur_content.append(line)
    if cur_content:
        messages.append({"role": cur_role, "content": "\n".join(cur_content).strip()})
    return messages


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True),
    help="Path to glaive-function-calling-v2.json dataset",
)
@click.option(
    "-o", "--output-path", type=click.Path(), help="Path to save formatted json dataset"
)
def convert_glaive_data(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        data = json.load(f)
    all_messages = []
    for item in data:
        chat_part, system_part = item["chat"], item["system"]
        messages_list = process_msg(system_part)
        messages_list.extend(process_msg(chat_part))
        if check_function_call_consistency(messages_list):
            all_messages.append(messages_list)
    with open(output_path, "w") as f:
        json.dump(all_messages, f)
    return all_messages


if __name__ == "__main__":
    convert_glaive_data()
