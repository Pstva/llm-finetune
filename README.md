# Дообучение LLM на задачу Function Calling


## 0. Активация виртуального окружения и вызов консольного интерфейса для чата с моделью

```
conda create -n mtc_venv python=3.10
poetry install
```

Затем можно пообщаться с моделью:

```sh
python run_chat.py
```

Отчет о задании [тут](report/REPORT.md).


Далее будет описаны шаги и приложен код к каждому шагу.

## 1. Данные

### Glaive Dataset

Для файнтьюна модели был взят Function Calling датасет: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2


```sh
# load
wget -P data/raw/ https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json

# preprocess
python llm_finetune/src/data/preprocess_data.py -i data/raw/glaive-function-calling-v2.json -o data/preprocessed/glaive_function.json

# sample
python llm_finetune/src/data/sample_data.py -i data/preprocessed/glaive_function.json -o data/preprocessed/glaive_function_sampled.json
```

### Генерация моего тестового датасета

Для тестирования был сгенерирован мой датасет с вопросами про билеты на самолеты, а так же просто другимим вопросами, не предполагающими вызов API-функции.

#### 1. Генерация данных о билетах

Сначала были сгенерированы данные информации о городе вылета и прилета, датах прилета и вылета и багаже в разных комбинациях:

```sh
python llm_finetune/src/data/generate_test_data.py -o data/preprocessed/test.csv
```


#### 2. Генерация вопросов про билеты

Затем по этим данным были сгенерированы вопросы с помощью моделей Zephyr и Llama.

Предварительно скачаны модели в gguf-формате отсюда:
[zephyr-7b-beta.Q4_K_M.gguf](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) и [llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF).

Блокнот с генерацией вопросов с помощью LLama2 и Zephyr [notebooks/generate_questions.ipynb](notebooks/generate_questions.ipynb)


3. Далее был небольшой этап, когда я вручную проверила и поправила некоторое кол-во вопросов по авиабилетам, из чего получился [датасет](data/test/raw/test_with_questions_fixed_part.csv) на 118 вопросов.


[Финальный датасет в json-формате](data/test/final/function_calling_test.json) соединяет в себе 118 вопросов по авиабилетам (где необходим function calling) и 113 общих вопросов.

Именно на нем я тестировала свою модель.

## 2. Finetuning модели

Дообучала модель Deepseek-coder-Instruct 1.3B на [моем сэмпле Glaive](data/preprocessed/glaive_function_sampled.json).

Код прогоняла на Google Colab, [notebook с кодом](notebooks/deepseek_finetuning.ipynb).

Обученная модель и она же в формате gguf залита на HF [pestova/deepseek-coder-1.3b-function-calling-v1](https://huggingface.co/pestova/deepseek-coder-1.3b-function-calling-v1).

Для перевода в gguf использовалась библиотека [llama.cpp](https://github.com/ggerganov/llama.cpp). 


**Скачать модель в папку models**
```sh
cd models
git lfs clone https://huggingface.co/pestova/deepseek-coder-1.3b-function-calling-v1
```

Сейчас там уже лежит модель в .gguf формате. Он был получен следующим образом:


```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
pipenv install
pipenv shell

python convert.py "../models/deepseek-coder-1.3b-function-calling-v1" --vocabtype bpe
```


## 3. Оценка модели




```sh
python -m llm_finetune.src.test_models.get_model_answers \
                 --model-path "models/deepseek-coder-1.3b-function-calling-v1/ggml-model-f16.gguf" \
                 --model-name "deepseek" \
                 --test-data-path  "data/test/final/function_calling_test.json" \
                 --output-path "data/test/final/deepseek_answers.json"


python -m llm_finetune.src.test_models.calculate_metrics --test-data-path "data/test/final/function_calling_test.json" \
                 --model-outputs-path "data/test/final/deepseek_answers.json" \
                 --result-df-path "data/test/final/result_df.csv"
```


Данными скриптами был получен датасет с groung truth значениями для functions и их аргументов и ответы модели.
Результаты для отчета подсчитаны в [ноутбуке](notebooks/results.ipynb).
