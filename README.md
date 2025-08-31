# Simple Evals

## Installation

To get started with Simple Evals, you need to install the following Python packages. You can do this easily using pip. Run the following command in your terminal:

```bash
pip install vllm anthropic pandas human_eval bitsandbytes furiosa-llm
```

## API Tokens

To use Simple Evals, you need to set up your API tokens for Hugging Face and OpenAI. You can do this by exporting the tokens as environment variables in your terminal:

```bash
export HUGGINGFACE_TOKEN="token"
export OPENAI_API_KEY="token"
```

## Running the API Server

Before running the tests, ensure that an API endpoint is served. This is necessary for the models that require a predefined server. You can start the server by running the following command:

For GPU:
```bash
vllm serve [model_name] --tensor-parallel-size [num_gpu]
vllm serve [model_name] --pipeline-parallel-size [num_gpu]
```

For RNGD:
```bash
furiosa-llm serve [model_name]
```


## Command Line Arguments

When running the `main.py` script, you can specify several command line arguments to customize the behavior of the evaluations. Below are the available options:

- `--model`: Select a model by name. This is a required argument.
- `--examples`: Specify the number of examples to use. This is an optional argument.
- `--benchmarks`: Choose from available benchmarks: math, gpqa, mgsm, drop, humaneval, simpleqa, mmlu, browsecomp. This is a required argument.
- `--temperature`: Set the sampling temperature. Default is 0.
- `--max_tokens`: Define the maximum number of tokens. Default is 4096.
- `--reasoning_max_gen_tokens`: Set the maximum number of tokens for reasoning generation. Default is 4096.
- `--output_dir`: Specify the directory to save the results. Default is `./results`.
- `--api-key`: Provide the API key for authentication. Default is `token-abc123`.
- `--port`: Set the port number for the server. Default is 8000.

To run the script with these options, use the following command:

```bash
cd simple-evals
python main.py --model=[model_name] --benchmarks=[benchmark_name]
python main.py --model=furiosa-ai/Llama-3.3-70B-Instruct --benchmarks=mmlu,humaneval,mgsm
```
