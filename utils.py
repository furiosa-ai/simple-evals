from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.chat_completion_sampler import ChatCompletionSampler, OPENAI_SYSTEM_MESSAGE_API
from sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS

from eval_drop import DropEval
from eval_gpqa import GPQAEval
from eval_browsecomp import BrowseCompEval
from eval_humaneval import HumanEval
from eval_math import MathEval
from eval_mgsm import MGSMEval
from eval_mmlu import MMLUEval
from eval_simpleqa import SimpleQAEval

def model_loader(args):
    models = {}
    if "meta-llama" in args.model:
        print("Loading meta-llama model")
        models[args.model] = ChatCompletionSampler(
                model=args.model,
                use_predefined_server=True,
                api_key=args.api_key,
                port=args.port,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
        )
    elif "deepseek-ai" in args.model:
        print("Loading deepseek-ai model")
        models[args.model] = ChatCompletionSampler(
                model=args.model,
                use_predefined_server=True,
                api_key=args.api_key,
                port=args.port,
                temperature=args.temperature,
                max_tokens=args.reasoning_max_gen_tokens,
        )
    elif "o1-" in args.model or "o3-" in args.model:
        print("Loading o1- or o3- model")
        models[args.model] = OChatCompletionSampler(
                model=args.model,
        )
    elif "gpt-4" in args.model:
        print("Loading gpt-4 model")
        models[args.model] = ChatCompletionSampler(
                model=args.model,
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=args.max_tokens,
        )
    elif "claude" in args.model:
        print("Loading claude model")
        models[args.model] = ClaudeCompletionSampler(
                model=args.model,
                system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        )
    elif "furiosa-ai" in args.model:
        print("Loading furiosa-ai model")
        models[args.model] = ChatCompletionSampler(
                model=args.model,
                use_predefined_server=True,
                api_key=args.api_key,
                port=args.port,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
        )
    else:
        print("Model not supported, using default sampler")
        models[args.model] = ChatCompletionSampler(
                model=args.model,
                use_predefined_server=True,
                api_key=args.api_key,
                port=args.port,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
        )
    return models


def get_evals(args):
    grading_sampler = ChatCompletionSampler(model="gpt-4o")
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    benchmarks: list[str] = args.benchmarks.split(",")
    num_examples = (args.examples if args.examples else None)
    evals = {}
    
    for benchmark in benchmarks:
        match benchmark:
            case "mmlu":
                evals[benchmark] = MMLUEval(num_examples=num_examples)
            case "math":
                evals[benchmark] = MathEval(equality_checker=equality_checker, num_examples=num_examples, n_repeats=1, split="math_500_test")
            case "gpqa":
                evals[benchmark] = GPQAEval(n_repeats=1, num_examples=num_examples)
            case "mgsm":
                evals[benchmark] = MGSMEval(num_examples_per_lang=num_examples)
            case "drop":
                evals[benchmark] = DropEval(num_examples=num_examples, train_samples_per_prompt=1)
            case "humaneval":
                evals[benchmark] = HumanEval(num_examples=num_examples, num_samples_per_task=1, ks_passes=[1])
            case "simpleqa":
                evals[benchmark] = SimpleQAEval(grader_model=grading_sampler, num_examples=num_examples)
            case "browsecomp":
                evals[benchmark] = BrowseCompEval(grader_model=grading_sampler, num_examples=num_examples)
            case _:
                raise Exception(f"Unsupported benchmark: {benchmark}")
    return evals