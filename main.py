import json
import argparse
import os

from utils import model_loader, get_evals
from common import make_report

def main():
    parser = argparse.ArgumentParser(description="Run sampling and evaluations using different samplers and evaluations.")
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--examples", type=int, help="Number of examples to use")
    parser.add_argument("--benchmarks", type=str, help="Benchmarks available: math, gpqa, mgsm, drop, humaneval, simpleqa, mmlu, browsecomp")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--reasoning_max_gen_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--api-key", type=str, default="token-abc123")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    # mergekey2resultpath = {}
    models = model_loader(args)
    benchmarks = get_evals(args)

    print(args.examples)
    print(len(benchmarks))
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for model_name, sampler in models.items():
        for benchmark_name, benchmark_obj in benchmarks.items():
            result = benchmark_obj(sampler)
            model_base_name = os.path.basename(model_name)
            file_stem = f"{benchmark_name}_{model_base_name}"

            result_filename = f"{args.output_dir}/{file_stem}.json"
            print(f"Writing results to {result_filename}")
            with open(result_filename, "w") as f:
                json.dump(result, f, indent=2)
            
    #         model_base_name = os.path.basename(model_name)
    #         file_stem = f"{benchmark_name}_{model_base_name}"

    #         report_filename = f"{args.output_dir}/{file_stem}.html"
    #         print(f"Writing report to {report_filename}")
    #         with open(report_filename, "w") as fh:
    #             # fh.write(make_report(result))
    #             fh.write()
    #         metrics = result.metrics | {"score": result.score}
    #         print(metrics)
    #         result_filename = f"{args.output_dir}/{file_stem}.json"
    #         with open(result_filename, "w") as f:
    #             f.write(json.dumps(metrics, indent=2))
    #         print(f"Writing results to {result_filename}")
    #         mergekey2resultpath[f"{file_stem}"] = result_filename

    # merge_metrics = []
    # for eval_model_name, result_filename in mergekey2resultpath.items():
    #     try:
    #         result = json.load(open(result_filename, "r+"))
    #     except Exception as e:
    #         print(e, result_filename)
    #         continue
    #     result = result.get("f1_score", result.get("score", None))
    #     eval_name = eval_model_name[: eval_model_name.find("_")]
    #     model_name = eval_model_name[eval_model_name.find("_") + 1 :]
    #     merge_metrics.append({"eval_name": eval_name, "model_name": model_name, "metric": result})
    # return merge_metrics


if __name__ == "__main__":
    main()
