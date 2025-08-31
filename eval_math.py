"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import re
from typing import Literal
import pandas
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from common import ANSWER_PATTERN, HTML_JINJA, check_equality, map_with_progress, aggregate_results, jinja_env
from _types import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


class MathEval(Eval):
    def __init__(self, equality_checker: SamplerBase, num_examples: int | None = None, n_repeats: int = 16, split: Literal["math_test", "math_500_test"] = "math_test"):
        df = pandas.read_csv(f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")]
            fail_reason = None
            response_text = sampler(prompt_messages)
            if response_text is None:
                fail_reason = "[finish when reasoning]"
                extracted_answer = None
            else:
                match = re.search(ANSWER_PATTERN, response_text)
                extracted_answer = match.group(1) if match else None
            score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))
            return {
                "prompt_messages": prompt_messages,
                "response_text": response_text,
                "correct_answer": row["Answer"],
                "extracted_answer": extracted_answer,
                "fail_reason": fail_reason,
                "score": score,
            }
        with ThreadPool(min(50, len(self.examples))) as pool:
            results = list(tqdm(pool.imap(fn, self.examples), total=len(self.examples), desc="Evaluating Math"))

        return results
        #     html = jinja_env.from_string(HTML_JINJA).render(
        #         prompt_messages=prompt_messages,
        #         next_message=dict(content=response_text, role="assistant"),
        #         score=score,
        #         correct_answer=row["Answer"],
        #         extracted_answer=extracted_answer,
        #         fail_reason="",
        #     )
        #     metrics_ = {"stop_in_reasoning": 1 if fail_reason else 0}
        #     convo = prompt_messages + [dict(content=response_text, role="assistant")]
        #     return SingleEvalResult(html=html, score=score, convo=convo, metrics=metrics_)

        # results = map_with_progress(fn, self.examples)
        # return aggregate_results(results)
