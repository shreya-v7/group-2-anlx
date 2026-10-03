'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Minimal dependencies (standard library only)

class PrepareEvaluationInput:
    """
    Prepare and format evaluation inputs for LLM evaluation. Allows selection
    of a representative sample. Supports data with query/response (pair)
    or response only (single).
    """
    def __init__(self, data_path: str, sample_size: int = 10, kind: str = "auto"):
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        if kind not in {"auto", "pair", "response"}:
            raise ValueError("kind must be 'auto', 'pair', or 'response'")
        self.kind = kind

    def load_data(self) -> List[Dict[str, Any]]:
        ext = self.data_path.suffix.lower()
        if ext == ".jsonl":
            with self.data_path.open("r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            with self.data_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, dict) and any(isinstance(v, list) for v in content.values()):
                records = []
                for v in content.values():
                    if isinstance(v, list):
                        records.extend(v)
            elif isinstance(content, list):
                records = content
            else:
                records = [content]
        else:
            raise ValueError("input file must be .json or .jsonl")
        return records

    def _detect_kind(self, record: Dict[str, Any]) -> str:
        if 'query' in record and 'response' in record:
            return "pair"
        elif 'prompt' in record and 'completion' in record:
            return "pair"
        elif 'response' in record or 'completion' in record:
            return "response"
        elif 'messages' in record:
            messages = record['messages']
            if isinstance(messages, list):
                roles = [m.get('role') for m in messages]
                if 'user' in roles and 'assistant' in roles:
                    return 'pair'
                elif 'assistant' in roles:
                    return 'response'
            return 'response'
        return 'response'

    def select_sample(self, records: List[Dict[str, Any]], random_seed: int = 42) -> List[Dict[str, Any]]:
        if len(records) <= self.sample_size:
            return records
        random.seed(random_seed)
        return random.sample(records, self.sample_size)

    def format_for_evaluation(self) -> List[Dict[str, Any]]:
        records = self.load_data()
        records = self.select_sample(records)
        kind = self.kind
        if kind == "auto" and records:
            kind = self._detect_kind(records[0])
        formatted = []
        for rec in records:
            if kind == 'pair':
                obj = {}
                if 'query' in rec and 'response' in rec:
                    obj = {'query': rec['query'], 'response': rec['response']}
                elif 'prompt' in rec and 'completion' in rec:
                    obj = {'query': rec['prompt'], 'response': rec['completion']}
                elif 'messages' in rec:
                    messages = rec['messages']
                    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
                    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)
                    obj = {'query': user_msg, 'response': assistant_msg}
                else:
                    obj = rec
                formatted.append(obj)
            elif kind == 'response':
                obj = {}
                if 'response' in rec:
                    obj = {'response': rec['response']}
                elif 'completion' in rec:
                    obj = {'response': rec['completion']}
                elif 'messages' in rec:
                    messages = rec['messages']
                    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)
                    obj = {'response': assistant_msg}
                else:
                    obj = rec
                formatted.append(obj)
            else:
                formatted.append(rec)
        return formatted

    def save_evaluation_dataset(self, out_path: str = "evaluation_dataset.json") -> str:
        data = self.format_for_evaluation()
        Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path


class HumanEvaluation:
    """
    Human-in-the-loop rating of AI responses with Likert scale judgments per example,
    logging criteria and instructions and saving results to JSON. Collects method choice for agreement
    and writes both data and a criteria/notes markdown file.
    """
    LIKERT_OPTIONS = {
        "1": "Completely Violates",
        "2": "Mostly Violates",
        "3": "Borderline",
        "4": "Mostly Complies",
        "5": "Fully Complies"
    }
    LIKERT_PROMPT = "Likert scale [1: Completely violates, 2: Mostly violates, 3: Borderline, 4: Mostly complies, 5: Fully complies]"

    def __init__(self, evaldataset_path: str):
        self.evaldataset_path = Path(evaldataset_path)
        self.evaldataset = self._load_dataset()
        self.labels = []
        self.criteria_description = ""

    def _load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.evaldataset_path, encoding="utf-8") as f:
            return json.load(f)

    def run(self):
        print("\n---- LLM Human Evaluation ----")
        print("You will be asked to rate each AI response according to your own evaluation criteria.")
        self.criteria_description = input("\nPlease enter YOUR EVALUATION CRITERIA that will guide your ratings (this instruction will be shown each time):\n> ").strip()
        print("\nInstructions: For each example, consider if the AI response COMPLIES or VIOLATES your criteria. Rate using this Likert scale:")
        for num, desc in self.LIKERT_OPTIONS.items():
            print(f"  {num}: {desc}")
        print()
        for idx, ex in enumerate(self.evaldataset, 1):
            print(f"\nExample {idx} of {len(self.evaldataset)}:")
            print(f"Your Criteria: {self.criteria_description}")
            if 'query' in ex:
                print(f"Prompt/Query: {ex['query']}")
            print(f"AI Response: {ex.get('response')}")
            while True:
                inp = input(f"Your rating ({self.LIKERT_PROMPT}): ").strip()
                if inp in self.LIKERT_OPTIONS:
                    rating = int(inp)
                    break
                print(f"  Please enter a number from 1 to 5.")
            self.labels.append({
                **ex,
                "rating": rating
            })
        # Save labeled dataset
        labeled_path = self.evaldataset_path.parent / "labeled_evaluation_dataset.json"
        labeled_path.write_text(json.dumps(self.labels, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] Labeled evaluation data saved to {labeled_path}\n")
        self._save_criteria_md(labeled_path.parent)
        print("Now, please select an appropriate agreement metric for future evaluations.")
        metric_name = input("Enter the agreement metric you will use (e.g. Cohen's kappa, Krippendorff's alpha): ").strip()
        criteria_md_path = labeled_path.parent / "evaluation_criteria.md"
        with open(criteria_md_path, "a", encoding="utf-8") as f:
            f.write(f"\n\nAgreement metric chosen: {metric_name}\n")
        print(f"\nAll done!\n1. Edit {criteria_md_path} to DESCRIBE your evaluation criteria in your own words.\n2. Email the following to the teaching team:\n\n   * {labeled_path.name}\n   * evaluation_criteria.md\n")
        return str(labeled_path)

    def _save_criteria_md(self, outdir: Path):
        md_path = outdir / "evaluation_criteria.md"
        if not md_path.exists():
            with open(md_path, 'w', encoding="utf-8") as f:
                f.write(f"# Evaluation Criteria\n\nDescribe IN YOUR OWN WORDS the instructions or rubric you used for rating the AI.\n\n{self.criteria_description}\n")
        else:
            pass


class LLMJudgeEvaluation:
    """
    Automatically evaluate a dataset using an LLM judge. Accepts Pydantic model or dataclass (could be function/class).
    Records labels and computes the chosen evaluation measure.
    """
    def __init__(self, evaldataset_path: str, judge_fn, judge_schema_source: Optional[str] = None):
        """judge_fn should accept (example, criteria_description) and return label/int/decision"""
        self.evaldataset_path = Path(evaldataset_path)
        self.evaldataset = self._load_dataset()
        self.judge_fn = judge_fn
        self.criteria_description = ""
        self.llmjudge_labels = []
        self.judge_schema_source = judge_schema_source

    def _load_dataset(self):
        with open(self.evaldataset_path, encoding="utf-8") as f:
            return json.load(f)

    def run(self):
        print("\n---- LLM Judge Evaluation ----")
        self.criteria_description = input("Enter the EVALUATION CRITERIA for the LLM Judge to use: ").strip()
        print("LLM Judge is now evaluating...\n")
        for idx, ex in enumerate(self.evaldataset, 1):
            labeled = {
                **ex,
                "judge_label": self.judge_fn(ex, self.criteria_description)
            }
            self.llmjudge_labels.append(labeled)
        eval_dir = Path("data/evaluations/llmjudge")
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / f"llmjudgeevaluation_{os.getpid()}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in self.llmjudge_labels:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"LLM Judge labels written to {out_path}")
        # Standard/custom eval measures
        print("Apply a standard (e.g. accuracy) or custom evaluation measure to the LLM Judge's output.")
        measure_type = input("Type 'standard' or 'custom' for standard_eval_measure or custom_eval_measure: ").strip().lower()
        if measure_type == 'standard':
            measure_name = input("Enter the standard measure (e.g. accuracy): ").strip()
            result = StandardEvaluationMeasures.apply(self.llmjudge_labels, measure_name)
        else:
            measure_name = input("Enter your custom evaluation measure: ").strip()
            result = CustomEvaluationMeasures.apply(self.llmjudge_labels, measure_name)
        results_file = eval_dir / "llmjudgeevaluationresults.jsonl"
        with open(results_file, 'w', encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
        # Prepare the "criteria.md" file for instructions
        criteria_md = eval_dir / "evaluation_criteria.md"
        with open(criteria_md, 'w', encoding="utf-8") as f:
            f.write(f"# Evaluation Criteria (LLM Judge)\n\nPlease DESCRIBE in your own words the criteria used.\n\n{self.criteria_description}\n")
            if self.judge_schema_source:
                f.write(f"\nPython file used for labeling: {self.judge_schema_source}\n")
            f.write("\nDescribe your chosen metrics for the LLM Judge evaluation here before emailing.\n")
        print("\nAction Items:\n1. Edit evaluation_criteria.md to DESCRIBE your evaluation criteria and the metrics you chose.\n2. Email the following files to the teaching team:\n   * llmjudgeevaluation_{os.getpid()}.jsonl\n   * evaluation_criteria.md\n   * The python file(s) containing your judge (Pydantic model or dataclass)\n   * llmjudgeevaluationresults.jsonl\n")
        return str(results_file)


class AgreementMeasures:
    """
    Minimal metrics for agreement between multiple labelers (inter-rater agreement).
    Example: simple percent agreement, placeholder for Cohen's kappa, etc.
    """
    @staticmethod
    def percent_agreement(labels1: List[Any], labels2: List[Any]) -> float:
        if len(labels1) != len(labels2) or not labels1:
            raise ValueError('Label lists must be non-empty and of same length.')
        agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
        return agree / len(labels1)

    @staticmethod
    def cohen_kappa(labels1, labels2):
        # Simple Cohen's kappa for two raters and categorical labels.
        from collections import Counter
        if len(labels1) != len(labels2) or not labels1:
            raise ValueError('Label lists must be same length and non-empty.')
        N = len(labels1)
        pa = sum(a == b for a, b in zip(labels1, labels2)) / N
        label_counts_1 = Counter(labels1)
        label_counts_2 = Counter(labels2)
        label_keys = set(label_counts_1) | set(label_counts_2)
        pe = sum(label_counts_1[k]/N * label_counts_2[k]/N for k in label_keys)
        if pe == 1.0:  # Prevent div by zero
            return 1.0
        return (pa - pe) / (1.0 - pe)


class StandardEvaluationMeasures:
    """
    Minimal standard metrics for LLM evaluation (accuracy, average rating, etc.)
    """
    @staticmethod
    def apply(dataset: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        if not dataset:
            return {"result": None, "error": "No data"}
        # Accept 'accuracy', 'average', etc.
        ratings = [ex.get("judge_label") or ex.get("rating") for ex in dataset]
        ratings = [r for r in ratings if isinstance(r, int) or isinstance(r, float)]
        if not ratings:
            return {"result": None, "error": "No ratings in dataset."}
        name = name.lower()
        if name == "accuracy":
            # Assume ground-truth label is in ex['label'] or ex['gold']
            truths = [ex.get('label') or ex.get('gold') for ex in dataset]
            matches = [a == b for a, b in zip(ratings, truths) if b is not None]
            return {"accuracy": sum(matches)/len(matches) if matches else None}
        elif name in ("average", "mean", "average_rating"):
            return {"average_rating": sum(ratings)/len(ratings)}
        else:
            return {"result": None, "error": f"Unknown metric {name}"}


class CustomEvaluationMeasures:
    """
    Minimal stub allowing a user to define their own evaluation function.
    """
    @staticmethod
    def apply(dataset: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        print(f"Running custom evaluation: {name} (implement your logic here)")
        # For students: replace with your own custom code! Example: count how many labels == 5
        labels = [ex.get('judge_label') or ex.get('rating') for ex in dataset]
        num_five = sum(1 for l in labels if l == 5)
        return {"num_label_five": num_five, "total": len(labels)}

# End of llm_evaluation.py
