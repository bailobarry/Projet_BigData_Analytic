# Analysis – analyse quantitative, sémantique, qualitative et LLM-as-a-Judge

from src.analysis.quantitative import (
    load_run_results,
    compute_basic_stats,
    compare_runs,
    generate_report as generate_quantitative_report,
)
from src.analysis.semantic import (
    load_model,
    diversity_score,
    robustness_score,
    combined_score,
    compare_runs_semantic,
    generate_report as generate_semantic_report,
)
from src.analysis.qualitative import (
    select_extreme_cases,
    detect_instruction_violations,
    classify_errors,
    analyze_by_category,
    generate_qualitative_report,
)
from src.analysis.llm_judge import (
    LLMJudge,
    evaluate_diversity,
    evaluate_robustness,
    generate_report as generate_llm_judge_report,
)

__all__ = [
    # Quantitatif
    "load_run_results",
    "compute_basic_stats",
    "compare_runs",
    "generate_quantitative_report",
    # Sémantique
    "load_model",
    "diversity_score",
    "robustness_score",
    "combined_score",
    "compare_runs_semantic",
    "generate_semantic_report",
    # Qualitatif
    "select_extreme_cases",
    "detect_instruction_violations",
    "classify_errors",
    "analyze_by_category",
    "generate_qualitative_report",
    # LLM Judge
    "LLMJudge",
    "evaluate_diversity",
    "evaluate_robustness",
    "generate_llm_judge_report",
]