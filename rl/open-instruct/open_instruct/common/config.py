"""
Configuration Loader Module

Handles loading and parsing of config.yaml with support for:
- Path template interpolation (e.g., {paths.domains}/{domain_id})
- Nested configuration access
- Environment variable substitution
"""

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


def _get_dataclass_field_names(cls):
    """Get set of field names for a dataclass."""
    return {f.name for f in fields(cls)}


@dataclass
class PathsConfig:
    """Data paths configuration."""

    domains: str = "Data/domains"
    data_synthesis: str = "Data/data_synthesis_stage"
    env_synthesis: str = "Data/env_synthesis_stage"


@dataclass
class ExecutionConfig:
    """Execution configuration for parallel processing."""

    max_concurrent: int = 20
    max_retries: int = 10
    retry_delay: float = 10.0
    json_parse_retries: int = (
        3  # JSON parse retry config: 0=no retry, -1=infinite, >0=retry count
    )


# Data Synthesis Stage Configs


@dataclass
class Step1GraphConfig:
    """Step 1: Graph Sampling configuration."""

    input_topics_dir: str = "{paths.domains}/{domain_id}"
    input_topic_path: str = "{paths.domains}/{domain_id}/{topic_id}.json"
    output_graphs_dir: str = "{paths.data_synthesis}/graphs/{domain_id}/{topic_id}"
    num_graphs: int = 2
    num_nodes_range: tuple = (4, 5)
    search_node_ratio: float = 0.5
    manipulation_node_ratio: float = 0.5
    max_indegrees: int = 3
    graph_configs: Optional[list] = None  # Multiple graph configurations


@dataclass
class Step2MetaSubtaskConfig:
    """Step 2: Meta Subtask Generation configuration."""

    input_graphs_dir: str = "{paths.data_synthesis}/graphs/{domain_id}/{topic_id}"
    output_mtg_dir: str = (
        "{paths.data_synthesis}/meta_subtask_generator/{domain_id}/{topic_id}"
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class Step3RawCallDocConfig:
    """Step 3: Raw Call Doc Generation configuration."""

    input_mtg_dir: str = (
        "{paths.data_synthesis}/meta_subtask_generator/{domain_id}/{topic_id}"
    )
    output_rcsg_dir: str = (
        "{paths.data_synthesis}/rawcalldoc_subtask_graphs/{domain_id}/{topic_id}"
    )
    output_rawcalldocs_dir: str = (
        "{paths.data_synthesis}/rawcalldocs/{domain_id}/{topic_id}"
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )
    max_schema_retries: int = 3  # Retry when schema validation fails


@dataclass
class Step4MergeConfig:
    """Step 4: Call Doc Merge configuration (task-level)."""

    input_rawcalldocs_dir: str = "{paths.data_synthesis}/rawcalldocs/{domain_id}"
    output_calldocs_dir: str = "{paths.data_synthesis}/calldocs/{domain_id}"
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )
    merge_mode: str = "merge"  # "merge" or "rename" mode


@dataclass
class Step5SubtaskConfig:
    """Step 5: Scene-Driven Subtask Description Generation configuration."""

    input_mtg_dir: str = (
        "{paths.data_synthesis}/meta_subtask_graphs/{domain_id}/{topic_id}"
    )
    input_calldocs_dir: str = "{paths.data_synthesis}/calldocs/{domain_id}"
    input_merge_report_path: str = (
        "{paths.data_synthesis}/calldocs/{domain_id}/merge_report.json"
    )
    output_rcsg_dir: str = (
        "{paths.data_synthesis}/calldoc_subtask_graphs/{domain_id}/{topic_id}"
    )
    scene_count: int = 3
    multi_scene_autoregressive: bool = True  # True: Batch-scene, False: Single-scene
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class Step6TscgConfig:
    """Step 6: TSCG (Task-Subtask-CallDoc-Graph) Generation configuration."""

    input_rcsg_dir: str = (
        "{paths.data_synthesis}/calldoc_subtask_graphs/{domain_id}/{topic_id}"
    )
    output_tscg_dir: str = "{paths.data_synthesis}/tscgs/{domain_id}/{topic_id}"
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class Step7ReviewConfig:
    """Step 7: Review & Refine configuration."""

    input_tscgs_dir: str = "{paths.data_synthesis}/tscgs/{domain_id}/{topic_id}"
    output_refined_tscgs_dir: str = (
        "{paths.data_synthesis}/refined_tscgs/{domain_id}/{topic_id}"
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class DataSynthesisConfig:
    """Data Synthesis Stage configuration container."""

    step1_graph: Step1GraphConfig = field(default_factory=Step1GraphConfig)
    step2_meta_subtask: Step2MetaSubtaskConfig = field(
        default_factory=Step2MetaSubtaskConfig
    )
    step3_rawcalldoc: Step3RawCallDocConfig = field(
        default_factory=Step3RawCallDocConfig
    )
    step4_merge: Step4MergeConfig = field(default_factory=Step4MergeConfig)
    step5_subtask: Step5SubtaskConfig = field(default_factory=Step5SubtaskConfig)
    step6_tscg: Step6TscgConfig = field(default_factory=Step6TscgConfig)
    step7_review: Step7ReviewConfig = field(default_factory=Step7ReviewConfig)


# Environment Synthesis Stage Configs (6 steps in v0.4)


@dataclass
class Step1SimuDocConfig:
    """Step 1: SimuDoc Generation configuration (task-level)."""

    input_refined_tscgs_dir: str = (
        "{paths.data_synthesis}/refined_tscgs/{domain_id}/{topic_id}"
    )
    output_tooldocs_dir: str = (
        "{paths.env_synthesis}/tooldocs/{domain_id}/{topic_id}/{graph_id}"
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )
    use_split_prompts: bool = (
        True  # Use split prompts (search/manipulation) instead of unified prompt
    )


@dataclass
class Step2DbPlanConfig:
    """Step 2: DB Plan Generation configuration (graph-level)."""

    input_mtg_dir: str = (
        "{paths.data_synthesis}/meta_subtask_graphs/{domain_id}/{topic_id}"
    )
    input_tscgs_dir: str = "{paths.data_synthesis}/refined_tscgs/{domain_id}/{topic_id}"
    input_tooldocs_dir: str = "{paths.env_synthesis}/tooldocs/{domain_id}/{topic_id}"
    output_dbplans_dir: str = "{paths.env_synthesis}/db_plans/{domain_id}/{topic_id}"
    max_dbplan_retries: int = 3
    max_subdb_schema_retries: int = 3
    schema_cot: bool = (
        False  # If true, request step-by-step reasoning before schema JSON output
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class Step3ToolscriptConfig:
    """Step 3: Toolscript synthesis configuration (graph/toolset-level)."""

    input_tooldocs_dir: str = (
        "{paths.env_synthesis}/tooldocs/{domain_id}/{topic_id}/{graph_id}"
    )
    input_dbplans_dir: str = "{paths.env_synthesis}/db_plans/{domain_id}/{topic_id}"
    output_scripts_dir: str = (
        "{paths.env_synthesis}/tool_scripts/{domain_id}/{topic_id}/{graph_id}"
    )
    output_verifiers_dir: str = (
        "{paths.env_synthesis}/verifiers/{domain_id}/{topic_id}/{graph_id}"
    )
    output_reports_dir: str = (
        "{paths.env_synthesis}/toolscript_reports/{domain_id}/{topic_id}"
    )
    toolset_workers: int = 1
    tool_workers: int = 4
    fail_fast: bool = False
    max_toolscript_retries: int = 3
    # ===== Optional test-driven refinement (recommended) =====
    enable_verifiers: bool = False
    records_per_record_type: int = 2
    max_fixture_retries: int = 5
    positive_tests_per_tool: int = 2
    negative_tests_per_tool: int = 1
    max_testgen_retries: int = 5
    unittest_timeout_seconds: int = 60
    max_repair_rounds: int = 5
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )

    # SimuDoc Fix Configuration (formerly Step 3.5)
    enable_simudoc_fix: bool = True
    simudoc_fix_temperature: float = 0.2

    # Tool Logic Generation (Step 3.0 - before scripts)
    enable_tool_logic_generation: bool = True  # Generate tool_logic before scripts
    tool_logic_use_split_prompts: bool = (
        True  # Use split prompts (search/manipulation) for tool_logic
    )

    # Template mode control
    use_split_prompts: bool = (
        True  # Use split prompts (search/manipulation) for scripts and repairs
    )

    # Test generation mode control
    split_test_generation: bool = True  # If True, generate positive and negative tests separately with different prompts (async parallel)


@dataclass
class Step4InitialDbConfig:
    """Step 4: InitialDB Generation configuration (task-level)."""

    input_tscgs_dir: str = "{paths.data_synthesis}/refined_tscgs/{domain_id}/{topic_id}"
    input_dbplans_dir: str = "{paths.env_synthesis}/db_plans/{domain_id}/{topic_id}"
    input_tooldocs_dir: str = (
        "{paths.env_synthesis}/tooldocs/{domain_id}/{topic_id}/{graph_id}"
    )
    output_jsondbs_dir: str = "{paths.env_synthesis}/json_dbs/{domain_id}/{topic_id}"
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class Step5GtGraphConfig:
    """Step 5: GroundTruthGraph Generation configuration (task-level)."""

    input_tscgs_dir: str = "{paths.data_synthesis}/refined_tscgs/{domain_id}/{topic_id}"
    input_dbplans_dir: str = "{paths.env_synthesis}/db_plans/{domain_id}/{topic_id}"
    input_tooldocs_dir: str = (
        "{paths.env_synthesis}/tooldocs/{domain_id}/{topic_id}/{graph_id}"
    )
    input_jsondbs_dir: str = "{paths.env_synthesis}/json_dbs/{domain_id}/{topic_id}"
    input_scripts_dir: str = (
        "{paths.env_synthesis}/tool_scripts/{domain_id}/{topic_id}/{graph_id}"
    )
    output_gtgraphs_dir: str = (
        "{paths.env_synthesis}/groundtruth_graphs/{domain_id}/{topic_id}"
    )
    output_jsondbs_dir: str = "{paths.env_synthesis}/json_dbs/{domain_id}/{topic_id}"
    max_toolcall_retries: int = 3
    max_submit_answer_retries: int = 3
    max_patch_attempts: int = 3
    max_patch_schema_retries: int = 3
    save_error_traces: bool = True
    error_log_dir: str = "{paths.env_synthesis}/error_logs/{domain_id}/{topic_id}"
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )
    # Auto-repair configuration
    enable_auto_repair: bool = True  # Enable automatic tool script repair
    max_repair_rounds: int = 3  # Maximum repair rounds per graph
    repair_include_subdb: bool = True  # Include subdb state in repair context
    allow_two_of_three_repair: bool = True  # Repair when 2/3 tasks fail at same node
    repair_use_split_prompts: bool = (
        True  # Use split prompts (search/manipulation) for gtgraph repair
    )


@dataclass
class Step6ReviewRefineConfig:
    """Step 6: Review & refine submit_answer for training outputs (task-level)."""

    output_submit_answers_dir: str = (
        "{paths.env_synthesis}/submit_answers/{domain_id}/{topic_id}"
    )
    max_refine_retries: int = 3
    max_concurrent: Optional[int] = None  # Optional: override execution.max_concurrent
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class SimilarityThresholdsConfig:
    """Similarity thresholds for noise tool filtering."""

    description_similarity: float = 0.3
    parameter_similarity: float = 0.2
    output_similarity: float = 0.1


@dataclass
class Step7NoiseInjectionConfig:
    """Step 7: Noise Injection configuration (task-level)."""

    noise_tool_ratio: float = 0.2  # Ratio of noise tools to original task tools
    db_copy_mode: str = "deep"  # "deep" (full copy) or "shallow" (reference path only)
    similarity_thresholds: SimilarityThresholdsConfig = field(
        default_factory=SimilarityThresholdsConfig
    )
    matching_strategy: str = (
        "strict"  # "strict" (all must differ) or "any" (any one differs)
    )


@dataclass
class Step8AddUserConfig:
    """Step 8: User Agent Prompt Generation configuration (task-level)."""

    input_tscgs_dir: str = "{paths.data_synthesis}/refined_tscgs/{domain_id}/{topic_id}"
    output_user_prompts_dir: str = (
        "{paths.env_synthesis}/user_prompts/{domain_id}/{topic_id}"
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class Step9ChecklistConfig:
    """Step 9: Checklist Generation configuration (task-level)."""

    output_checklists_dir: str = (
        "{paths.env_synthesis}/step9_checklists/{domain_id}/{topic_id}"
    )
    llm_backend: Optional[str] = None  # Backend to use for this step
    temperature: Optional[float] = (
        None  # Optional: override backend default temperature
    )


@dataclass
class EnvSynthesisConfig:
    """Environment Synthesis Stage configuration container (9 steps)."""

    step1_simudoc: Step1SimuDocConfig = field(default_factory=Step1SimuDocConfig)
    step2_dbplan: Step2DbPlanConfig = field(default_factory=Step2DbPlanConfig)
    step3_toolscript: Step3ToolscriptConfig = field(
        default_factory=Step3ToolscriptConfig
    )
    step4_initialdb: Step4InitialDbConfig = field(default_factory=Step4InitialDbConfig)
    step5_gtgraph: Step5GtGraphConfig = field(default_factory=Step5GtGraphConfig)
    step6_review_refine: Step6ReviewRefineConfig = field(
        default_factory=Step6ReviewRefineConfig
    )
    step7_noise_injection: Step7NoiseInjectionConfig = field(
        default_factory=Step7NoiseInjectionConfig
    )
    step8_adduser: Step8AddUserConfig = field(default_factory=Step8AddUserConfig)
    step9_checklist: Step9ChecklistConfig = field(default_factory=Step9ChecklistConfig)


# Prompts Configs


@dataclass
class DataSynthesisPromptsConfig:
    """Data Synthesis Stage prompts configuration."""

    meta_subtask_generator: str = "prompts/data_synthesis/meta_subtask_generator.md"
    scene_generator: str = "prompts/data_synthesis/scene_generator.md"
    subtask_generator: str = "prompts/data_synthesis/subtask_generator.md"
    autoregressive_subtask_generator: str = (
        "prompts/data_synthesis/autoregressive_subtask_generator.md"
    )
    single_scene_autoregressive_subtask_generator: str = (
        "prompts/data_synthesis/single_scene_autoregressive_subtask_generator.md"
    )
    rawcalldoc_generator: str = "prompts/data_synthesis/rawcalldoc_generator.md"
    task_level_tool_mapper: str = "prompts/data_synthesis/task_level_tool_mapper.md"
    parameter_synonym_mapper: str = "prompts/data_synthesis/parameter_synonym_mapper.md"
    parameter_rename_mapper: str = "prompts/data_synthesis/parameter_rename_mapper.md"
    tool_rename_and_unify: str = "prompts/data_synthesis/tool_rename_and_unify.md"
    tscg_generator: str = "prompts/data_synthesis/tscg_generator.md"
    review_query_refiner: str = "prompts/data_synthesis/review_query_refiner.md"
    review_submit_format_generator: str = (
        "prompts/data_synthesis/review_submit_format_generator.md"
    )


@dataclass
class EnvSynthesisPromptsConfig:
    """Environment Synthesis Stage prompts configuration (6 steps in v0.4)."""

    simudoc_generator: str = "prompts/env_synthesis/step1_simudoc/simudoc_generator.md"
    simudoc_generator_search: str = (
        "prompts/env_synthesis/step1_simudoc/simudoc_generator_search.md"
    )
    simudoc_generator_manipulation: str = (
        "prompts/env_synthesis/step1_simudoc/simudoc_generator_manipulation.md"
    )
    augment_output_schema: str = (
        "prompts/env_synthesis/step1_simudoc/augment_output_schema.md"
    )

    dbplan_generator: str = "prompts/env_synthesis/step2_dbplan/dbplan_generator.md"
    subdb_schema_generator: str = (
        "prompts/env_synthesis/step2_dbplan/subdb_schema_generator.md"
    )

    tool_logic_generator: str = (
        "prompts/env_synthesis/step3_toolscript/tool_logic_generator.md"
    )
    tool_logic_generator_search: str = (
        "prompts/env_synthesis/step3_toolscript/tool_logic_generator_search.md"
    )
    tool_logic_generator_manipulation: str = (
        "prompts/env_synthesis/step3_toolscript/tool_logic_generator_manipulation.md"
    )
    toolscript_generator: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_generator.md"
    )
    toolscript_generator_search: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_generator_search.md"
    )
    toolscript_generator_manipulation: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_generator_manipulation.md"
    )
    toolscript_fixture_generator: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_fixture_generator.md"
    )
    toolscript_test_methods_generator: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_test_methods_generator.md"
    )
    toolscript_test_methods_generator_positive: str = "prompts/env_synthesis/step3_toolscript/toolscript_test_methods_generator_positive.md"
    toolscript_test_methods_generator_negative: str = "prompts/env_synthesis/step3_toolscript/toolscript_test_methods_generator_negative.md"
    toolscript_repair: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_repair.md"
    )
    toolscript_repair_search: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_repair_search.md"
    )
    toolscript_repair_manipulation: str = (
        "prompts/env_synthesis/step3_toolscript/toolscript_repair_manipulation.md"
    )
    simudoc_fixer: str = "prompts/env_synthesis/step3_toolscript/simudoc_fixer.md"

    initial_db_generator: str = (
        "prompts/env_synthesis/step4_initialdb/initial_db_generator.md"
    )

    gtgraph_toolcall: str = "prompts/env_synthesis/step5_gtgraph/gtgraph_toolcall.md"
    gtgraph_submit_answer_toolcall: str = (
        "prompts/env_synthesis/step5_gtgraph/gtgraph_submit_answer_toolcall.md"
    )
    # GTgraph judge prompts (split by tool_response.status)
    gtgraph_judge_success: str = (
        "prompts/env_synthesis/step5_gtgraph/gtgraph_judge_success.md"
    )
    gtgraph_judge_failure_search: str = (
        "prompts/env_synthesis/step5_gtgraph/gtgraph_judge_failure_search.md"
    )
    gtgraph_judge_failure_manipulate: str = (
        "prompts/env_synthesis/step5_gtgraph/gtgraph_judge_failure_manipulate.md"
    )
    toolscript_gtgraph_repair: str = (
        "prompts/env_synthesis/step5_gtgraph/toolscript_gtgraph_repair.md"
    )
    toolscript_gtgraph_repair_search: str = (
        "prompts/env_synthesis/step5_gtgraph/toolscript_gtgraph_repair_search.md"
    )
    toolscript_gtgraph_repair_manipulation: str = (
        "prompts/env_synthesis/step5_gtgraph/toolscript_gtgraph_repair_manipulation.md"
    )

    submit_answer_review_refine: str = (
        "prompts/env_synthesis/step6_review/submit_answer_review_refine.md"
    )

    # Step 8: User Agent Prompt
    user_prompt_subqueries: str = (
        "prompts/env_synthesis/step8_adduser/subqueries_generator.md"
    )
    user_prompt_instructions: str = (
        "prompts/env_synthesis/step8_adduser/instructions_generator.md"
    )

    # Step 9: Checklist
    checklist_generator_submit: str = (
        "prompts/env_synthesis/step9_checklist/checklist_submit.md"
    )
    checklist_generator_db: str = (
        "prompts/env_synthesis/step9_checklist/checklist_db.md"
    )


@dataclass
class PromptsConfig:
    """Prompts configuration container."""

    data_synthesis: DataSynthesisPromptsConfig = field(
        default_factory=DataSynthesisPromptsConfig
    )
    env_synthesis: EnvSynthesisPromptsConfig = field(
        default_factory=EnvSynthesisPromptsConfig
    )


class Config:
    """
    Main configuration class.

    Provides access to all configuration values with support for:
    - Path template interpolation
    - Environment variable substitution
    - Nested attribute access
    """

    def __init__(self, config_dict: Dict[str, Any], project_root: Optional[str] = None):
        """
        Initialize configuration from a dictionary.

        Args:
            config_dict: Parsed YAML configuration
            project_root: Optional override for project root path
        """
        self._config = config_dict
        self._project_root = project_root or config_dict.get("global", {}).get(
            "project_root", "."
        )

        # Parse paths config
        paths_dict = config_dict.get("paths", {})
        paths_fields = _get_dataclass_field_names(PathsConfig)
        self.paths = PathsConfig(
            **{k: v for k, v in paths_dict.items() if k in paths_fields}
        )

        # Parse execution config
        exec_dict = config_dict.get("execution", {})
        exec_fields = _get_dataclass_field_names(ExecutionConfig)
        self.execution = ExecutionConfig(
            **{k: v for k, v in exec_dict.items() if k in exec_fields}
        )

        # Parse data_synthesis_stage config
        ds_dict = config_dict.get("data_synthesis_stage", {})
        self.data_synthesis = self._parse_data_synthesis_config(ds_dict)

        # Parse env_synthesis_stage config
        env_dict = config_dict.get("env_synthesis_stage", {})
        self.env_synthesis = self._parse_env_synthesis_config(env_dict)

        # Store raw configs for other sections
        self.llm_backends = config_dict.get("llm_backends", {})

        # Parse prompts config
        prompts_dict = config_dict.get("prompts", {})
        self.prompts = self._parse_prompts_config(prompts_dict)

    def _parse_prompts_config(self, prompts_dict: Dict[str, Any]) -> PromptsConfig:
        """Parse prompts configuration."""
        # Parse data_synthesis prompts
        ds_prompts_dict = prompts_dict.get("data_synthesis", {})
        ds_prompts_fields = _get_dataclass_field_names(DataSynthesisPromptsConfig)
        ds_prompts = DataSynthesisPromptsConfig(
            **{k: v for k, v in ds_prompts_dict.items() if k in ds_prompts_fields}
        )

        # Parse env_synthesis prompts
        env_prompts_dict = prompts_dict.get("env_synthesis", {})
        env_prompts_fields = _get_dataclass_field_names(EnvSynthesisPromptsConfig)
        env_prompts = EnvSynthesisPromptsConfig(
            **{k: v for k, v in env_prompts_dict.items() if k in env_prompts_fields}
        )

        return PromptsConfig(
            data_synthesis=ds_prompts,
            env_synthesis=env_prompts,
        )

    def _parse_data_synthesis_config(
        self, ds_dict: Dict[str, Any]
    ) -> DataSynthesisConfig:
        """Parse data synthesis stage configuration."""
        # Parse step1_graph
        step1_dict = ds_dict.get("step1_graph", {})
        if "num_nodes_range" in step1_dict:
            step1_dict["num_nodes_range"] = tuple(step1_dict["num_nodes_range"])
        step1_fields = _get_dataclass_field_names(Step1GraphConfig)
        step1 = Step1GraphConfig(
            **{k: v for k, v in step1_dict.items() if k in step1_fields}
        )

        # Parse step2_subtask
        step2_dict = ds_dict.get("step2_meta_subtask", {})
        step2_fields = _get_dataclass_field_names(Step2MetaSubtaskConfig)
        step2 = Step2MetaSubtaskConfig(
            **{k: v for k, v in step2_dict.items() if k in step2_fields}
        )

        # Parse step3_rawcalldoc
        step3_dict = ds_dict.get("step3_rawcalldoc", {})
        step3_fields = _get_dataclass_field_names(Step3RawCallDocConfig)
        step3 = Step3RawCallDocConfig(
            **{k: v for k, v in step3_dict.items() if k in step3_fields}
        )

        # Parse step4_merge
        step4_dict = ds_dict.get("step4_merge", {})
        step4_fields = _get_dataclass_field_names(Step4MergeConfig)
        step4 = Step4MergeConfig(
            **{k: v for k, v in step4_dict.items() if k in step4_fields}
        )

        # Parse step5_subtask
        step5_subtask_dict = ds_dict.get("step5_subtask", {})
        step5_subtask_fields = _get_dataclass_field_names(Step5SubtaskConfig)
        step5_subtask = Step5SubtaskConfig(
            **{k: v for k, v in step5_subtask_dict.items() if k in step5_subtask_fields}
        )

        # Parse step6_tscg
        step6_dict = ds_dict.get("step6_tscg", {})
        step6_fields = _get_dataclass_field_names(Step6TscgConfig)
        step6 = Step6TscgConfig(
            **{k: v for k, v in step6_dict.items() if k in step6_fields}
        )
        step7_dict = ds_dict.get("step7_review", {})
        step7_fields = _get_dataclass_field_names(Step7ReviewConfig)
        step7 = Step7ReviewConfig(
            **{k: v for k, v in step7_dict.items() if k in step7_fields}
        )

        return DataSynthesisConfig(
            step1_graph=step1,
            step2_meta_subtask=step2,
            step3_rawcalldoc=step3,
            step4_merge=step4,
            step5_subtask=step5_subtask,
            step6_tscg=step6,
            step7_review=step7,
        )

    def _parse_env_synthesis_config(
        self, env_dict: Dict[str, Any]
    ) -> EnvSynthesisConfig:
        """Parse environment synthesis stage configuration (6 steps in v0.4)."""
        # Parse step1_simudoc (domain-level)
        step1_dict = env_dict.get("step1_simudoc", {})
        step1_fields = _get_dataclass_field_names(Step1SimuDocConfig)
        step1 = Step1SimuDocConfig(
            **{k: v for k, v in step1_dict.items() if k in step1_fields}
        )

        # Parse step2_dbplan (task-level)
        step2_dict = env_dict.get("step2_dbplan", {})
        step2_fields = _get_dataclass_field_names(Step2DbPlanConfig)
        step2 = Step2DbPlanConfig(
            **{k: v for k, v in step2_dict.items() if k in step2_fields}
        )

        # Parse step3_toolscript (graph-level)
        step3_toolscript_dict = env_dict.get("step3_toolscript", {})
        step3_toolscript_fields = _get_dataclass_field_names(Step3ToolscriptConfig)
        step3_toolscript = Step3ToolscriptConfig(
            **{
                k: v
                for k, v in step3_toolscript_dict.items()
                if k in step3_toolscript_fields
            }
        )

        # Parse step4_initialdb (task-level)
        step4_dict = env_dict.get("step4_initialdb", {})
        step4_fields = _get_dataclass_field_names(Step4InitialDbConfig)
        step4 = Step4InitialDbConfig(
            **{k: v for k, v in step4_dict.items() if k in step4_fields}
        )

        # Parse step5_gtgraph (task-level)
        step5_dict = env_dict.get("step5_gtgraph", {})
        step5_fields = _get_dataclass_field_names(Step5GtGraphConfig)
        step5 = Step5GtGraphConfig(
            **{k: v for k, v in step5_dict.items() if k in step5_fields}
        )

        # Parse step6_review_refine (task-level)
        step6_dict = env_dict.get("step6_review_refine", {})
        step6_fields = _get_dataclass_field_names(Step6ReviewRefineConfig)
        step6 = Step6ReviewRefineConfig(
            **{k: v for k, v in step6_dict.items() if k in step6_fields}
        )

        # Parse step7_noise_injection (task-level)
        step7_dict = env_dict.get("step7_noise_injection", {})
        # Parse nested similarity_thresholds config
        if "similarity_thresholds" in step7_dict:
            sim_thresh_dict = step7_dict["similarity_thresholds"]
            sim_thresh_fields = _get_dataclass_field_names(SimilarityThresholdsConfig)
            step7_dict["similarity_thresholds"] = SimilarityThresholdsConfig(
                **{k: v for k, v in sim_thresh_dict.items() if k in sim_thresh_fields}
            )
        step7_fields = _get_dataclass_field_names(Step7NoiseInjectionConfig)
        step7 = Step7NoiseInjectionConfig(
            **{k: v for k, v in step7_dict.items() if k in step7_fields}
        )

        # Parse step8_adduser (task-level)
        step8_dict = env_dict.get("step8_adduser", {})
        step8_fields = _get_dataclass_field_names(Step8AddUserConfig)
        step8 = Step8AddUserConfig(
            **{k: v for k, v in step8_dict.items() if k in step8_fields}
        )

        # Parse step9_checklist (task-level)
        step9_dict = env_dict.get("step9_checklist", {})
        step9_fields = _get_dataclass_field_names(Step9ChecklistConfig)
        step9 = Step9ChecklistConfig(
            **{k: v for k, v in step9_dict.items() if k in step9_fields}
        )

        return EnvSynthesisConfig(
            step1_simudoc=step1,
            step2_dbplan=step2,
            step3_toolscript=step3_toolscript,
            step4_initialdb=step4,
            step5_gtgraph=step5,
            step6_review_refine=step6,
            step7_noise_injection=step7,
            step8_adduser=step8,
            step9_checklist=step9,
        )

    @property
    def project_root(self) -> str:
        """Get project root directory."""
        return self._project_root

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., "llm_backends.deepseek_main.model")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default

        return value

    def format_path(self, template: str, **kwargs) -> str:
        """
        Format a path template with variable substitution.

        Supports:
        - {paths.xxx} - Reference to paths config
        - {domain_id} - Domain identifier
        - Other kwargs passed to the method

        Args:
            template: Path template string
            **kwargs: Additional variables for substitution

        Returns:
            Formatted path string
        """
        # Build substitution context
        context = {"paths": self.paths, **kwargs}

        # Pattern to match {paths.xxx} or {variable}
        pattern = re.compile(r"\{([^}]+)\}")

        def replace_var(match: re.Match) -> str:
            var_path = match.group(1)
            parts = var_path.split(".")

            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, match.group(0))
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return match.group(0)  # Keep original if not found

            return str(value)

        return pattern.sub(replace_var, template)

    def get_llm_config_for_step(self, backend_name: str) -> Dict[str, Any]:
        """
        Get LLM configuration for a specific backend (for steps).

        Args:
            backend_name: Backend name

        Returns:
            LLM backend configuration dict with api_config if available

        Raises:
            ConfigError: If backend not found
        """
        if not backend_name:
            raise ConfigError("backend_name is required")

        backend_config = self.llm_backends.get(backend_name)
        if not backend_config:
            raise ConfigError(f"LLM backend not found: {backend_name}")

        # Get api_key_list from config
        api_key_list = self._config.get("api_key_list", [])

        # Check if backend has base_url (new auto-matching format)
        if "base_url" in backend_config and api_key_list:
            # Auto-match API keys based on base_url
            backend_base_url = backend_config.get("base_url")

            # Find all API keys with matching base_url
            selected_keys = [
                key for key in api_key_list if key.get("base_url") == backend_base_url
            ]

            if selected_keys:
                # Build api_config list from auto-matched keys, splitting
                # bypass keys (content-safety exempt) from regular ones.
                processed_api_configs = []
                bypass_api_configs = []
                for key_info in selected_keys:
                    api_config = {
                        "model": backend_config.get("model", "deepseek-chat"),
                        "base_url": key_info.get("base_url"),
                        "api_key": key_info.get("api_key"),
                        "rpm": backend_config.get("rpm", 50),
                    }
                    # Substitute environment variables in API key
                    if api_config["api_key"] and api_config["api_key"].startswith("${"):
                        env_var = api_config["api_key"][2:-1]
                        api_config["api_key"] = os.environ.get(
                            env_var, api_config["api_key"]
                        )
                    if key_info.get("bypass"):
                        bypass_api_configs.append(api_config)
                    else:
                        processed_api_configs.append(api_config)

                # Create result config
                result_config = {k: v for k, v in backend_config.items()}
                result_config["api_config"] = processed_api_configs
                result_config["bypass_api_config"] = bypass_api_configs
                result_config["_has_api_config"] = True
                return result_config

        # No api_key_list, use backend config directly
        backend_config["_has_api_config"] = False
        return backend_config

    def get_llm_config(self, backend_name: str) -> Dict[str, Any]:
        """
        Get LLM configuration for a specific role.

        Args:
            backend_name: LLM Backend name (generator, judge)

        Returns:
            LLM backend configuration dict with api_config if available

        Raises:
            ConfigError: If role or backend not found
        """

        backend_config = self.llm_backends.get(backend_name)
        if not backend_config:
            raise ConfigError(f"LLM backend not found: {backend_name}")

        # Get model from role config (new format)
        model = backend_config.get("model")
        # Get temperature from role config, fallback to backend or default
        temperature = backend_config.get("temperature", 0.7)

        # Get api_key_list from config
        api_key_list = self._config.get("api_key_list", [])

        # Check if backend has base_url (new auto-matching format)
        if "base_url" in backend_config and api_key_list:
            # Auto-match API keys based on base_url
            backend_base_url = backend_config.get("base_url")

            # Find all API keys with matching base_url
            selected_keys = [
                key for key in api_key_list if key.get("base_url") == backend_base_url
            ]

            if not selected_keys:
                raise ConfigError(
                    f"No API keys found for backend '{backend_name}' with base_url '{backend_base_url}'. "
                    f"Please add API keys with matching base_url to api_key_list."
                )

            # Build api_config list from auto-matched keys, splitting
            # bypass keys (content-safety exempt) from regular ones.
            processed_api_configs = []
            bypass_api_configs = []
            for key_info in selected_keys:
                api_config = {
                    "model": model
                    or backend_config.get(
                        "model", "deepseek-chat"
                    ),  # Use model from role
                    "base_url": key_info.get("base_url"),
                    "api_key": key_info.get("api_key"),
                    "rpm": backend_config.get("rpm", 50),
                }
                # Substitute environment variables in API key
                if api_config["api_key"] and api_config["api_key"].startswith("${"):
                    env_var = api_config["api_key"][2:-1]
                    api_config["api_key"] = os.environ.get(
                        env_var, api_config["api_key"]
                    )
                if key_info.get("bypass"):
                    bypass_api_configs.append(api_config)
                else:
                    processed_api_configs.append(api_config)

            # Create result config
            result_config = {k: v for k, v in backend_config.items()}
            result_config["api_config"] = processed_api_configs
            result_config["bypass_api_config"] = bypass_api_configs
            result_config["temperature"] = temperature  # Add temperature from role
            result_config["_has_api_config"] = True
            return result_config
        else:
            # Legacy format: no api_config or api_key_pool, use backend config directly
            # Substitute environment variables in API key
            if "api_key" in backend_config:
                api_key = backend_config["api_key"]
                if api_key and api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    backend_config["api_key"] = os.environ.get(env_var, api_key)
            backend_config["_has_api_config"] = False
            return backend_config

    def get_agent(
        self, backend_name: Optional[str] = None, role: str = "generator"
    ) -> "Agent":
        """
        Create an Agent for a specific backend or role.

        Args:
            backend_name: Backend name (if specified, use this backend)
            role: Role name (used if backend_name is not specified)

        Returns:
            Configured Agent instance

        Raises:
            ConfigError: If backend or role not found
        """
        from .agent import Agent

        # Get LLM config
        if backend_name:
            llm_config = self.get_llm_config_for_step(backend_name=backend_name)
        else:
            llm_config = self.get_llm_config(role)

        # Add execution config
        llm_config["max_retries"] = self.execution.max_retries
        llm_config["retry_delay"] = self.execution.retry_delay
        llm_config["json_parse_retries"] = self.execution.json_parse_retries

        # Create Agent
        return Agent(config=llm_config, role_name=backend_name or role)


def load_config(config_path: Union[str, Path] = "config.yaml") -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config instance

    Raises:
        ConfigError: If file not found or invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")

    if not isinstance(config_dict, dict):
        raise ConfigError("Configuration must be a YAML mapping")

    # Determine project root from config file location
    project_root = str(config_path.parent.absolute())

    return Config(config_dict, project_root=project_root)
