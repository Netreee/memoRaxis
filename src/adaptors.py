# -*- coding: utf-8 -*-
"""三种推理范式适配器

- R1 SingleTurnAdaptor: 一次检索直接生成
- R2 IterativeAdaptor: 迭代式检索-判断循环
- R3 PlanAndActAdaptor: 先规划再执行，支持重规划
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .config import get_config
from .llm_interface import BaseLLMClient
from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence


@dataclass
class AdaptorResult:
    """适配器执行结果"""

    answer: str
    steps_taken: int = 0
    token_consumption: int = 0
    replan_count: int = 0
    evidence_collected: List[Evidence] = field(default_factory=list)


class BaseAdaptor(ABC):
    """适配器基类"""

    def __init__(self, llm_client: BaseLLMClient, memory_system: BaseMemorySystem):
        self._llm = llm_client
        self._memory = memory_system
        self._logger = get_logger()
        self._config = get_config()

    @abstractmethod
    def run(self, task: str) -> AdaptorResult:
        """执行推理任务

        Args:
            task: 任务描述

        Returns:
            AdaptorResult 包含答案和执行指标
        """
        pass

    def _format_evidence_list(self, evidences: List[Evidence]) -> str:
        """格式化证据列表"""
        if not evidences:
            return "暂无"

        lines = []
        for i, e in enumerate(evidences, 1):
            source = e.metadata.get("source", "未知来源")
            lines.append(f"- [{i}] {e.content} (来源: {source})")
        return "\n".join(lines)

    def _log_evidences(self, evidences: List[Evidence], context: str = "") -> None:
        """记录检索到的证据详情"""
        if not evidences:
            self._logger.debug("%s 无检索结果", context)
            return

        self._logger.info("%s 检索到 %d 条证据:", context, len(evidences))
        for i, e in enumerate(evidences, 1):
            score = e.metadata.get("score", "N/A")
            content_preview = e.content[:80] + "..." if len(e.content) > 80 else e.content
            self._logger.info("  [%d] (score=%.4f) %s", i, score if isinstance(score, float) else 0.0, content_preview)


class SingleTurnAdaptor(BaseAdaptor):
    """R1: 一次检索直接生成

    流程: run(task) → retrieve(task) → synthesize → answer
    """

    def run(self, task: str, top_k: int = 10) -> AdaptorResult:
        """执行单轮推理

        Args:
            task: 任务描述
            top_k: 检索结果数量

        Returns:
            AdaptorResult
        """
        self._logger.info("[SingleTurn] 开始处理任务: %s", task)

        # 步骤 1: 检索
        evidences = self._memory.retrieve(task, top_k=top_k)
        self._log_evidences(evidences, "[SingleTurn]")

        # 步骤 2: 综合生成
        prompt = self._config.get_prompt("single_turn", "synthesis").format(
            task=task, evidence_list=self._format_evidence_list(evidences)
        )
        answer = self._llm.generate(prompt)

        # 获取 Token 统计（如果 LLM 客户端支持）
        token_consumption = getattr(self._llm, "total_tokens", 0)

        self._logger.info("[SingleTurn] 任务完成")

        return AdaptorResult(
            answer=answer,
            steps_taken=1,
            token_consumption=token_consumption,
            evidence_collected=evidences,
        )


class IterativeAdaptor(BaseAdaptor):
    """R2: 迭代式检索-判断循环

    流程:
    run(task) → loop(max=5):
        decide(task, context) → ANSWER or search_query
        if ANSWER: break
        retrieve(search_query) → append context
    → synthesize → answer
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory_system: BaseMemorySystem,
        max_iterations: int = 5,
    ):
        super().__init__(llm_client, memory_system)
        self._max_iterations = max_iterations

    def run(self, task: str, top_k: int = 10) -> AdaptorResult:
        """执行迭代推理

        Args:
            task: 任务描述
            top_k: 每次检索结果数量

        Returns:
            AdaptorResult
        """
        self._logger.info("[Iterative] 开始处理任务: %s", task)

        all_evidences: List[Evidence] = []
        previous_queries: List[str] = []
        steps = 0

        for iteration in range(self._max_iterations):
            steps += 1
            self._logger.debug("[Iterative] 迭代 %d/%d", iteration + 1, self._max_iterations)

            # 格式化历史 query
            queries_str = "\n".join(f"- {q}" for q in previous_queries) if previous_queries else "暂无"

            # 构建决策 prompt
            decision_prompt = self._config.get_prompt("iterative", "decision").format(
                task=task,
                context=self._format_evidence_list(all_evidences),
                previous_queries=queries_str,
            )

            # 获取决策
            decision = self._llm.generate_json(decision_prompt)
            action = decision.get("action", "ANSWER")

            if action == "ANSWER":
                self._logger.info("[Iterative] 迭代 %d: 决策=ANSWER，信息足够", iteration + 1)
                break

            # 执行搜索
            query = decision.get("query", task)
            previous_queries.append(query)
            self._logger.info("[Iterative] 迭代 %d: 决策=SEARCH, query='%s'", iteration + 1, query)
            new_evidences = self._memory.retrieve(query, top_k=top_k)
            all_evidences.extend(new_evidences)
            self._log_evidences(new_evidences, f"[Iterative] 迭代 {iteration + 1}")

        # 综合生成
        synthesis_prompt = self._config.get_prompt("iterative", "synthesis").format(
            task=task, evidence_list=self._format_evidence_list(all_evidences)
        )
        answer = self._llm.generate(synthesis_prompt)

        token_consumption = getattr(self._llm, "total_tokens", 0)

        self._logger.info("[Iterative] 任务完成，共 %d 步", steps)

        return AdaptorResult(
            answer=answer,
            steps_taken=steps,
            token_consumption=token_consumption,
            evidence_collected=all_evidences,
        )


class PlanAndActAdaptor(BaseAdaptor):
    """R3: 两阶段动态规划 + 迭代检查 (Discovery → Expansion → Execute with Check)

    流程:
    Phase 1 (Discovery): 生成探索性步骤，获取概览信息
    Phase 2 (Expansion): 基于探索结果，生成原子化的细粒度计划
    Phase 3 (Execute + Check): 执行步骤，每步后检查是否需要调整
        - ANSWER: 提前结束
        - CONTINUE: 继续执行
        - ADD_STEPS: 补充新步骤
    Phase 4 (Synthesis): 综合所有证据生成答案
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory_system: BaseMemorySystem,
        max_expansion_steps: int = 6,
        max_additions: int = 2,
        check_interval: int = 1,
    ):
        super().__init__(llm_client, memory_system)
        self._max_expansion_steps = max_expansion_steps
        self._max_additions = max_additions
        self._check_interval = check_interval

    def run(self, task: str, top_k: int = 10) -> AdaptorResult:
        """执行两阶段动态规划推理（带迭代检查）

        Args:
            task: 任务描述
            top_k: 每次检索结果数量

        Returns:
            AdaptorResult
        """
        self._logger.info("[PlanAndAct] 开始处理任务: %s", task)

        all_evidences: List[Evidence] = []
        executed_steps: List[Dict[str, Any]] = []
        steps = 0
        additions_count = 0

        # ===== Phase 1: Discovery (探索阶段) =====
        self._logger.info("[PlanAndAct] Phase 1: Discovery")
        discovery_step = self._generate_discovery_step(task)
        self._logger.info("[PlanAndAct] 探索步骤: %s", discovery_step.get("description", ""))

        # 为探索步骤动态生成 query
        discovery_query = self._generate_query(task, discovery_step["description"], [])
        self._logger.info("[PlanAndAct] 探索 query: '%s'", discovery_query)

        # 执行探索检索
        discovery_evidences = self._memory.retrieve(discovery_query, top_k=top_k)
        all_evidences.extend(discovery_evidences)
        steps += 1
        self._log_evidences(discovery_evidences, "[PlanAndAct] Discovery")

        # ===== Phase 2: Expansion (扩展阶段) =====
        self._logger.info("[PlanAndAct] Phase 2: Expansion")
        plan = self._generate_expansion_plan(task, discovery_evidences)
        self._log_plan(plan, "[PlanAndAct] 扩展计划")

        # ===== Phase 3: Execute + Check (执行阶段，带迭代检查) =====
        self._logger.info("[PlanAndAct] Phase 3: Execute + Check")
        step_index = 0
        total_steps_limit = self._max_expansion_steps + self._max_additions * 2

        while step_index < len(plan) and steps < total_steps_limit:
            step = plan[step_index]
            steps += 1
            step_desc = step.get("description", "")

            # 动态生成 query
            query = self._generate_query(task, step_desc, all_evidences)
            self._logger.info(
                "[PlanAndAct] 步骤 %d/%d: %s (query='%s')",
                step_index + 1,
                len(plan),
                step_desc,
                query,
            )

            # 执行检索
            evidences = self._memory.retrieve(query, top_k=top_k)
            all_evidences.extend(evidences)
            self._log_evidences(evidences, f"[PlanAndAct] 步骤 {step_index + 1}")

            # 记录已执行步骤
            executed_steps.append(step)
            step_index += 1

            # 每 check_interval 步进行一次检查
            if step_index % self._check_interval == 0:
                remaining_steps = plan[step_index:]
                check_result = self._check_plan_progress(
                    task, plan, executed_steps, all_evidences, remaining_steps
                )
                action = check_result.get("action", "CONTINUE")

                if action == "ANSWER":
                    self._logger.info("[PlanAndAct] 检查结果: ANSWER，提前结束")
                    break
                elif action == "ADD_STEPS" and additions_count < self._max_additions:
                    new_steps = check_result.get("new_steps", [])
                    if new_steps:
                        additions_count += 1
                        # 为新步骤分配 step_id
                        for i, new_step in enumerate(new_steps):
                            new_step["step_id"] = len(plan) + i + 1
                        plan.extend(new_steps)
                        self._logger.info(
                            "[PlanAndAct] 检查结果: ADD_STEPS，补充 %d 个步骤 (第 %d 次补充)",
                            len(new_steps),
                            additions_count,
                        )
                        self._log_plan(new_steps, "[PlanAndAct] 补充步骤")
                else:
                    self._logger.debug("[PlanAndAct] 检查结果: CONTINUE")

        # ===== Phase 4: Synthesis (综合阶段) =====
        self._logger.info("[PlanAndAct] Phase 4: Synthesis")
        plan_summary = self._format_plan_summary(executed_steps, discovery_step)
        synthesis_prompt = self._config.get_prompt("plan_and_act", "synthesis").format(
            task=task,
            plan_summary=plan_summary,
            evidence_list=self._format_evidence_list(all_evidences),
        )
        answer = self._llm.generate(synthesis_prompt)

        token_consumption = getattr(self._llm, "total_tokens", 0)

        self._logger.info(
            "[PlanAndAct] 任务完成，共 %d 步，补充 %d 次", steps, additions_count
        )

        return AdaptorResult(
            answer=answer,
            steps_taken=steps,
            token_consumption=token_consumption,
            replan_count=additions_count,
            evidence_collected=all_evidences,
        )

    def _generate_discovery_step(self, task: str) -> Dict[str, Any]:
        """Phase 1: 生成探索性步骤"""
        prompt = self._config.get_prompt("plan_and_act", "discovery").format(task=task)
        result = self._llm.generate_json(prompt)
        return result.get("step", {"description": task})

    def _generate_expansion_plan(
        self, task: str, discovery_evidences: List[Evidence]
    ) -> List[Dict[str, Any]]:
        """Phase 2: 基于探索结果生成扩展计划"""
        prompt = self._config.get_prompt("plan_and_act", "expansion").format(
            task=task,
            discovery_evidence=self._format_evidence_list(discovery_evidences),
        )
        result = self._llm.generate_json(prompt)
        plan = result.get("plan", [])

        # 限制步骤数量
        if len(plan) > self._max_expansion_steps:
            self._logger.warning(
                "[PlanAndAct] 计划步骤过多 (%d)，截断至 %d",
                len(plan),
                self._max_expansion_steps,
            )
            plan = plan[: self._max_expansion_steps]

        return plan

    def _generate_query(
        self, task: str, step_description: str, context_evidences: List[Evidence]
    ) -> str:
        """动态生成检索 query"""
        prompt = self._config.get_prompt("plan_and_act", "query_generation").format(
            task=task,
            step_description=step_description,
            context=self._format_evidence_list(context_evidences),
        )
        result = self._llm.generate_json(prompt)
        return result.get("query", step_description)

    def _check_plan_progress(
        self,
        task: str,
        current_plan: List[Dict[str, Any]],
        executed_steps: List[Dict[str, Any]],
        all_evidences: List[Evidence],
        remaining_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """检查计划执行进度，决定下一步动作"""
        # 格式化当前计划
        plan_str = "\n".join(
            f"  步骤 {s.get('step_id', i+1)}: {s.get('description', '')}"
            for i, s in enumerate(current_plan)
        )
        # 格式化已执行步骤
        executed_str = "\n".join(
            f"  步骤 {s.get('step_id', i+1)}: {s.get('description', '')}"
            for i, s in enumerate(executed_steps)
        ) if executed_steps else "暂无"
        # 格式化剩余步骤
        remaining_str = "\n".join(
            f"  步骤 {s.get('step_id', '?')}: {s.get('description', '')}"
            for s in remaining_steps
        ) if remaining_steps else "暂无"

        prompt = self._config.get_prompt("plan_and_act", "plan_check").format(
            task=task,
            current_plan=plan_str,
            executed_steps=executed_str,
            evidence_list=self._format_evidence_list(all_evidences),
            remaining_steps=remaining_str,
        )
        result = self._llm.generate_json(prompt)
        return result

    def _log_plan(self, plan: List[Dict[str, Any]], context: str = "") -> None:
        """记录计划详情"""
        self._logger.info("%s，共 %d 步:", context, len(plan))
        for i, step in enumerate(plan):
            self._logger.info(
                "  步骤 %d: %s",
                step.get("step_id", i + 1),
                step.get("description", "无描述"),
            )

    def _format_plan_summary(
        self, executed_steps: List[Dict[str, Any]], discovery_step: Dict[str, Any]
    ) -> str:
        """格式化计划摘要（只包含已执行的步骤）"""
        lines = [f"- 探索: {discovery_step.get('description', '')}"]
        for i, step in enumerate(executed_steps):
            lines.append(f"- 步骤 {step.get('step_id', i + 1)}: {step.get('description', '')}")
        return "\n".join(lines)


# Helper Functions to simplify calls
def run_r1_single_turn(task: str, memory_system: BaseMemorySystem) -> tuple[str, dict]:
    """Helper to run R1 adaptor"""
    from .llm_interface import OpenAIClient
    from .config import get_config
    
    config = get_config()
    llm = OpenAIClient(config.llm["api_key"], config.llm["base_url"], config.llm["model"])
    adaptor = SingleTurnAdaptor(llm, memory_system)
    
    result = adaptor.run(task)
    
    meta = {
        "steps": result.steps_taken,
        "total_tokens": result.token_consumption
    }
    return result.answer, meta

def run_r2_iterative(task: str, memory_system: BaseMemorySystem) -> tuple[str, dict]:
    """Helper to run R2 adaptor"""
    from .llm_interface import OpenAIClient
    from .config import get_config
    
    config = get_config()
    llm = OpenAIClient(config.llm["api_key"], config.llm["base_url"], config.llm["model"])
    adaptor = IterativeAdaptor(llm, memory_system)
    
    result = adaptor.run(task)
    
    meta = {
        "steps": result.steps_taken,
        "total_tokens": result.token_consumption
    }
    return result.answer, meta

def run_r3_plan_act(task: str, memory_system: BaseMemorySystem) -> tuple[str, dict]:
    """Helper to run R3 adaptor (两阶段动态规划)"""
    from .llm_interface import OpenAIClient
    from .config import get_config

    config = get_config()
    llm = OpenAIClient(config.llm["api_key"], config.llm["base_url"], config.llm["model"])
    adaptor = PlanAndActAdaptor(llm, memory_system, max_expansion_steps=6)

    result = adaptor.run(task)

    meta = {
        "steps": result.steps_taken,
        "total_tokens": result.token_consumption,
    }
    return result.answer, meta
