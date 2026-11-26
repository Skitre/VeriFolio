"""Stage 4 端到端演示：茅台多专家对抗辩论 + 高危未决争议 interrupt 复核 + 3 例单 Agent 对照评测。

运行：uv run python examples/stage4_demo.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.deep import (
    MockDeepResearchEngine,
    evaluate_ab_comparison,
    load_ab_eval_cases,
)
from zhengyan.graph import ResearchRunner

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"
AB_CASES_PATH = ROOT / "evals" / "golden" / "ab_eval_cases.json"


def main() -> None:
    as_of = date(2025, 8, 20)

    print("=== 1. 贵州茅台争议深研全流程（LangGraph Send 并行 + 首轮隔离 + 反证官 + 非投票裁决） ===\n")
    runner = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
    )
    out = runner.start("贵州茅台 600519 合同负债下降 42%，估值合理吗，还要不要继续持有？", thread_id="demo-deep-mt")
    dossier = out["dossier"]
    adj = out["deep_adjudication"]

    print(f"  [执行轨迹] {' -> '.join(runner.runtime.node_calls)}")
    print(f"  [并行专家] {runner.runtime.deep_runtime.expert_executions}")
    print(f"  [裁决摘要] {adj.summary}")
    print(f"  [质询成立拦截] upheld={adj.upheld_challenge_ids}")
    for c in out["deep_excluded_claims"]:
        print(f"    ↳ 拦在结论外: {c.claim_id} ({c.author}) -> {c.text}")
    print(f"  [保留意见] 共 {len(adj.dissenting_opinions)} 条（不靠多数投票强行抹平）")
    for d in adj.dissenting_opinions:
        print(f"    ↳ {d.topic}: 反方={d.bear_or_counter_view[:48]}...")

    print("\n=== 2. 高严重度未决争议触发 LangGraph interrupt 人在环复核 ===\n")
    runner_hr = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        deep_engine=MockDeepResearchEngine(inject_unresolved_high_dispute=True),
    )
    tid = "demo-deep-hr"
    runner_hr.start("贵州茅台现在估值合理吗，还要不要继续持有", thread_id=tid)
    pending = runner_hr.get_pending_interrupts(thread_id=tid)
    print(f"  [暂停等待人工复核] type={pending[0]['type']} | 未决质询={pending[0]['unresolved_challenge_ids']}")
    resumed = runner_hr.resume({"approved": True}, thread_id=tid)
    print(f"  [人工确认后提交] 档案状态={resumed['dossier'].status.value}（按规则以“部分结论”提交）")

    print("\n=== 3. 单 Agent vs 多专家对照评测（3 例，隔离命名空间，纯代码综合质量分） ===\n")
    cases = load_ab_eval_cases(AB_CASES_PATH)
    rep = evaluate_ab_comparison(cases, policy_path=POLICY_PATH, search_sources_path=SEARCH_PATH)
    print(
        f"  单 Agent 均值：综合质量={rep.single_avg_quality:.4f} | "
        f"数字可复算率={rep.single_avg_recompute_rate*100:.1f}% | 平均 Token={rep.single_avg_tokens:.0f}"
    )
    print(
        f"  多专家   均值：综合质量={rep.multi_avg_quality:.4f} | "
        f"数字可复算率={rep.multi_avg_recompute_rate*100:.1f}% | 平均 Token={rep.multi_avg_tokens:.0f}"
    )
    rel_gain = (rep.quality_delta / rep.single_avg_quality) * 100 if rep.single_avg_quality else 0.0
    print(
        f"  对照结论     ：质量差={rep.quality_delta:+.4f}（相对提升 {rel_gain:+.1f}%）| "
        f"Token 倍数={rep.token_ratio:.2f}x | 准入判定={'通过' if rep.admitted else '拒绝'}"
    )


if __name__ == "__main__":
    main()
