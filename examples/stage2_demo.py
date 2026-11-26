"""Stage 2 端到端演示脚本：运行 `uv run python examples/stage2_demo.py` 查看完整效果。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.graph import ResearchRunner
from zhengyan.llm import MockLLMClient

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"


def main() -> None:
    as_of = date(2025, 8, 20)

    print("=== 演示 1：含个人隐私提问 → authorize 原地 interrupt 暂停 → Command(resume) 断点续跑 ===\n")
    llm = MockLLMClient(inject_hallucination=True)
    runner = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        llm_client=llm,
    )
    tid = "demo-pii-flow"
    question = "我有存款 30万，房贷每月 8000元，帮我复核贵州茅台 600519 半年报利润质量"

    runner.start(question, thread_id=tid)
    pending = runner.get_pending_interrupts(thread_id=tid)
    print(f"[暂停] 触发人在环审批：{pending[0]}")
    print(f"[暂停] 已执行节点：{runner.runtime.node_calls}\n")

    print("[续跑] 用户点击同意，使用 Command(resume={'approved': True}) 从检查点继续...")
    result = runner.resume({"approved": True}, thread_id=tid)
    dossier = result["dossier"]

    print(f"[完成] 全程执行节点：{runner.runtime.node_calls}（注意 intake 没有重跑！）")
    print(f"[档案] ID={dossier.dossier_id} v{dossier.version} | 档位={dossier.tier.value} | 状态={dossier.status.value}")
    print(f"[核验] 已进入结论主张={len(dossier.verified_claims)} 条，被拦截隔离主张={len(dossier.excluded_claims)} 条")
    for c in dossier.excluded_claims:
        print(f"       ↳ 拦截主张 {c.claim_id}: {c.text}")
    print(f"[脱敏] 发给 LLM 的最后一条 Prompt 首行：{llm.received_prompts[-1].splitlines()[0]}")
    print(f"[预算] {dossier.budget_snapshot}")
    print(f"[审计] Trace={runner.harness.audit.trace_id}，共记录 {len(runner.harness.audit.spans)} 条 Span\n")

    print("=== 演示 2：质量门控触发退回补证（1 轮上限） ===\n")
    runner2 = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        llm_client=MockLLMClient(simulate_gap_on_round_0=True),
    )
    res2 = runner2.start("复核贵州茅台半年报利润质量", thread_id="demo-retry-flow")
    print(f"[节点轨迹] {' -> '.join(runner2.runtime.node_calls)}")
    print(f"[结果] 补证轮次={res2['supplement_rounds']} | 最终状态={res2['dossier'].status.value}")


if __name__ == "__main__":
    main()
