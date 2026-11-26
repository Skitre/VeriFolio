# VeriFolio（溯研）

**基于 Harness Engineering 的可信投研多 Agent 系统**

> 不预测涨跌、不给买卖指令——让每一次研究得出的结论都**查得到出处、对得上口径、留得住分歧、随新信息增量更新**。

---

## 核心特性

- **A/B/C 证据分级与纯规则核验（Stage 1）**：财报、公告、行情与 `Decimal` 派生计算为 A 级事实，网页按权威度与全文快照分 B/C 级；严格按披露日（`disclosed_at <= as_of`）过滤防前视偏差，SimHash 转载聚类去重，纯规则核验器保证 **C 级线索进入结论 0 次**。
- **Harness 安全控制台与 LangGraph 固定状态图（Stage 2）**：固定 `intake → authorize → route → plan → collect → search → draft → verify → quality_gate → commit` 链路；权限白名单、`action.*` 全局禁令、Fail-Closed 前置检查与隐私脱敏、600 秒参数哈希审批票据、同任务幂等、指数退避重试，以及原生 `interrupt()` / `Command(resume=...)` 检查点续跑。
- **双轴路由与概率校准（Stage 3）**：任务轴（`quick / standard / deep`）与模型轴（`config/models.yaml`）解耦；离线温度缩放（Temperature Scaling / ECE / NLL）+ `< 0.55` 向重档位拒判升级，守住 **深研召回 100%** 与交易意图 **20/20 零误报**。
- **多专家争议深研子图（Stage 4）**：主席按缺口召集最多 5 位专家 + 1 位反证官，LangGraph `Send` 并行扇出并**首轮物理隔离**，反证官质询 + 主席按去重独立来源**非投票裁决**，高危未决争议提交前 `interrupt` 人工复核，配套纯代码综合质量分对照评测。
- **五类版本化记忆与前向模拟盘（Stage 5）**：强写入权限隔离 + 只追加决策档案版本链（`Claim → Evidence → ReportPeriod` 精确依赖图）、四类变更订阅扫描与增量复核、纯本地个人情境官、以及禁止历史回填的等权整手前向模拟盘与 5 条件晋级门控。
- **RSI 受控自进化闭环（Stage 6）**：不训练模型权重，仅迭代外围资产（S1 经验卡 / S2 路由参数 / S3 检索模板），通过**三权分立 + 六道纯代码门控 + 只追加三表事件溯源账本 + 人工批准/一键回滚**确保改进真实有效且可被下一轮继承。
- **System 1 混合判别、FastMCP 与控制台（Stage 7）**：本地零训练决策小模型 **Kev-0.8B** 支持 `llm / small / shadow` 三态开关与自动回退大模型（路由中位耗时 `2222 ms → 69 ms`、3 次零波动）；对外通过 FastMCP 仅暴露只读取数与计算工具，内置 FastAPI 服务与 5-Tab 控制台。

---

## 快速开始

> 说明：当前 Python 包路径保留为 `src/zhengyan`（与 7 阶段教程文档及 93 项测试保持一致），离线确定性运行无需独立显卡与外部 API Key。

```bash
# 1. 同步开发环境
uv sync --extra dev

# 2. 运行全量 93 项回归测试（约 1 秒）
uv run pytest -v

# 3. 逐阶段运行演示脚本（Stage 1 ~ Stage 7）
uv run python examples/stage1_demo.py
uv run python examples/stage6_demo.py
uv run python examples/stage7_demo.py

# 4. 可选：启动本地 FastAPI 服务与 5-Tab 控制台（http://127.0.0.1:8000）
uv sync --extra server
uv run uvicorn zhengyan.api.app:app --reload
```

---

## 分阶段学习与架构文档

完整架构图（10 张 Mermaid 流程图）、模块导读与面试高频 Q&A 见 [docs/README.md](./docs/README.md)：

1. [Stage 1：数据契约、A/B/C 证据链路与纯规则核验器](./docs/stage1_evidence_and_verifier.md)
2. [Stage 2：Harness 安全控制台与 LangGraph 10 节点固定状态图](./docs/stage2_harness_and_langgraph.md)
3. [Stage 3：双轴路由、温度缩放校准（ECE / NLL）与评测体系](./docs/stage3_dual_axis_routing_and_calibration.md)
4. [Stage 4：多专家争议深研子图与单 Agent 对照评测](./docs/stage4_multi_expert_deep_research.md)
5. [Stage 5：五类版本化记忆、依赖链增量复核与前向模拟盘](./docs/stage5_versioned_memory_and_paper_trading.md)
6. [Stage 6：RSI 受控自进化闭环、只追加账本与六道硬门控](./docs/stage6_rsi_self_improvement.md)
7. [Stage 7：Kev-0.8B 混合判别、FastMCP 只读服务与 FastAPI 控制台](./docs/stage7_kev_hybrid_mcp_and_api.md)
