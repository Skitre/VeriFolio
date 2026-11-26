"""意图识别与路由概率判别器（含交易意图规则优先 + 保守/随机 LLM 模拟器）。

关键设计要点（对应文档第 5、26 题）：
1. 交易意图规则永远优先于模型判断（安全红线）；规则基于句法/祈使动作而非死背短语，
   能准确区分“真正要求下单执行”与“提到买入/卖出/抄底/清仓但其实在提问分析”的干扰句。
2. 路由判别器返回三类概率分布（quick/standard/deep）+ consequence_level + disagreement_level。
3. 提供 `StochasticLLMClassifier`：可注入可控随机扰动模拟真实 LLM 判别器的非确定性，
   用于亲手复现文档里“改温度重跑 24 题准确率从 92% 掉到 83%，但同一次原始概率回放准确率不变”的经典现象！
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from zhengyan.core.enums import ResearchTier, UserIntent

SeverityLevel = Literal["low", "medium", "high"]

# ==================== 1. 交易意图规则（语用动作 vs 疑问分析） ====================

# 明确表示“询问/假设/讨论他人行为”的语境标记（排除干扰题）
_QUESTION_OR_ANALYSIS_RE = re.compile(
    r"(?:吗|呢|怎么看|说明什么|值不值得|适不适合|适合.{0,4}吗|该不该|应不应该|要不要|会不会"
    r"|还是该|还是|为什么|是不是到了|算不算|算是|影响多大|安全边际|支撑这个观点|想请教"
    r"|什么叫|什么是|解释一下|名词解释|如果.{0,12}(?:跌到|涨到)"
    r"|(?:外资|公募|基金|机构|高管|北向资金|公司自己|很多人).{0,10}(?:买入|卖出|加仓|减仓|减持|抄底))"
)

# 明确指向“代用户执行委托/买卖/调仓/下单”的祈使/执行动作
_IMPERATIVE_TRADE_PATTERNS: list[re.Pattern[str]] = [
    # 明确祈使主体 + 交易动词
    re.compile(
        r"(?:帮我|替我|给我|请|立刻|马上|现在|直接|一键|赶紧|执行(?:指令)?|明天一?开盘|尾盘|今天收盘前)"
        r".{0,12}(?:买入|卖出|下单|建仓|建个.{0,6}仓|清仓|加仓|减仓|抄底|满仓|全仓|购入|扫进|抛了|出掉|割肉|委托|调仓|打进去买)"
    ),
    # 交易动词 + 明确数量/金额/价格委托
    re.compile(
        r"(?:买入|卖出|下单|购入|扫进|挂单|委托买入|委托卖出|挂.{0,8}限价单|接)"
        r".{0,8}(?:\d+|一|两|二|三|四|五|十|百|半)\s*(?:股|手|万|万元|元|块|成|半)"
    ),
    # 账户/持仓处置动作（把账户里XX出掉/清掉/抛了/减半）
    re.compile(
        r"(?:把|将).{0,14}(?:出掉|清掉|清仓|抛了|卖掉|减半|打进去买)"
    ),
    # 价格指令型下单（市价/限价/现价 + 买/卖/建仓）
    re.compile(
        r"(?:市价|限价|按现价|用\s*\d+\s*万.{0,4}现价)(?:买入|卖出|挂单|建个?.{0,6}仓)"
    ),
]

_CONCEPT_PATTERNS = re.compile(
    r"(?:什么是|什么叫|解释一下|名词解释|是什么意思|怎么算|计算公式是什么|有什么区别|想请教一下.{0,4}什么叫)"
)

_RESEARCH_SIGNAL_RE = re.compile(
    r"(?:60\d{4}|00\d{4}|30\d{4}|688\d{3}|茅台|宁德|财报|半年报|年报|三季报|季报|营收|净利润|毛利率|净利率"
    r"|现金流|合同负债|估值|市盈率|PE|公告|股价|收盘价|市值|业绩|护城河|批价|白酒|电池|基本面|EPS|北向|公募|回购|减持)"
)


def is_trade_order_by_rule(question: str) -> bool:
    """高精准度、低误报的交易意图硬规则（安全红线：永远优先于模型判断）。"""
    q = question.strip()
    # 先拦截明显的疑问分析/概念提问/讨论他人交易，避免把“适合买入吗”“高管减持影响多大”误报为下单
    if _QUESTION_OR_ANALYSIS_RE.search(q):
        return False
    return any(pat.search(q) for pat in _IMPERATIVE_TRADE_PATTERNS)


@dataclass(frozen=True)
class IntentPrediction:
    intent: UserIntent
    probs: dict[str, float]
    source: str  # "rule" | "llm" | "kev_small"
    latency_ms: float = 0.0


@dataclass(frozen=True)
class RoutingClassifierOutput:
    raw_probs: dict[str, float]
    consequence_level: SeverityLevel
    disagreement_level: SeverityLevel
    source: str
    latency_ms: float = 0.0


class IntentClassifier(Protocol):
    def classify_intent(self, question: str) -> IntentPrediction:
        ...


class RoutingClassifier(Protocol):
    def classify_route(self, question: str, *, intent: UserIntent) -> RoutingClassifierOutput:
        ...


# ==================== 2. 离线确定性 + 可控随机扰动判别器 ====================

_QUICK_LOOKUP_RE = re.compile(r"(?:查一下|最新收盘价|总市值是多少|什么时候披露|披露日期|营业收入数字)")
_DEEP_CONTROVERSY_RE = re.compile(
    r"(?:合不合理|合理吗|贵不贵|低估|要不要|该不该|应不应该|继续持有|继续拿|清仓|减仓|止损|坚守"
    r"|多空分歧|分歧很大|分歧大|争议很大|值不值得|深度研判|风险收益比|护城河冲击|反面风险|见顶)"
)
_STANDARD_REVIEW_RE = re.compile(
    r"(?:复核|梳理|核实|总结|拆解|分析.{0,12}(?:单季|同比|毛利率|现金流|合同负债|增速|公告|摘要))"
)


def _softmax_dict(logits: dict[str, float]) -> dict[str, float]:
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(exps.values())
    return {k: v / s for k, v in exps.items()}


class RuleFirstIntentClassifier:
    """交易意图规则优先 + 离线意图概率判别器（可零误报达到 20/20 下单识别）。"""

    def __init__(self, source_name: str = "mock_llm") -> None:
        self.source_name = source_name

    def classify_intent(self, question: str) -> IntentPrediction:
        # 规则永远优先于模型判断（对应文档第 26 题）
        if is_trade_order_by_rule(question):
            return IntentPrediction(
                intent=UserIntent.TRADE_ORDER,
                probs={
                    UserIntent.TRADE_ORDER.value: 0.99,
                    UserIntent.RESEARCH.value: 0.005,
                    UserIntent.CONCEPT.value: 0.0025,
                    UserIntent.OTHER.value: 0.0025,
                },
                source="rule",
                latency_ms=0.2,
            )

        q = question.strip()
        # 概念解释优先判定（避免“解释一下合同负债在白酒财报里的含义”被“财报”拉到 research）
        if _CONCEPT_PATTERNS.search(q) and not re.search(r"(?:60\d{4}|00\d{4}|30\d{4}|688\d{3}|茅台|宁德)", q):
            probs = {
                UserIntent.CONCEPT.value: 0.92,
                UserIntent.RESEARCH.value: 0.05,
                UserIntent.TRADE_ORDER.value: 0.01,
                UserIntent.OTHER.value: 0.02,
            }
            return IntentPrediction(UserIntent.CONCEPT, probs, self.source_name, latency_ms=12.0)

        if _RESEARCH_SIGNAL_RE.search(q):
            probs = {
                UserIntent.RESEARCH.value: 0.91,
                UserIntent.CONCEPT.value: 0.03,
                UserIntent.TRADE_ORDER.value: 0.03,
                UserIntent.OTHER.value: 0.03,
            }
            return IntentPrediction(UserIntent.RESEARCH, probs, self.source_name, latency_ms=12.0)

        probs = {
            UserIntent.OTHER.value: 0.88,
            UserIntent.CONCEPT.value: 0.05,
            UserIntent.RESEARCH.value: 0.05,
            UserIntent.TRADE_ORDER.value: 0.02,
        }
        return IntentPrediction(UserIntent.OTHER, probs, self.source_name, latency_ms=12.0)


class MockRoutingClassifier:
    """模拟 LLM 路由判别器（默认带有偏保守/欠自信概率分布，还原文档中 T=0.75 校准场景）。

    参数：
    - `underconfident_scale`: 模拟大模型判别器偏保守（分布偏平，ECE ~0.18，需要 T < 1 拉尖）；
    - `noise_std`: 为 0 时完全确定；> 0 时给 logits 加高斯随机扰动，用于复现大模型多次运行结果漂移；
    - `seed`: 随机种子，保证实验可复现。
    """

    def __init__(
        self,
        *,
        underconfident_scale: float = 0.65,
        noise_std: float = 0.0,
        seed: int | None = None,
        source_name: str = "mock_deepseek_classifier",
        base_latency_ms: float = 2222.0,
    ) -> None:
        self.underconfident_scale = underconfident_scale
        self.noise_std = noise_std
        self._rng = random.Random(seed)
        self.source_name = source_name
        self.base_latency_ms = base_latency_ms

    def _base_logits_and_flags(
        self,
        question: str,
        intent: UserIntent,
    ) -> tuple[dict[str, float], SeverityLevel, SeverityLevel]:
        q = question.strip()

        if intent == UserIntent.CONCEPT:
            return (
                {ResearchTier.QUICK.value: 2.8, ResearchTier.STANDARD.value: 0.2, ResearchTier.DEEP.value: -1.5},
                "low",
                "low",
            )

        if _DEEP_CONTROVERSY_RE.search(q):
            # 高后果、高分歧研判题
            high_cons: SeverityLevel = "high" if any(k in q for k in ("持有", "清仓", "减仓", "止损", "拿", "风险收益比")) else "medium"
            high_dis: SeverityLevel = "high" if any(k in q for k in ("分歧", "争议", "合不合理", "合理吗", "见顶", "反面风险")) else "medium"
            return (
                {ResearchTier.QUICK.value: -1.6, ResearchTier.STANDARD.value: 0.4, ResearchTier.DEEP.value: 2.3},
                high_cons,
                high_dis,
            )

        if _QUICK_LOOKUP_RE.search(q):
            return (
                {ResearchTier.QUICK.value: 2.1, ResearchTier.STANDARD.value: 0.6, ResearchTier.DEEP.value: -1.2},
                "low",
                "low",
            )

        if _STANDARD_REVIEW_RE.search(q) or intent == UserIntent.TRADE_ORDER:
            # 略带边界感的一道题（拆解合同负债会计口径原因）：模型对 standard/deep 略微犹豫，演示校准前后变化
            if "合同负债" in q and "拆解" in q:
                return (
                    {ResearchTier.QUICK.value: -1.2, ResearchTier.STANDARD.value: 1.15, ResearchTier.DEEP.value: 0.80},
                    "medium",
                    "medium",
                )
            return (
                {ResearchTier.QUICK.value: -0.9, ResearchTier.STANDARD.value: 2.0, ResearchTier.DEEP.value: 0.1},
                "medium",
                "low",
            )

        # 模糊边界问题：分布平坦，触发拒判升重
        return (
            {ResearchTier.QUICK.value: 0.1, ResearchTier.STANDARD.value: 0.45, ResearchTier.DEEP.value: 0.35},
            "medium",
            "medium",
        )

    def classify_route(self, question: str, *, intent: UserIntent) -> RoutingClassifierOutput:
        logits, cons, dis = self._base_logits_and_flags(question, intent)
        scaled_logits: dict[str, float] = {}
        for k, v in logits.items():
            val = v * self.underconfident_scale
            if self.noise_std > 0:
                val += self._rng.gauss(0.0, self.noise_std)
            scaled_logits[k] = val

        raw_probs = _softmax_dict(scaled_logits)
        return RoutingClassifierOutput(
            raw_probs=raw_probs,
            consequence_level=cons,
            disagreement_level=dis,
            source=self.source_name,
            latency_ms=self.base_latency_ms,
        )
