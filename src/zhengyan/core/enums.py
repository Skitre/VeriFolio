"""核心枚举定义：证据分级、主张类型、核验状态、研究档位与治理枚举。"""

from enum import StrEnum


class EvidenceGrade(StrEnum):
    """证据等级（对应文档第 6 题）。"""

    A = "A"
    B = "B"
    C = "C"


class EvidenceSourceType(StrEnum):
    FINANCIAL_REPORT = "financial_report"
    MARKET_QUOTE = "market_quote"
    ANNOUNCEMENT = "announcement"
    COMPUTE = "compute"
    WEB = "web"


class PITGuarantee(StrEnum):
    """时点保证级别（对应文档第 10 题已知局限）。"""

    STRICT = "strict"
    WEAK_RESTATED = "weak_restated"


class AccountingBasis(StrEnum):
    """财务指标口径，防止跨口径硬算同比。"""

    GAAP = "gaap"
    NON_GAAP = "non_gaap"
    CUMULATIVE = "cumulative"
    SINGLE_QUARTER = "single_quarter"


class MetricUnit(StrEnum):
    """统一量纲，供计算 Skill 与核验器按同类指标匹配。"""

    CNY = "元"
    TEN_THOUSAND_CNY = "万元"
    HUNDRED_MILLION_CNY = "亿元"
    PERCENT = "%"
    PERCENTAGE_POINT = "个百分点"
    MULTIPLE = "倍"
    SHARES = "股"


class ClaimType(StrEnum):
    """主张类型（对应文档第 3 题与第 7 题）。"""

    FACT = "fact"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    SCENARIO = "scenario"


class VerificationStatus(StrEnum):
    """规则核验器输出状态（对应文档第 7 题）。"""

    VERIFIED = "verified"
    DOUBTFUL = "doubtful"
    REJECTED = "rejected"


# ==================== Stage 2 新增治理与编排枚举 ====================


class ResearchTier(StrEnum):
    """任务轴研究深度档位（对应文档第 4、5 题）。"""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class UserIntent(StrEnum):
    """用户意图四分类（对应文档第 1.1、3、26 题）。"""

    RESEARCH = "research"
    CONCEPT = "concept"
    TRADE_ORDER = "trade_order"
    OTHER = "other"


class ApprovalLevel(StrEnum):
    """Harness Skill 审批三档（对应文档第 11、11.2 题）。"""

    AUTO = "auto"
    CONFIRM = "confirm"
    FORBID = "forbid"


class QualityGateAction(StrEnum):
    """质量门控动作（对应文档第 16 题与第 5 题升级检查点）。"""

    PASS = "pass"
    RETRY_EVIDENCE = "retry_evidence"
    PARTIAL = "partial"
    ESCALATE_DEEP = "escalate_deep"


class DossierStatus(StrEnum):
    """提交档案的完整性状态。"""

    COMPLETE = "complete"
    PARTIAL = "partial"
    ABORTED = "aborted"


# ==================== Stage 4 新增多专家争议深研枚举 ====================


class ExpertRole(StrEnum):
    """6 位可选专业投研专家角色（主席每次最多召集 5 位 + 1 位反证官）。"""

    FUNDAMENTAL = "fundamental"  # 基本面研究员
    INDUSTRY = "industry"        # 行业/产业链研究员
    VALUATION = "valuation"      # 估值研究员（模型轴绑定 reason）
    TECHNICAL = "technical"      # 技术与资金面研究员
    EVENT = "event"              # 事件/新闻研究员
    RISK = "risk"                # 风险审阅研究员


class ChallengeSeverity(StrEnum):
    """反证官质询严重程度。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChallengeResponseAction(StrEnum):
    """专家面对质询时的回应动作。"""

    CONCEDE = "concede"  # 承认原主张/推断站不住脚
    DEFEND = "defend"    # 补充证据维持原主张
    NARROW = "narrow"    # 收窄前提条件，改为条件化判断

