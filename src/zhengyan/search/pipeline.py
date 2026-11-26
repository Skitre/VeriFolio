"""网页搜索过滤、转载聚类、白名单清洗与 B/C 证据分级流水线。

对应文档第 3、6、8、9、10 题完整链路：
搜索查询构造（自动追加 -site: 黑名单 & 时间窗口）
  → 过滤黑名单站点
  → 按站点权威度降序排序
  → 清洗 HTML 与疑似提示注入
  → 转载聚类（规范化标题相同 OR 前400字 SimHash 汉明距离 <= 10 OR 同原始发布方）
  → B / C 级判定（权威度 >= 3 AND 已抓取全文快照 AND 日期可知且 <= as_of → B 级；其余 C 级）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import yaml

from zhengyan.core.enums import EvidenceGrade, EvidenceSourceType
from zhengyan.core.models import EvidenceCard, RawWebResult
from zhengyan.search.sanitizer import sanitize_html
from zhengyan.search.simhash import (
    compute_simhash64,
    extract_original_source,
    hamming_distance,
    normalize_title,
)


@dataclass(frozen=True)
class SiteEntry:
    domain: str
    name: str
    authority: int


class SourcePolicy:
    """加载自 config/search_sources.yaml 的白名单权威度与黑名单策略。"""

    def __init__(
        self,
        whitelist: dict[str, SiteEntry],
        blacklist: set[str],
        *,
        min_b_grade_authority: int = 3,
        default_lookback_days: int = 120,
    ) -> None:
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.min_b_grade_authority = min_b_grade_authority
        self.default_lookback_days = default_lookback_days

    @classmethod
    def from_yaml(cls, path: Path) -> SourcePolicy:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        wl: dict[str, SiteEntry] = {}
        for domain, info in (data.get("whitelist") or {}).items():
            wl[domain.lower()] = SiteEntry(
                domain=domain.lower(),
                name=str(info["name"]),
                authority=int(info["authority"]),
            )
        bl = {str(d).lower() for d in (data.get("blacklist") or [])}
        return cls(
            whitelist=wl,
            blacklist=bl,
            min_b_grade_authority=int(data.get("min_b_grade_authority", 3)),
            default_lookback_days=int(data.get("default_search_lookback_days", 120)),
        )

    @staticmethod
    def extract_host(url: str) -> str:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host

    def is_blacklisted(self, url: str) -> bool:
        host = self.extract_host(url)
        return any(host == b or host.endswith(f".{b}") for b in self.blacklist)

    def match_whitelist(self, url: str) -> SiteEntry | None:
        host = self.extract_host(url)
        # 优先精确匹配，再匹配后缀子域名（最长的域名优先）
        if host in self.whitelist:
            return self.whitelist[host]
        matched = [
            entry for domain, entry in self.whitelist.items()
            if host.endswith(f".{domain}")
        ]
        if not matched:
            return None
        return max(matched, key=lambda e: len(e.domain))

    def build_safe_query(self, query: str, as_of: date, lookback_days: int | None = None) -> str:
        """构造带 `-site:` 黑名单排除与研究时点时间窗口的安全查询词。"""
        days = lookback_days if lookback_days is not None else self.default_lookback_days
        start = as_of - timedelta(days=days)
        exclusions = " ".join(f"-site:{domain}" for domain in sorted(self.blacklist))
        return f"{query} after:{start.isoformat()} before:{as_of.isoformat()} {exclusions}".strip()


@dataclass
class _ClusterRep:
    cluster_id: str
    norm_title: str
    simhash: int
    original_source: str | None


class WebSearchPipeline:
    """将原始网页搜索结果转化为已去重、已清洗、已分级的 EvidenceCard 列表。"""

    def __init__(
        self,
        policy: SourcePolicy,
        *,
        simhash_threshold: int = 10,
    ) -> None:
        self.policy = policy
        self.simhash_threshold = simhash_threshold

    def process(
        self,
        raw_results: list[RawWebResult],
        *,
        as_of: date,
        symbol: str | None = None,
        lookback_days: int | None = None,
    ) -> list[EvidenceCard]:
        # 1. 过滤黑名单站点
        filtered: list[tuple[int, int, RawWebResult, SiteEntry | None]] = []
        for idx, item in enumerate(raw_results):
            if self.policy.is_blacklisted(item.url):
                continue
            entry = self.policy.match_whitelist(item.url)
            authority = entry.authority if entry is not None else 0
            filtered.append((authority, idx, item, entry))

        # 2. 按权威度降序排序（确保每个事件簇的代表是最权威的站点）
        filtered.sort(key=lambda x: (-x[0], x[1]))

        max_window = lookback_days if lookback_days is not None else self.policy.default_lookback_days
        earliest_allowed = as_of - timedelta(days=max_window)

        clusters: list[_ClusterRep] = []
        cards: list[EvidenceCard] = []

        for seq, (authority, _, item, entry) in enumerate(filtered, start=1):
            # 仅白名单站点允许保留抓取正文快照；非白名单只保留摘要
            allow_snapshot = entry is not None and item.fetched and bool(item.raw_html)
            raw_body = item.raw_html if allow_snapshot and item.raw_html else item.snippet
            sanitized = sanitize_html(raw_body)

            norm_title = normalize_title(item.title)
            fp = compute_simhash64(sanitized.cleaned_text or item.title)
            orig_src = extract_original_source(sanitized.cleaned_text)

            # 3. 事件簇聚类：规范化标题相同 OR SimHash 汉明距离 <= 10 OR 同原始发布方且指纹接近
            cluster_id: str | None = None
            for rep in clusters:
                same_title = bool(norm_title) and norm_title == rep.norm_title
                near_dup = fp != 0 and rep.simhash != 0 and hamming_distance(fp, rep.simhash) <= self.simhash_threshold
                same_orig_source = (
                    orig_src is not None
                    and rep.original_source is not None
                    and orig_src == rep.original_source
                )
                if same_title or near_dup or same_orig_source:
                    cluster_id = rep.cluster_id
                    break

            if cluster_id is None:
                cluster_id = f"CLU-{len(clusters) + 1:03d}"
                clusters.append(
                    _ClusterRep(
                        cluster_id=cluster_id,
                        norm_title=norm_title,
                        simhash=fp,
                        original_source=orig_src,
                    )
                )

            # 4. 证据分级（严格对应文档第 6 题）：
            # B 级必须同时满足：权威度 >= 3 + 已抓全文快照 + 日期可知且不晚于 as_of（且在窗口内）
            date_valid = (
                item.published_at is not None
                and earliest_allowed <= item.published_at <= as_of
            )
            if (
                authority >= self.policy.min_b_grade_authority
                and allow_snapshot
                and date_valid
            ):
                grade = EvidenceGrade.B
            else:
                grade = EvidenceGrade.C

            source_name = entry.name if entry else self.policy.extract_host(item.url)
            cards.append(
                EvidenceCard(
                    evidence_id=f"EV-WEB-{seq:03d}",
                    grade=grade,
                    source_type=EvidenceSourceType.WEB,
                    source_name=source_name,
                    title=item.title,
                    symbol=symbol,
                    published_at=item.published_at,
                    url=item.url,
                    authority=authority,
                    has_full_snapshot=allow_snapshot,
                    excerpt=sanitized.cleaned_text[:600],
                    simhash=fp,
                    event_cluster_id=cluster_id,
                    original_source=orig_src,
                    suspected_injection=sanitized.suspected_injection,
                    stripped_injection_lines=sanitized.stripped_lines,
                )
            )

        return cards
