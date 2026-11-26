"""SimHash 局部敏感哈希与转载事件簇去重（对应文档第 8 题与八股 SimHash）。

算法细节：
1. 标题规范化：去掉“_网站名”、“|频道名”、“- 媒体名”及标点空白，标题相同直接归入同簇。
2. 64 位字符二元组（bigram）SimHash：
   - 取正文前 400 字（去除多余空白与标点噪音），按相邻 2 字切分特征；
   - 每个特征用 blake2b(digest_size=8) 算确定性 64 位哈希（避免 Python hash() 进程随机化）；
   - 按位投票（该位为 1 加权重，为 0 减权重），最终每位 > 0 记 1，否则记 0；
   - 汉明距离 <= 10 判定为近似转载同一事件簇。
3. 原始来源抽取：提取正文中的“来源：XX”，同一原始发布方也标记为同源。
"""

from __future__ import annotations

import hashlib
import re

# 剥离标题末尾常见的站点/频道后缀，如 "_新浪财经"、"| 东方财富网"、" - 财联社"
_TITLE_SUFFIX_RE = re.compile(r"(?:\s*[_\-|｜—–]\s*[^_\-|｜—–]{2,20})+$")
# 仅保留中英文和数字用于标题比对
_NON_ALNUM_ZH_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
# 抽取正文中的“来源：证券时报”或“来源: 财联社”
_ORIGINAL_SOURCE_RE = re.compile(
    r"(?:来源|转自|出处)\s*[:：]\s*([\u4e00-\u9fffA-Za-z0-9]{2,20})"
)


def normalize_title(title: str) -> str:
    """规范化新闻标题：去掉站点后缀与标点符号，转小写。"""
    stripped = title.strip()
    stripped = _TITLE_SUFFIX_RE.sub("", stripped)
    return _NON_ALNUM_ZH_RE.sub("", stripped).lower()


def extract_original_source(text: str) -> str | None:
    """从正文抽取原始发布方（“来源：XX”）。"""
    match = _ORIGINAL_SOURCE_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _hash64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def compute_simhash64(text: str, *, max_chars: int = 400) -> int:
    """计算正文前 400 字的 64 位字符二元组 SimHash 指纹。"""
    cleaned = _NON_ALNUM_ZH_RE.sub("", text)[:max_chars].lower()
    if not cleaned:
        return 0

    if len(cleaned) == 1:
        features = [cleaned]
    else:
        features = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]

    # 统计词频作为投票权重
    weights: dict[str, int] = {}
    for feat in features:
        weights[feat] = weights.get(feat, 0) + 1

    bits = [0] * 64
    for feat, weight in weights.items():
        h = _hash64(feat)
        for i in range(64):
            if (h >> i) & 1:
                bits[i] += weight
            else:
                bits[i] -= weight

    fingerprint = 0
    for i, score in enumerate(bits):
        if score > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(fp1: int, fp2: int) -> int:
    """计算两个 64 位指纹的汉明距离。"""
    return (fp1 ^ fp2).bit_count()
