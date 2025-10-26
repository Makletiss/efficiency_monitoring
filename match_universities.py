"""University matching script."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import regex
from rapidfuzz import distance, fuzz, process

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_LIMIT = 50
TOP_M = 10
MANUAL_DELTA = 2.0
CONFLICT_DELTA = 1.0

STOP_WORDS = {"№"}
QUOTE_CHARS = '"\'«»“”„‟‹›'
BRACKETS = "()[]{}"
ELLIPSIS_PATTERN = regex.compile(r"\.\.\.|…")
DASH_PATTERN = regex.compile(r"[–—―]")
MULTI_DASH_PATTERN = regex.compile(r"[-–—―]{2,}")
SPACE_PATTERN = regex.compile(r"[\s\u00A0\u2000-\u200F\u202F\u205F\u3000]+")
TAIL_PATTERNS = [
    regex.compile(r"\bим\.[\s\-]+.+$", regex.IGNORECASE),
    regex.compile(r"\bимени[\s\-]+.+$", regex.IGNORECASE),
    regex.compile(r"\((?:главный\s+кампус|главный\s+корпус)\)$", regex.IGNORECASE),
    regex.compile(r"\bфилиал(?:\s+в\s+г\.\s*[^,;]+|\s+в\s+[^,;]+|\s+№\s*\d+)?$", regex.IGNORECASE),
]
BRANCH_PATTERN = regex.compile(r"\bфилиал(?:[ауеом]?|\b)\b", regex.IGNORECASE)
BRANCH_TAIL_PATTERN = regex.compile(r"филиал\b.*$", regex.IGNORECASE)
TOKEN_SPLIT_PATTERN = regex.compile(r"[\s\-\u2010-\u2015]+")
UNIQUE_TOKEN_MIN_LEN = 6


@dataclass
class SourceRecord:
    record_id: str
    name: str
    normalized_name: str
    city: Optional[str]
    region: Optional[str]
    inn: Optional[str]
    ogrn: Optional[str]
    aliases: List[str]
    normalized_aliases: List[str]
    tokens: List[str]
    token_set: set
    initials: set
    prefixes2: set
    prefixes3: set
    is_branch: bool


@dataclass
class TargetRecord:
    index: int
    name: str
    normalized_name: str
    city: Optional[str]
    region: Optional[str]
    inn: Optional[str]
    ogrn: Optional[str]
    aliases: List[str]
    normalized_aliases: List[str]
    tokens: List[str]
    token_set: set
    initials: set
    prefixes2: set
    prefixes3: set
    likely_branch: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    source: SourceRecord
    candidate_variant: str
    display_name: str
    score_token_set: float
    score_partial: float
    score_wratio: float
    score_lcs: float
    score_final: float
    city_match: bool
    region_match: bool
    id_match: bool
    likely_branch: bool
    penalties: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    target: TargetRecord
    match_type: str
    candidate: Optional[SourceRecord]
    candidate_name: Optional[str]
    candidate_id: Optional[str]
    score_token_set: Optional[float]
    score_partial: Optional[float]
    score_wratio: Optional[float]
    score_lcs: Optional[float]
    score_final: Optional[float]
    city_match: bool = False
    region_match: bool = False
    likely_branch: bool = False
    notes: List[str] = field(default_factory=list)
    ranked_candidates: List[CandidateScore] = field(default_factory=list)


@dataclass
class BlockingIndex:
    initials: Dict[str, set]
    prefixes2: Dict[str, set]
    prefixes3: Dict[str, set]
    city: Dict[str, set]
    region: Dict[str, set]
    inn: Dict[str, set]
    ogrn: Dict[str, set]
    alias_exact: Dict[str, set]
    all_ids: set


class MatchingError(Exception):
    pass


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def normalize_simple(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value)
    if not value_str.strip():
        return None
    value_str = unicodedata.normalize("NFC", value_str)
    value_str = SPACE_PATTERN.sub(" ", value_str).strip()
    return value_str.casefold() if value_str else None


def split_aliases(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    parts = [part.strip() for part in str(value).split(";")]
    return [part for part in parts if part]


def clean_digits(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    digits = regex.sub(r"\D", "", str(value))
    return digits or None


def _strip_stop_words(value: str) -> str:
    tokens = value.split()
    while tokens and tokens[0] in STOP_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in STOP_WORDS:
        tokens.pop()
    return " ".join(tokens)


def detect_branch(text: str) -> bool:
    if not text:
        return False
    return bool(BRANCH_PATTERN.search(text) or BRANCH_TAIL_PATTERN.search(text))


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = [token for token in TOKEN_SPLIT_PATTERN.split(text) if token]
    return tokens


def build_feature_sets(tokens: Sequence[str]) -> Tuple[set, set, set]:
    initials = {token[0] for token in tokens if token}
    prefixes2 = {token[:2] for token in tokens if len(token) >= 2}
    prefixes3 = {token[:3] for token in tokens if len(token) >= 3}
    return initials, prefixes2, prefixes3


def normalize_name(value: str) -> Tuple[str, List[str]]:
    notes: List[str] = []
    original = str(value) if value is not None else ""
    if not original:
        return "", notes
    text = unicodedata.normalize("NFC", original)
    text = text.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    text = SPACE_PATTERN.sub(" ", text).strip()
    text = text.translate({ord(ch): None for ch in QUOTE_CHARS + BRACKETS})
    text = ELLIPSIS_PATTERN.sub("", text)
    text = DASH_PATTERN.sub("-", text)
    text = MULTI_DASH_PATTERN.sub("-", text)
    text = SPACE_PATTERN.sub(" ", text).strip()
    for tail_pattern in TAIL_PATTERNS:
        new_text = tail_pattern.sub("", text)
        if new_text != text:
            text = new_text.strip()
    text = SPACE_PATTERN.sub(" ", text).strip()
    text = _strip_stop_words(text)
    text = SPACE_PATTERN.sub(" ", text).strip()
    normalized = text.casefold()
    if not normalized:
        fallback = SPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", original)).strip()
        notes.append("name_became_empty_safeguard")
        return fallback, notes
    return normalized, notes

def load_table(
    path: Path,
    sheet: Optional[str],
    id_col: str,
    name_col: str,
    city_col: str,
    region_col: str,
    inn_col: str,
    ogrn_col: str,
    alias_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise MatchingError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        read_kwargs = {"dtype": str, "keep_default_na": False}
        if sheet:
            read_kwargs["sheet_name"] = sheet
        df = pd.read_excel(path, **read_kwargs)
    else:
        raise MatchingError(f"Unsupported file format: {path.suffix}")
    columns_needed = {name_col}
    if id_col:
        columns_needed.add(id_col)
    for col in [city_col, region_col, inn_col, ogrn_col, alias_col]:
        if col:
            columns_needed.add(col)
    missing = [col for col in columns_needed if col not in df.columns]
    if missing:
        raise MatchingError(f"Missing columns in {path}: {missing}")
    return df


def prepare_source_records(
    df: pd.DataFrame,
    id_col: str,
    name_col: str,
    city_col: str,
    region_col: str,
    inn_col: str,
    ogrn_col: str,
    alias_col: str,
) -> Tuple[List[SourceRecord], List[str]]:
    records: List[SourceRecord] = []
    duplicate_logs: List[str] = []
    for _, row in df.iterrows():
        raw_id = str(row[id_col]) if id_col in row and str(row[id_col]).strip() else ""
        name_raw = str(row[name_col]) if name_col in row else ""
        normalized_name, notes = normalize_name(name_raw)
        aliases_raw = split_aliases(row.get(alias_col)) if alias_col in row else []
        normalized_aliases = []
        alias_notes: List[str] = []
        for alias in aliases_raw:
            normalized_alias, alias_note = normalize_name(alias)
            normalized_aliases.append(normalized_alias)
            alias_notes.extend(alias_note)
        city = normalize_simple(row.get(city_col)) if city_col in row else None
        region = normalize_simple(row.get(region_col)) if region_col in row else None
        inn = clean_digits(row.get(inn_col)) if inn_col in row else None
        ogrn = clean_digits(row.get(ogrn_col)) if ogrn_col in row else None
        tokens = tokenize(normalized_name)
        initials, prefixes2, prefixes3 = build_feature_sets(tokens)
        alias_tokens = [token for alias in normalized_aliases for token in tokenize(alias)]
        alias_initials, alias_prefixes2, alias_prefixes3 = build_feature_sets(alias_tokens)
        combined_tokens = list(dict.fromkeys(tokens + alias_tokens))
        record = SourceRecord(
            record_id=raw_id,
            name=name_raw,
            normalized_name=normalized_name,
            city=city,
            region=region,
            inn=inn,
            ogrn=ogrn,
            aliases=aliases_raw,
            normalized_aliases=normalized_aliases,
            tokens=combined_tokens,
            token_set=set(combined_tokens),
            initials=initials.union(alias_initials),
            prefixes2=prefixes2.union(alias_prefixes2),
            prefixes3=prefixes3.union(alias_prefixes3),
            is_branch=detect_branch(normalized_name),
        )
        if notes or alias_notes:
            duplicate_logs.extend([f"note:{raw_id}:{n}" for n in notes + alias_notes])
        records.append(record)
    deduped, dup_logs = deduplicate_sources(records)
    duplicate_logs.extend(dup_logs)
    return deduped, duplicate_logs


def completeness_score(record: SourceRecord) -> Tuple[int, int, str]:
    filled = sum(bool(field) for field in [record.inn, record.ogrn, record.city, record.region, record.aliases])
    return (filled, len(record.normalized_name), record.record_id)


def choose_best_record(a: SourceRecord, b: SourceRecord) -> SourceRecord:
    score_a = completeness_score(a)
    score_b = completeness_score(b)
    if score_a > score_b:
        return a
    if score_b > score_a:
        return b
    return a if a.record_id <= b.record_id else b


def deduplicate_sources(records: List[SourceRecord]) -> Tuple[List[SourceRecord], List[str]]:
    logs: List[str] = []
    by_name: Dict[str, SourceRecord] = {}
    intermediate: List[SourceRecord] = []
    for record in records:
        key = record.normalized_name
        if not key:
            intermediate.append(record)
            continue
        if key not in by_name:
            by_name[key] = record
        else:
            best = choose_best_record(by_name[key], record)
            dropped = record if best is by_name[key] else by_name[key]
            logs.append(f"duplicate_name:{key}:{dropped.record_id}->{best.record_id}")
            by_name[key] = best
    intermediate.extend(by_name.values())
    final_records: List[SourceRecord] = []
    key_to_index: Dict[Tuple[str, str], int] = {}
    for record in intermediate:
        candidate = record
        duplicate_found = False
        for key_name, value in (("inn", record.inn), ("ogrn", record.ogrn)):
            if not value:
                continue
            key = (key_name, value)
            if key in key_to_index:
                duplicate_found = True
                idx = key_to_index[key]
                existing = final_records[idx]
                best = choose_best_record(existing, candidate)
                dropped = candidate if best is existing else existing
                logs.append(f"duplicate_{key_name}:{value}:{dropped.record_id}->{best.record_id}")
                final_records[idx] = best
                candidate = best
            else:
                key_to_index[key] = len(final_records)
        if not duplicate_found:
            idx = len(final_records)
            final_records.append(candidate)
            for key_name, value in (("inn", candidate.inn), ("ogrn", candidate.ogrn)):
                if value:
                    key_to_index[(key_name, value)] = idx
    return final_records, logs


def prepare_target_records(
    df: pd.DataFrame,
    name_col: str,
    city_col: str,
    region_col: str,
    inn_col: str,
    ogrn_col: str,
    alias_col: str,
) -> List[TargetRecord]:
    records: List[TargetRecord] = []
    for idx, row in df.iterrows():
        name_raw = str(row[name_col]) if name_col in row else ""
        normalized_name, notes = normalize_name(name_raw)
        aliases_raw = split_aliases(row.get(alias_col)) if alias_col in row else []
        normalized_aliases = []
        for alias in aliases_raw:
            norm_alias, alias_notes = normalize_name(alias)
            normalized_aliases.append(norm_alias)
            notes.extend(alias_notes)
        city = normalize_simple(row.get(city_col)) if city_col in row else None
        region = normalize_simple(row.get(region_col)) if region_col in row else None
        inn = clean_digits(row.get(inn_col)) if inn_col in row else None
        ogrn = clean_digits(row.get(ogrn_col)) if ogrn_col in row else None
        tokens = tokenize(normalized_name)
        initials, prefixes2, prefixes3 = build_feature_sets(tokens)
        alias_tokens = [token for alias in normalized_aliases for token in tokenize(alias)]
        alias_initials, alias_prefixes2, alias_prefixes3 = build_feature_sets(alias_tokens)
        combined_tokens = list(dict.fromkeys(tokens + alias_tokens))
        record = TargetRecord(
            index=idx,
            name=name_raw,
            normalized_name=normalized_name,
            city=city,
            region=region,
            inn=inn,
            ogrn=ogrn,
            aliases=aliases_raw,
            normalized_aliases=normalized_aliases,
            tokens=combined_tokens,
            token_set=set(combined_tokens),
            initials=initials.union(alias_initials),
            prefixes2=prefixes2.union(alias_prefixes2),
            prefixes3=prefixes3.union(alias_prefixes3),
            likely_branch=detect_branch(normalized_name),
            notes=notes,
        )
        records.append(record)
    return records


def build_blocking_index(records: Sequence[SourceRecord]) -> BlockingIndex:
    initials_index: Dict[str, set] = defaultdict(set)
    prefixes2_index: Dict[str, set] = defaultdict(set)
    prefixes3_index: Dict[str, set] = defaultdict(set)
    city_index: Dict[str, set] = defaultdict(set)
    region_index: Dict[str, set] = defaultdict(set)
    inn_index: Dict[str, set] = defaultdict(set)
    ogrn_index: Dict[str, set] = defaultdict(set)
    alias_exact: Dict[str, set] = defaultdict(set)
    all_ids: set = set()
    for idx, record in enumerate(records):
        all_ids.add(idx)
        for initial in record.initials:
            initials_index[initial].add(idx)
        for pref in record.prefixes2:
            prefixes2_index[pref].add(idx)
        for pref in record.prefixes3:
            prefixes3_index[pref].add(idx)
        if record.city:
            city_index[record.city].add(idx)
        if record.region:
            region_index[record.region].add(idx)
        if record.inn:
            inn_index[record.inn].add(idx)
        if record.ogrn:
            ogrn_index[record.ogrn].add(idx)
        alias_exact[record.normalized_name].add(idx)
        for alias in record.normalized_aliases:
            alias_exact[alias].add(idx)
    return BlockingIndex(
        initials=initials_index,
        prefixes2=prefixes2_index,
        prefixes3=prefixes3_index,
        city=city_index,
        region=region_index,
        inn=inn_index,
        ogrn=ogrn_index,
        alias_exact=alias_exact,
        all_ids=all_ids,
    )

def exact_match(target: TargetRecord, sources: Sequence[SourceRecord], index: BlockingIndex) -> Optional[MatchResult]:
    notes = list(target.notes)

    def make_result(src_idx: int, match_type: str, note: str) -> MatchResult:
        source = sources[src_idx]
        return MatchResult(
            target=target,
            match_type=match_type,
            candidate=source,
            candidate_name=source.name,
            candidate_id=source.record_id,
            score_token_set=100.0,
            score_partial=100.0,
            score_wratio=100.0,
            score_lcs=100.0,
            score_final=100.0,
            city_match=bool(target.city and target.city == source.city),
            region_match=bool(target.region and target.region == source.region),
            likely_branch=target.likely_branch,
            notes=notes + [note],
        )

    candidate_indices: set = set()
    if target.inn:
        candidate_indices |= index.inn.get(target.inn, set())
    if target.ogrn:
        candidate_indices |= index.ogrn.get(target.ogrn, set())
    if candidate_indices:
        if len(candidate_indices) == 1:
            src_idx = next(iter(candidate_indices))
            return make_result(src_idx, "exact_by_id", "match_by_id")
        else:
            return MatchResult(
                target=target,
                match_type="conflict",
                candidate=None,
                candidate_name=None,
                candidate_id=None,
                score_token_set=None,
                score_partial=None,
                score_wratio=None,
                score_lcs=None,
                score_final=None,
                city_match=False,
                region_match=False,
                likely_branch=target.likely_branch,
                notes=notes + ["multiple_id_candidates"],
                ranked_candidates=[
                    CandidateScore(
                        source=sources[idx],
                        candidate_variant=sources[idx].normalized_name,
                        display_name=sources[idx].name,
                        score_token_set=100.0,
                        score_partial=100.0,
                        score_wratio=100.0,
                        score_lcs=100.0,
                        score_final=100.0,
                        city_match=bool(target.city and target.city == sources[idx].city),
                        region_match=bool(target.region and target.region == sources[idx].region),
                        id_match=True,
                        likely_branch=target.likely_branch,
                        bonuses=["id_match"],
                    )
                    for idx in candidate_indices
                ],
            )
    normalized_name = target.normalized_name
    if normalized_name in index.alias_exact:
        candidates = list(index.alias_exact[normalized_name])
        if len(candidates) == 1:
            return make_result(candidates[0], "exact", "normalized_name_match")
        else:
            return MatchResult(
                target=target,
                match_type="conflict",
                candidate=None,
                candidate_name=None,
                candidate_id=None,
                score_token_set=None,
                score_partial=None,
                score_wratio=None,
                score_lcs=None,
                score_final=None,
                city_match=False,
                region_match=False,
                likely_branch=target.likely_branch,
                notes=notes + ["multiple_exact_candidates"],
                ranked_candidates=[
                    CandidateScore(
                        source=sources[idx],
                        candidate_variant=sources[idx].normalized_name,
                        display_name=sources[idx].name,
                        score_token_set=100.0,
                        score_partial=100.0,
                        score_wratio=100.0,
                        score_lcs=100.0,
                        score_final=100.0,
                        city_match=bool(target.city and target.city == sources[idx].city),
                        region_match=bool(target.region and target.region == sources[idx].region),
                        id_match=False,
                        likely_branch=target.likely_branch,
                    )
                    for idx in candidates
                ],
            )
    candidates_name = list(index.alias_exact.get(normalized_name, [])) if normalized_name else []
    if target.city and normalized_name:
        matches_city = [idx for idx in candidates_name if sources[idx].city == target.city]
        if len(matches_city) == 1:
            return make_result(matches_city[0], "exact", "name_city_match")
    if target.region and normalized_name:
        matches_region = [idx for idx in candidates_name if sources[idx].region == target.region]
        if len(matches_region) == 1:
            return make_result(matches_region[0], "exact", "name_region_match")
    return None


def block_candidates(
    target: TargetRecord,
    sources: Sequence[SourceRecord],
    index: BlockingIndex,
    limit: int = DEFAULT_LIMIT,
) -> List[int]:
    candidate_ids: set = set()
    if target.inn:
        candidate_ids |= index.inn.get(target.inn, set())
    if target.ogrn:
        candidate_ids |= index.ogrn.get(target.ogrn, set())
    narrowed = False
    narrowed_candidates: set = set()
    if target.city and target.city in index.city:
        narrowed_candidates |= index.city[target.city]
        narrowed = True
    if not narrowed_candidates and target.region and target.region in index.region:
        narrowed_candidates |= index.region[target.region]
        narrowed = True
    if narrowed_candidates:
        candidate_ids |= narrowed_candidates
    if not candidate_ids:
        candidate_ids = set(index.all_ids)
    for alias in target.normalized_aliases + [target.normalized_name]:
        candidate_ids |= index.alias_exact.get(alias, set())
    for initial in target.initials:
        candidate_ids |= index.initials.get(initial, set())
    for prefix in target.prefixes2:
        candidate_ids |= index.prefixes2.get(prefix, set())
    for prefix in target.prefixes3:
        candidate_ids |= index.prefixes3.get(prefix, set())
    filtered: List[int] = []
    for idx in candidate_ids:
        source = sources[idx]
        if narrowed and target.city and source.city and target.city != source.city:
            continue
        if narrowed and target.region and source.region and target.region != source.region:
            continue
        if target.initials and source.initials and not (target.initials & source.initials):
            continue
        if target.prefixes2 and source.prefixes2 and not (target.prefixes2 & source.prefixes2):
            continue
        if target.prefixes3 and source.prefixes3 and not (target.prefixes3 & source.prefixes3):
            continue
        if target.normalized_name and source.normalized_name:
            len_a = len(target.normalized_name)
            len_b = len(source.normalized_name)
            if len_a and len_b:
                diff = abs(len_a - len_b) / max(len_a, len_b)
                if diff > 0.6 and not (
                    (target.city and source.city and target.city == source.city)
                    or (target.region and source.region and target.region == source.region)
                ):
                    continue
        filtered.append(idx)
    return filtered[:limit]


def compute_unique_token_penalty(
    target_tokens: set,
    candidate_tokens: set,
    source_token_freq: Dict[str, int],
) -> bool:
    for token in target_tokens:
        if len(token) < UNIQUE_TOKEN_MIN_LEN:
            continue
        if token not in candidate_tokens and source_token_freq.get(token, 0) == 1:
            return True
    return False


def score_pair(
    target: TargetRecord,
    candidate: SourceRecord,
    candidate_variant: str,
    source_token_freq: Dict[str, int],
    scores: Dict[str, float],
) -> CandidateScore:
    tsr = scores.get("tsr", 0.0)
    pr = scores.get("pr", 0.0)
    wr = scores.get("wr", 0.0)
    lcs = scores.get("lcs", 0.0)
    s_base = 0.45 * tsr + 0.30 * pr + 0.20 * wr + 0.05 * lcs
    bonuses: List[str] = []
    penalties: List[str] = []
    city_match = bool(target.city and candidate.city and target.city == candidate.city)
    region_match = bool(target.region and candidate.region and target.region == candidate.region)
    id_match = bool(
        (target.inn and candidate.inn and target.inn == candidate.inn)
        or (target.ogrn and candidate.ogrn and target.ogrn == candidate.ogrn)
    )
    if id_match:
        s_base += 5
        bonuses.append("id_match")
    if city_match:
        s_base += 3
        bonuses.append("city_match")
    if region_match:
        s_base += 2
        bonuses.append("region_match")
    if compute_unique_token_penalty(target.token_set, candidate.token_set, source_token_freq):
        s_base -= 4
        penalties.append("unique_token_missing")
    if target.likely_branch and not candidate.is_branch:
        s_base -= 2
        penalties.append("branch_mismatch")
    len_a = len(target.normalized_name)
    len_b = len(candidate_variant)
    if len_a and len_b:
        diff = abs(len_a - len_b) / max(len_a, len_b)
        if diff > 0.5 and not (city_match or region_match):
            s_base -= 3
            penalties.append("length_mismatch")
    return CandidateScore(
        source=candidate,
        candidate_variant=candidate_variant,
        display_name=candidate.name,
        score_token_set=tsr,
        score_partial=pr,
        score_wratio=wr,
        score_lcs=lcs,
        score_final=s_base,
        city_match=city_match,
        region_match=region_match,
        id_match=id_match,
        likely_branch=target.likely_branch,
        penalties=penalties,
        bonuses=bonuses,
    )

def fuzzy_match(
    target: TargetRecord,
    sources: Sequence[SourceRecord],
    index: BlockingIndex,
    source_token_freq: Dict[str, int],
) -> MatchResult:
    candidate_indices = block_candidates(target, sources, index)
    variants: List[Tuple[int, str, str]] = []
    for idx in candidate_indices:
        record = sources[idx]
        variants.append((idx, record.normalized_name, record.name))
        for alias_norm, alias_orig in zip(record.normalized_aliases, record.aliases):
            variants.append((idx, alias_norm, alias_orig))
    if not variants:
        return MatchResult(
            target=target,
            match_type="not_found",
            candidate=None,
            candidate_name=None,
            candidate_id=None,
            score_token_set=None,
            score_partial=None,
            score_wratio=None,
            score_lcs=None,
            score_final=None,
            likely_branch=target.likely_branch,
            notes=list(target.notes) + ["no_candidates_after_blocking"],
        )
    variant_strings = [variant[1] for variant in variants]
    tsr_scores = process.cdist([target.normalized_name], variant_strings, scorer=fuzz.token_set_ratio, workers=1)[0]
    pr_scores = process.cdist([target.normalized_name], variant_strings, scorer=fuzz.partial_ratio, workers=1)[0]
    prelim = [0.5 * tsr + 0.5 * pr for tsr, pr in zip(tsr_scores, pr_scores)]
    top_indices = sorted(range(len(variants)), key=lambda i: prelim[i], reverse=True)[: min(TOP_M, len(variants))]
    wr_scores = [0.0] * len(variants)
    lcs_scores = [0.0] * len(variants)
    for idx in top_indices:
        candidate_text = variant_strings[idx]
        wr_scores[idx] = fuzz.WRatio(target.normalized_name, candidate_text, processor=None)
        lcs_scores[idx] = 100 * distance.LCSseq.normalized_similarity(target.normalized_name, candidate_text)
    candidate_scores: Dict[int, CandidateScore] = {}
    for i, (src_idx, variant_text, _) in enumerate(variants):
        scores_dict = {
            "tsr": float(tsr_scores[i]),
            "pr": float(pr_scores[i]),
            "wr": float(wr_scores[i]),
            "lcs": float(lcs_scores[i]),
        }
        candidate_score = score_pair(target, sources[src_idx], variant_text, source_token_freq, scores_dict)
        existing = candidate_scores.get(src_idx)
        if existing is None or candidate_score.score_final > existing.score_final:
            candidate_scores[src_idx] = candidate_score
    ranked = sorted(candidate_scores.values(), key=lambda cs: cs.score_final, reverse=True)
    if not ranked:
        return MatchResult(
            target=target,
            match_type="not_found",
            candidate=None,
            candidate_name=None,
            candidate_id=None,
            score_token_set=None,
            score_partial=None,
            score_wratio=None,
            score_lcs=None,
            score_final=None,
            likely_branch=target.likely_branch,
            notes=list(target.notes) + ["no_ranked_candidates"],
        )
    best = ranked[0]
    match_type = "fuzzy"
    notes = list(target.notes)
    if best.score_final < 86:
        match_type = "not_found"
        notes.append("below_threshold")
        return MatchResult(
            target=target,
            match_type=match_type,
            candidate=best.source,
            candidate_name=best.display_name,
            candidate_id=best.source.record_id,
            score_token_set=best.score_token_set,
            score_partial=best.score_partial,
            score_wratio=best.score_wratio,
            score_lcs=best.score_lcs,
            score_final=best.score_final,
            city_match=best.city_match,
            region_match=best.region_match,
            likely_branch=best.likely_branch,
            notes=notes,
            ranked_candidates=ranked[:3],
        )
    high_candidates = [cand for cand in ranked if cand.score_final >= 92]
    conflict_candidates = [cand for cand in ranked if cand.score_final >= 92 and cand.score_final - best.score_final > -CONFLICT_DELTA]
    if high_candidates and len(conflict_candidates) > 1:
        resolved = resolve_conflicts(conflict_candidates)
        if resolved is None:
            return MatchResult(
                target=target,
                match_type="conflict",
                candidate=None,
                candidate_name=None,
                candidate_id=None,
                score_token_set=None,
                score_partial=None,
                score_wratio=None,
                score_lcs=None,
                score_final=None,
                city_match=False,
                region_match=False,
                likely_branch=best.likely_branch,
                notes=notes + ["unresolved_high_conflict"],
                ranked_candidates=ranked[:3],
            )
        best = resolved
    if best.score_final >= 92 or (best.score_token_set >= 94 and best.score_partial >= 90) or best.id_match:
        match_type = "fuzzy"
    elif best.score_final >= 86:
        notes.append("manual_review")
        match_type = "conflict"
    second = ranked[1] if len(ranked) > 1 else None
    if second and abs(best.score_final - second.score_final) < MANUAL_DELTA and match_type != "conflict":
        notes.append("close_second_candidate")
        match_type = "conflict"
    return MatchResult(
        target=target,
        match_type=match_type,
        candidate=best.source if match_type != "conflict" else None,
        candidate_name=best.display_name if match_type != "conflict" else None,
        candidate_id=best.source.record_id if match_type != "conflict" else None,
        score_token_set=best.score_token_set if match_type != "conflict" else None,
        score_partial=best.score_partial if match_type != "conflict" else None,
        score_wratio=best.score_wratio if match_type != "conflict" else None,
        score_lcs=best.score_lcs if match_type != "conflict" else None,
        score_final=best.score_final if match_type != "conflict" else None,
        city_match=best.city_match if match_type != "conflict" else False,
        region_match=best.region_match if match_type != "conflict" else False,
        likely_branch=best.likely_branch,
        notes=notes,
        ranked_candidates=ranked[:3],
    )


def resolve_conflicts(candidates: List[CandidateScore]) -> Optional[CandidateScore]:
    if not candidates:
        return None

    def tie_key(candidate: CandidateScore) -> Tuple[int, int, int, float, float]:
        return (
            0 if candidate.id_match else 1,
            0 if candidate.city_match else 1,
            0 if candidate.region_match else 1,
            -candidate.score_token_set,
            -candidate.score_wratio,
        )

    sorted_candidates = sorted(candidates, key=tie_key)
    best = sorted_candidates[0]
    if len(sorted_candidates) == 1:
        return best
    second = sorted_candidates[1]
    if tie_key(best) == tie_key(second):
        return None
    return best

def match_all(
    targets: Sequence[TargetRecord],
    sources: Sequence[SourceRecord],
    index: BlockingIndex,
    source_token_freq: Dict[str, int],
    logger: logging.Logger,
) -> List[MatchResult]:
    results: List[MatchResult] = []
    for target in targets:
        exact_result = exact_match(target, sources, index)
        if exact_result:
            logger.debug("Exact match for target %s: %s", target.name, exact_result.match_type)
            results.append(exact_result)
            continue
        fuzzy_result = fuzzy_match(target, sources, index, source_token_freq)
        logger.debug(
            "Fuzzy match for target %s: %s (%s)",
            target.name,
            fuzzy_result.match_type,
            fuzzy_result.candidate_id,
        )
        results.append(fuzzy_result)
    return results


def save_outputs(
    target_df: pd.DataFrame,
    results: Sequence[MatchResult],
    out_path: Path,
    review_path: Path,
) -> None:
    matched_rows = []
    review_rows = []
    for result in results:
        row = target_df.iloc[result.target.index].to_dict()
        row.update(
            {
                "id": result.candidate_id,
                "match_type": result.match_type,
                "score_token_set": result.score_token_set,
                "score_partial": result.score_partial,
                "score_wratio": result.score_wratio,
                "score_lcs": result.score_lcs,
                "score_final": result.score_final,
                "candidate_name": result.candidate_name,
                "candidate_id": result.candidate_id,
                "city_match": result.city_match,
                "region_match": result.region_match,
                "likely_branch": result.likely_branch,
                "notes": "|".join(result.notes) if result.notes else "",
            }
        )
        matched_rows.append(row)
        if result.match_type in {"conflict", "not_found"}:
            for candidate in result.ranked_candidates:
                review_rows.append(
                    {
                        "target_index": result.target.index,
                        "target_name": result.target.name,
                        "match_type": result.match_type,
                        "candidate_id": candidate.source.record_id,
                        "candidate_name": candidate.display_name,
                        "score_token_set": candidate.score_token_set,
                        "score_partial": candidate.score_partial,
                        "score_wratio": candidate.score_wratio,
                        "score_lcs": candidate.score_lcs,
                        "score_final": candidate.score_final,
                        "city_match": candidate.city_match,
                        "region_match": candidate.region_match,
                        "likely_branch": candidate.likely_branch,
                        "bonuses": "|".join(candidate.bonuses),
                        "penalties": "|".join(candidate.penalties),
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matched_rows).to_csv(out_path, index=False)
    pd.DataFrame(review_rows).to_csv(review_path, index=False)


def collect_token_statistics(sources: Sequence[SourceRecord]) -> Dict[str, int]:
    token_counter: Counter = Counter()
    for record in sources:
        for token in set(record.tokens):
            token_counter[token] += 1
    return dict(token_counter)


def report_metrics(
    logger: logging.Logger,
    results: Sequence[MatchResult],
    durations: Dict[str, float],
    duplicate_logs: Sequence[str],
) -> None:
    counts = Counter(result.match_type for result in results)
    total = sum(counts.values()) or 1
    logger.info("Match counts: %s", json.dumps(counts, ensure_ascii=False))
    logger.info(
        "Match shares: %s",
        json.dumps({k: round(v / total, 4) for k, v in counts.items()}, ensure_ascii=False),
    )
    for name, duration in durations.items():
        logger.info("Duration %s: %.3fs", name, duration)
    if duplicate_logs:
        logger.info("Potential duplicates in source: %s", "; ".join(duplicate_logs))
    conflict_tokens: Counter = Counter()
    conflict_bigrams: Counter = Counter()
    for result in results:
        if result.match_type not in {"conflict", "not_found"}:
            continue
        tokens = [token for token in result.target.token_set if len(token) >= 3 and token not in STOP_WORDS]
        conflict_tokens.update(tokens)
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            conflict_bigrams[bigram] += 1
    logger.info(
        "Top tokens in conflict/not_found: %s",
        json.dumps(conflict_tokens.most_common(10), ensure_ascii=False),
    )
    logger.info(
        "Top bigrams in conflict/not_found: %s",
        json.dumps([(" ".join(b), c) for b, c in conflict_bigrams.most_common(10)], ensure_ascii=False),
    )


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("university_matcher")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    return logger


def run_selftests() -> None:
    # Test normalize_name retains key words and removes tails
    examples = {
        "Московский государственный технический университет им. Н.Э. Баумана": "московский государственный технический университет",
        "Национальный исследовательский институт": "национальный исследовательский институт",
        "Колледж искусств ...": "колледж искусств",
        """"СПбГУ""": "спбгу",
    }
    for raw, expected in examples.items():
        normalized, notes = normalize_name(raw)
        assert normalized == expected, f"normalize_name failed for {raw}: {normalized}"  # noqa: S101
        if any(word in expected for word in ("университет", "институт", "колледж")):
            assert any(word in normalized for word in ("университет", "институт", "колледж")), "keywords missing"  # noqa: S101
        assert "name_became_empty_safeguard" not in notes, "unexpected safeguard"  # noqa: S101
    branch_text = "Филиал Московского университета в г. Тула"
    normalized_branch, _ = normalize_name(branch_text)
    assert detect_branch(normalized_branch), "branch detection failed"  # noqa: S101
    target_stub = TargetRecord(
        index=0,
        name="Test",
        normalized_name="тестовый университет",
        city="москва",
        region="",
        inn=None,
        ogrn=None,
        aliases=[],
        normalized_aliases=[],
        tokens=["тестовый", "университет"],
        token_set={"тестовый", "университет"},
        initials={"т", "у"},
        prefixes2={"те", "ун"},
        prefixes3={"тес", "уни"},
        likely_branch=False,
    )
    candidate_stub = SourceRecord(
        record_id="1",
        name="Test",
        normalized_name="тестовый университет",
        city="москва",
        region="",
        inn=None,
        ogrn=None,
        aliases=[],
        normalized_aliases=[],
        tokens=["тестовый", "университет"],
        token_set={"тестовый", "университет"},
        initials={"т", "у"},
        prefixes2={"те", "ун"},
        prefixes3={"тес", "уни"},
        is_branch=False,
    )
    score = score_pair(
        target_stub,
        candidate_stub,
        candidate_stub.normalized_name,
        {"тестовый": 1, "университет": 1},
        {"tsr": 95.0, "pr": 93.0, "wr": 90.0, "lcs": 92.0},
    )
    assert score.score_final >= 92, "Expected high confidence score"  # noqa: S101
    branch_candidate = SourceRecord(
        record_id="2",
        name="Филиал тестового университета",
        normalized_name="филиал тестового университета",
        city="москва",
        region="",
        inn=None,
        ogrn=None,
        aliases=[],
        normalized_aliases=[],
        tokens=["филиал", "тестового", "университета"],
        token_set={"филиал", "тестового", "университета"},
        initials={"ф", "т", "у"},
        prefixes2={"фи", "те", "ун"},
        prefixes3={"фил", "тес", "уни"},
        is_branch=True,
    )
    penalty_score = score_pair(
        target_stub,
        branch_candidate,
        branch_candidate.normalized_name,
        {"тестовый": 1, "университет": 1},
        {"tsr": 80.0, "pr": 78.0, "wr": 79.0, "lcs": 80.0},
    )
    assert penalty_score.score_final < 86, "Branch penalty not applied"  # noqa: S101
    high_confidence = CandidateScore(
        source=candidate_stub,
        candidate_variant=candidate_stub.normalized_name,
        display_name=candidate_stub.name,
        score_token_set=95.0,
        score_partial=94.0,
        score_wratio=92.0,
        score_lcs=90.0,
        score_final=95.0,
        city_match=True,
        region_match=False,
        id_match=False,
        likely_branch=False,
    )
    medium_confidence = CandidateScore(
        source=candidate_stub,
        candidate_variant=candidate_stub.normalized_name,
        display_name=candidate_stub.name,
        score_token_set=90.0,
        score_partial=88.0,
        score_wratio=85.0,
        score_lcs=87.0,
        score_final=88.0,
        city_match=False,
        region_match=False,
        id_match=False,
        likely_branch=False,
    )
    low_confidence = CandidateScore(
        source=candidate_stub,
        candidate_variant=candidate_stub.normalized_name,
        display_name=candidate_stub.name,
        score_token_set=80.0,
        score_partial=79.0,
        score_wratio=78.0,
        score_lcs=77.0,
        score_final=82.0,
        city_match=False,
        region_match=False,
        id_match=False,
        likely_branch=False,
    )
    assert resolve_conflicts([high_confidence, medium_confidence]) is high_confidence  # noqa: S101
    assert resolve_conflicts([medium_confidence, medium_confidence]) is None  # noqa: S101
    assert resolve_conflicts([low_confidence]) is low_confidence  # noqa: S101
    print("Self-tests passed")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match university records between datasets")
    parser.add_argument("--source", required=False, help="Path to source file")
    parser.add_argument("--target", required=False, help="Path to target file")
    parser.add_argument("--out", required=False, help="Output CSV path")
    parser.add_argument("--review", required=False, help="Review CSV path")
    parser.add_argument("--log", required=False, help="Log file path")
    parser.add_argument("--source-sheet", dest="source_sheet", help="Source sheet name")
    parser.add_argument("--target-sheet", dest="target_sheet", help="Target sheet name")
    parser.add_argument("--id-col", dest="id_col", default="id")
    parser.add_argument("--name-col", dest="name_col", default="name")
    parser.add_argument("--city-col", dest="city_col", default="city")
    parser.add_argument("--region-col", dest="region_col", default="region")
    parser.add_argument("--inn-col", dest="inn_col", default="inn")
    parser.add_argument("--ogrn-col", dest="ogrn_col", default="ogrn")
    parser.add_argument("--alias-col", dest="alias_col", default="alias")
    parser.add_argument("--selftest", action="store_true", help="Run self-tests and exit")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.selftest:
        return
    missing = [name for name in ["source", "target", "out", "review", "log"] if getattr(args, name) is None]
    if missing:
        raise MatchingError(f"Missing required arguments: {', '.join(missing)}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        run_selftests()
        return
    validate_args(args)
    source_path = resolve_path(args.source)
    target_path = resolve_path(args.target)
    out_path = resolve_path(args.out)
    review_path = resolve_path(args.review)
    log_path = resolve_path(args.log)
    logger = setup_logger(log_path)
    logger.info("Starting matching with parameters: %s", json.dumps(vars(args), ensure_ascii=False))
    start_time = time.perf_counter()
    source_df = load_table(
        source_path,
        args.source_sheet,
        args.id_col,
        args.name_col,
        args.city_col,
        args.region_col,
        args.inn_col,
        args.ogrn_col,
        args.alias_col,
    )
    target_df = load_table(
        target_path,
        args.target_sheet,
        args.id_col,
        args.name_col,
        args.city_col,
        args.region_col,
        args.inn_col,
        args.ogrn_col,
        args.alias_col,
    )
    load_duration = time.perf_counter() - start_time
    logger.info("Loaded tables in %.3fs", load_duration)
    prep_start = time.perf_counter()
    sources, duplicate_logs = prepare_source_records(
        source_df,
        args.id_col,
        args.name_col,
        args.city_col,
        args.region_col,
        args.inn_col,
        args.ogrn_col,
        args.alias_col,
    )
    targets = prepare_target_records(
        target_df,
        args.name_col,
        args.city_col,
        args.region_col,
        args.inn_col,
        args.ogrn_col,
        args.alias_col,
    )
    token_freq = collect_token_statistics(sources)
    index = build_blocking_index(sources)
    prep_duration = time.perf_counter() - prep_start
    match_start = time.perf_counter()
    results = match_all(targets, sources, index, token_freq, logger)
    match_duration = time.perf_counter() - match_start
    save_start = time.perf_counter()
    save_outputs(target_df, results, out_path, review_path)
    save_duration = time.perf_counter() - save_start
    total_duration = time.perf_counter() - start_time
    durations = {
        "load": load_duration,
        "prepare": prep_duration,
        "match": match_duration,
        "save": save_duration,
        "total": total_duration,
    }
    report_metrics(logger, results, durations, duplicate_logs)
    logger.info("Matching completed in %.3fs", total_duration)


if __name__ == "__main__":
    main(sys.argv[1:])
