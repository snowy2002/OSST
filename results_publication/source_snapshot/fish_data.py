"""下载、校验和清洗 Derek H. Ogle 的 InchLake1 逐个体捕获记录。"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "2db85a9eaed25f9d13be5a3a5fde5dce0cf125c7"
URL = f"https://raw.githubusercontent.com/fishR-Core-Team/FSAdata/{COMMIT}/data-raw/InchLake1.csv"
SHA256 = "0bcf180122609e77a220a46626004776c36a5176a1e4f663a4c81372b0219584"
SOURCE_PAGE = "https://fishr-core-team.github.io/FSAdata/reference/InchLake1.html"


def prepare_fish_data(data_dir: Path = ROOT / "data") -> list[dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = data_dir / "InchLake1.csv"
    if raw.exists():
        payload = raw.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=45) as response:
            payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise ValueError("原始 CSV SHA256 不匹配；停止分析，避免版本混用或下载到错误页面")
    if not raw.exists():
        raw.write_bytes(payload)
    original = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(original) != 4894:
        raise ValueError("原始记录数与固定版本不符")
    clean = []
    for i, row in enumerate(original, start=1):
        if row["species"] == "Rusty Crayfish":
            continue
        if not row["species"].strip():
            raise ValueError(f"第 {i} 条记录缺失物种名")
        # record_id 是源文件行号，不冒充真实个体编号或精确捕获先后顺序。
        clean.append({"record_id": i, "year": int(row["year"]),
                      "gear": row["netType"], "event_id": row["netID"],
                      "species": row["species"], "length_inch": row["length"]})
    with (data_dir / "inch_lake_fish.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(clean[0]))
        w.writeheader()
        w.writerows(clean)
    manifest = {"dataset": "InchLake1", "creator": "Derek H. Ogle, personal collection",
                "source_page": SOURCE_PAGE, "download_url": URL, "commit": COMMIT,
                "sha256": digest, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_records": len(original), "fish_records": len(clean),
                "excluded": {"Rusty Crayfish": len(original) - len(clean)},
                "observed_fish_species": len({x["species"] for x in clean}),
                "package_license": "GPL (>= 2), per upstream DESCRIPTION; attribution retained",
                "cleaning": "只去除明确的非鱼类 Rusty Crayfish；不按相同行去重，不按体长筛选",
                "sampling_caveats": ["含多种渔具和 2007/2008 两年；按年份和渔具分层",
                                     "netID 为捕获事件；同一网内的鱼可能相关",
                                     "无逐鱼时间戳；CSV 顺序不认定为捕获时间顺序",
                                     "记录的个体不保证跨事件从未重捕；没有可用于去重的鱼个体 ID",
                                     "完整数据中的物种数不是全湖物种普查结果"]}
    (data_dir / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    groups = {}
    for x in clean:
        key = (x["year"], x["gear"])
        groups.setdefault(key, Counter())[x["species"]] += 1
    with (data_dir / "strata_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "gear", "fish_records", "observed_species"])
        for (year, gear), c in sorted(groups.items()):
            writer.writerow([year, gear, sum(c.values()), len(c)])
    return clean


def load_counts_csv(path: Path, species_column: str = "species",
                    count_column: str | None = None) -> Counter:
    """兼容逐条标签 CSV 或 species,count 的汇总 CSV；拒绝无意义/分数计数。"""
    counts = Counter()
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = [species_column] + ([count_column] if count_column else [])
        if not reader.fieldnames or any(x not in reader.fieldnames for x in needed):
            raise ValueError(f"CSV 缺少列: {needed}")
        for line, row in enumerate(reader, start=2):
            species = (row.get(species_column) or "").strip()
            if not species:
                raise ValueError(f"CSV 第 {line} 行缺少类别标签")
            try:
                count = int(row[count_column]) if count_column else 1
            except (ValueError, TypeError) as e:
                raise ValueError(f"CSV 第 {line} 行计数不是整数") from e
            if count < 0:
                raise ValueError(f"CSV 第 {line} 行计数不能为负")
            if count:
                counts[species] += count
    return counts


if __name__ == "__main__":
    data = prepare_fish_data()
    print(f"已准备 {len(data)} 条鱼类记录：{ROOT / 'data' / 'inch_lake_fish.csv'}")
