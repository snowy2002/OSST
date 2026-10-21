"""跨领域公开数据：下载原始文件，保留清洗审计，生成类别计数及自然分组。

原始记录只用于定义有限参考总体；主实验从其经验分布独立采样。
MovieLens 只供本机研究，不放进便携包。没有伪造采样时间或捕获顺序。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "large"
SOURCES = {
    "nips": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz",
        "filename": "docword.nips.txt.gz", "domain": "scientific_text",
        "page": "https://archive.ics.uci.edu/dataset/164/bag+of+words",
        "citation": "Newman, D. (2008). Bag of Words. UCI. DOI:10.24432/C5ZG6P",
        "license": "CC BY 4.0", "redistributable": True,
        "sha256": "d45f9f7ceeadf118117e14346f018273007b765593a4dbe02915d37a1faa58b7",
    },
    "enron": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz",
        "filename": "docword.enron.txt.gz", "domain": "email_text",
        "page": "https://archive.ics.uci.edu/dataset/164/bag+of+words",
        "citation": "Newman, D. (2008). Bag of Words. UCI. DOI:10.24432/C5ZG6P",
        "license": "CC BY 4.0", "redistributable": True,
        "sha256": "bc8b51ddb8329b04f0777266fb076585db9f93808b1214b76ca2bc9c472567c0",
    },
    "movielens": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "filename": "ml-1m.zip", "domain": "recommendation",
        "page": "https://grouplens.org/datasets/movielens/1m/",
        "citation": "Harper, F. M. & Konstan, J. A. (2015). The MovieLens Datasets. DOI:10.1145/2827872",
        "license": "GroupLens research-use terms; redistribution requires permission",
        "license_url": "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt",
        "redistributable": False,
        "sha256": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    },
    "retail": {
        "url": "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        "filename": "online_retail.zip", "domain": "retail_transactions",
        "page": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "citation": "Chen, D. (2015). Online Retail. UCI. DOI:10.24432/C5BW33",
        "license": "CC BY 4.0", "redistributable": True,
        "sha256": "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b",
    },
    "bci": {
        "url": "https://raw.githubusercontent.com/vegandevs/vegan/19643ae1a547c1789e54a993f42cdcd03b212990/data/BCI.rda",
        "filename": "BCI.rda", "domain": "forest_ecology",
        "page": "https://vegandevs.github.io/vegan/reference/BCI.html",
        "citation": "Condit et al. (2002). Beta-diversity in tropical forest trees. DOI:10.1126/science.1066854; BCI matrix distributed by vegan",
        "license": "vegan distribution: GPL-2; cite original ecological data source",
        "redistributable": True,
        "sha256": "9730c9a9db6eb455ae97c61c863a2ca6a074f7d00a3062b9a3a9a873158a5fd1",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download(name: str, data: Path = DATA) -> dict:
    spec = SOURCES[name]
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / spec["filename"]
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                request = Request(spec["url"], headers={"User-Agent": "OSST research benchmark/1.0"})
                with urlopen(request, timeout=60) as source, part.open("wb") as target:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        target.write(chunk)
                part.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    digest = sha256(path)
    if spec.get("sha256") and digest != spec["sha256"]:
        raise ValueError(f"{name}: 原始数据 SHA256 与固定版本不符")
    print(f"数据 {name}: {path.stat().st_size:,} bytes，SHA256 {digest}", flush=True)
    return {**spec, "sha256": digest, "bytes": path.stat().st_size,
            "local_file": str(path.relative_to(data)), "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def save_population(name: str, matrix: sparse.csr_matrix, labels, details: dict, data: Path = DATA) -> dict:
    """matrix[自然分组, 类别] 是计数。只保留正频数的行列。"""
    matrix = matrix.tocsr().astype(np.int64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if np.any(matrix.data < 0):
        raise ValueError("不能有负计数")
    keep_cols = np.asarray(matrix.sum(axis=0)).ravel() > 0
    keep_rows = np.asarray(matrix.sum(axis=1)).ravel() > 0
    matrix = matrix[keep_rows][:, keep_cols].tocsr()
    labels = np.asarray(labels, dtype=str)[keep_cols]
    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    if len(counts) < 2:
        raise ValueError("参考总体至少需要两个类别")
    prepared = data / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared / f"{name}_counts.npz", labels=labels, counts=counts)
    sparse.save_npz(prepared / f"{name}_groups.npz", matrix)
    p = np.sort(counts / counts.sum())[::-1]
    stats = {"dataset": name, **details, "records": int(counts.sum()), "support": len(counts),
             "groups": matrix.shape[0], "nonzero_group_category_pairs": matrix.nnz,
             "min_category_count": int(counts.min()), "max_category_count": int(counts.max()),
             "top10_mass": float(p[:10].sum()),
             "counts_sha256": sha256(prepared / f"{name}_counts.npz"),
             "groups_sha256": sha256(prepared / f"{name}_groups.npz"),
             "reference_scope": "该数据文件清洗后的有限参考总体，不是未知外部总体的真实支持集"}
    save_json(prepared / f"{name}_summary.json", stats)
    print(f"预处理 {name}: {stats['records']:,} 次出现，{stats['support']:,} 类，{stats['groups']:,} 组", flush=True)
    return stats


def prepare_words(name: str, data: Path = DATA) -> dict:
    with gzip.open(data / "raw" / SOURCES[name]["filename"], "rt") as f:
        D, W, nnz = [int(f.readline()) for _ in range(3)]
        triples = np.loadtxt(f, dtype=np.int64)
    if triples.shape != (nnz, 3) or np.any(triples[:, 2] <= 0):
        raise ValueError("docword 格式或计数无效")
    matrix = sparse.coo_matrix((triples[:, 2], (triples[:, 0]-1, triples[:, 1]-1)), shape=(D, W))
    return save_population(name, matrix, np.arange(1, W+1), {
        "raw_rows": nnz, "declared_documents": D, "declared_vocabulary": W,
        "sample_unit": "保留词表中的一次词语出现", "category": "wordID", "group_unit": "document",
        "filters": "使用发布者预处理后的 docword；本地没有再次词频筛选",
        "limitations": "发布者已删除停用词并仅保留出现超过 10 次的词；词序不可恢复；只评估这个截断词表。文档内词语相关。",
    }, data)


def prepare_movies(data: Path = DATA) -> dict:
    with zipfile.ZipFile(data / "raw" / SOURCES["movielens"]["filename"]) as z:
        with z.open("ml-1m/ratings.dat") as f:
            rows = np.asarray([[int(x) for x in line.decode().split("::")[:2]] for line in f], dtype=np.int64)
    users, ri = np.unique(rows[:, 0], return_inverse=True)
    labels, ci = np.unique(rows[:, 1], return_inverse=True)
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.int64), (ri, ci)), shape=(len(users), len(labels)))
    return save_population("movielens", matrix, labels, {
        "raw_rows": len(rows), "sample_unit": "一条实际评分记录", "category": "movieID", "group_unit": "user",
        "filters": "保留所有评分，不使用 users.dat 人口属性，也不把未被评分的电影计入支持集",
        "limitations": "评分选择偏差、同用户相关、时间变化；有限电影曝光分布不代表所有电影。原始和派生数据不进分发包。",
    }, data)


def retail_reason(row) -> str | None:
    """互斥的剔除原因，顺序固定；样本按销售行计，不按 Quantity 扩展。"""
    invoice, code, _, qty, _, price, *_ = row
    if str(invoice).upper().startswith("C"):
        return "cancellation"
    if qty is None or float(qty) <= 0:
        return "nonpositive_quantity"
    if price is None or float(price) <= 0:
        return "nonpositive_price"
    if not re.fullmatch(r"[0-9]{5}[A-Z]*", str(code).strip().upper()):
        return "nonproduct_stockcode"
    return None


def prepare_retail(data: Path = DATA) -> dict:
    from openpyxl import load_workbook
    with zipfile.ZipFile(data / "raw" / SOURCES["retail"]["filename"]) as z:
        member = next(x for x in z.namelist() if x.endswith(".xlsx"))
        content = io.BytesIO(z.read(member))
    book = load_workbook(content, read_only=True, data_only=True)
    rows = book.active.iter_rows(values_only=True)
    next(rows)
    invoices, products, ri, ci = {}, {}, [], []
    reasons = Counter()
    raw_rows = 0
    for row in rows:
        raw_rows += 1
        reason = retail_reason(row)
        if reason:
            reasons[reason] += 1
            continue
        invoice, code = str(row[0]), str(row[1]).strip().upper()
        ri.append(invoices.setdefault(invoice, len(invoices)))
        ci.append(products.setdefault(code, len(products)))
    book.close()
    matrix = sparse.coo_matrix((np.ones(len(ri), dtype=np.int64), (ri, ci)), shape=(len(invoices), len(products)))
    return save_population("retail", matrix, list(products), {
        "raw_rows": raw_rows, "removed_rows": dict(reasons),
        "sample_unit": "一条正数量、正价格的商品销售行（不是一件商品）", "category": "StockCode", "group_unit": "InvoiceNo",
        "filters": "剔除取消、数量<=0、单价<=0、代码不匹配 [0-9]{5}[A-Z]* 的服务/特殊记录；缺失客户号不剔除；不擅自去重销售行",
        "limitations": "订单内相关、季节变化；SKU 是商品编码，不是生物学物种。只对清洗后的记录分布定义支持集。",
    }, data)


def prepare_bci(data: Path = DATA) -> dict:
    import rdata
    objects = rdata.conversion.convert(rdata.parser.parse_file(data / "raw" / "BCI.rda"))
    frame = objects["BCI"]
    matrix = sparse.csr_matrix(frame.to_numpy(dtype=np.int64))
    return save_population("bci", matrix, frame.columns, {
        "raw_rows": len(frame), "census": "1982, DBH>=10cm subset",
        "sample_unit": "该次普查 DBH>=10cm 的一棵树", "category": "发布者整理后的物种名", "group_unit": "1ha spatial quadrat",
        "filters": "原始 50 样方 x 225 物种计数矩阵全部使用；无需展开逐树标签",
        "limitations": "这是 1982 年大树子集，非 2019 全档案；用于算法基准，不用于当代生态结论。样方内空间相关。",
        "full_archive_attempt": "2019 Dryad 全档案 DOI:10.15146/5xcp-0d46 的公开文件下载返回 HTTP 403，API 下载返回 401，未使用该全档案，也未混合年份凑样本量。",
    }, data)


def prepare_all(data: Path = DATA, names=None) -> list[dict]:
    names = list(SOURCES) if names is None else list(names)
    data.mkdir(parents=True, exist_ok=True)
    # 下载互相独立；预处理顺序执行以限制内存。
    with ThreadPoolExecutor(max_workers=4) as pool:
        source_rows = list(pool.map(lambda name: download(name, data), names))
    save_json(data / "sources.json", dict(zip(names, source_rows)))
    result = []
    for name in names:
        cached = data / "prepared" / f"{name}_summary.json"
        if cached.exists():
            item = json.loads(cached.read_text())
            for suffix, key in [("counts", "counts_sha256"), ("groups", "groups_sha256")]:
                if sha256(cached.parent / f"{name}_{suffix}.npz") != item[key]:
                    raise ValueError(f"{name} 缓存校验失败")
        elif name in ("nips", "enron"):
            item = prepare_words(name, data)
        else:
            item = {"movielens": prepare_movies, "retail": prepare_retail, "bci": prepare_bci}[name](data)
        result.append(item)
    save_json(data / "dataset_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    prepare_all(args.data, args.datasets)
