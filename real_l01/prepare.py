#!/usr/bin/env python3
"""Prepare reproducible integer-abundance empirical populations, without fitting.

The selection rule is fixed before estimator experiments: among groups with at
least 20 identified species, take the six largest N and the two central ranks
of the remaining groups sorted by N (ties broken by the group key).
These are finite empirical distributions, not estimates of wild abundance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public_ecology"
OUT = ROOT / "results_real_l01_20261104/prepared"
MANIFEST = {r["file"]: r for r in csv.DictReader((DATA / "download_manifest.csv").open())}
CHUNK = 250_000


def clean(s):
    return s.fillna("").astype(str).str.strip()


class PopulationBuilder:
    def __init__(self, dataset, fields, sources, rule, notes):
        self.dataset, self.fields, self.sources = dataset, fields, sources
        self.rule, self.notes = rule, notes
        self.vectors = defaultdict(Counter)
        self.records = Counter()
        self.audit = Counter()

    def add(self, df, groups, species, counts, eligible=None):
        self.audit["input_rows"] += len(df)
        sp = species.fillna("")
        vals = pd.to_numeric(counts, errors="coerce")
        count_ok = vals.notna() & (vals > 0) & (vals % 1 == 0)
        species_ok = sp != ""
        group_ok = pd.Series(True, index=df.index)
        for series in groups:
            group_ok &= series.notna() & (series.astype(str).str.strip() != "")
        eligible = pd.Series(True, index=df.index) if eligible is None else eligible.fillna(False)
        # Mutually exclusive, sequential exclusion counts sum to input rows.
        self.audit["excluded_protocol_or_scope"] += int((~eligible).sum())
        self.audit["excluded_unresolved_species"] += int((eligible & ~species_ok).sum())
        self.audit["excluded_invalid_or_nonpositive_count"] += int((eligible & species_ok & ~count_ok).sum())
        self.audit["excluded_missing_group"] += int((eligible & species_ok & count_ok & ~group_ok).sum())
        ok = eligible & species_ok & count_ok & group_ok
        self.audit["retained_rows"] += int(ok.sum())
        self.audit["retained_count"] += int(vals[ok].sum())
        use = pd.DataFrame({f: clean(s[ok]) for f, s in zip(self.fields, groups)})
        use["species"] = sp[ok].astype(str)
        use["count"] = vals[ok].astype("int64")
        grouped = use.groupby(self.fields + ["species"], sort=False)["count"].agg(["sum", "size"])
        for key, val in grouped.iterrows():
            group, taxon = tuple(str(x) for x in key[:-1]), str(key[-1])
            self.vectors[group][taxon] += int(val["sum"])
            self.records[group] += int(val["size"])

    def finish(self):
        candidates = []
        for key, vec in self.vectors.items():
            candidates.append({"key": key, "N": int(sum(vec.values())), "S": len(vec),
                               "n_records": int(self.records[key])})
        eligible = sorted((r for r in candidates if r["S"] >= 20), key=lambda r: (-r["N"], r["key"]))
        top = eligible[:6]
        remaining = eligible[6:]
        middle = remaining[max(0, len(remaining)//2-1):len(remaining)//2+1]
        chosen = top + middle
        assert len(chosen) == 8, (self.dataset, len(eligible))
        sources = []
        for path in self.sources:
            rel = str(path.relative_to(DATA))
            record = MANIFEST.get(rel)
            digest = record["sha256"] if record else hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                            "source_url": record["source"] if record else None,
                            "sha256_provenance": "download_manifest.csv" if record else "computed"})
        pops = []
        ranks = {r["key"]: rank+1 for rank, r in enumerate(eligible)}
        for idx, r in enumerate(chosen):
            key = r["key"]
            vec = self.vectors[key]
            species = sorted(vec)
            digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:10]
            pid = f"{self.dataset}_{digest}"
            source_index = {"2008_2009": 0, "2013_2014": 1, "2018_2019": 2}[key[0]] if self.dataset == "epa_nrsa" else 0
            primary_source = sources[source_index]
            pop = {"id": pid, "dataset": self.dataset,
                   "group": dict(zip(self.fields, key)), "species": species,
                   "counts": [int(vec[s]) for s in species],
                   "N": r["N"], "S": r["S"], "n_records": r["n_records"],
                   "selection": "largest_N" if idx < 6 else "middle_N",
                   "N_rank_among_eligible": ranks[key],
                   "source_path": primary_source["path"], "source_sha256": primary_source["sha256"],
                   "source_files": sources, "grouping_rule": self.rule, "notes": self.notes}
            pops.append(pop)
            with (OUT / f"{pid}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["species", "count"])
                writer.writerows(zip(species, pop["counts"]))
        with (OUT / f"{self.dataset}_all_group_summary.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.fields + ["N", "S", "n_records", "eligible_S_ge_20", "selected"])
            chosen_keys = {r["key"] for r in chosen}
            for r in sorted(candidates, key=lambda r: (-r["N"], r["key"])):
                writer.writerow(list(r["key"]) + [r["N"], r["S"], r["n_records"], r["S"] >= 20, r["key"] in chosen_keys])
        audit = dict(self.audit) | {"dataset": self.dataset, "grouping_rule": self.rule,
                "notes": self.notes, "source_files": sources, "total_groups": len(candidates),
                "eligible_groups": len(eligible), "selected_groups": len(pops)}
        (OUT / f"{self.dataset}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        (OUT / f"{self.dataset}_populations.json").write_text(json.dumps(pops, indent=2) + "\n")
        print(json.dumps({k: audit[k] for k in ["dataset", "input_rows", "retained_rows", "retained_count", "eligible_groups", "selected_groups"]}), flush=True)
        return pops, audit


def midfish():
    base = DATA / "01_fish_counts/midfish"
    tax = pd.read_csv(base / "raw/Taxa_List.csv", keep_default_na=False)
    tax["Genus"], tax["Species"] = clean(tax.Genus), clean(tax.Species)
    good = tax.Genus.str.fullmatch("[A-Za-z]+") & tax.Species.str.fullmatch("[A-Za-z]+")
    lookup = dict(zip(tax.loc[good, "CommonName"], tax.loc[good, "Genus"].str.capitalize() + " " + tax.loc[good, "Species"].str.lower()))
    builder = PopulationBuilder("midfish", ["state", "year", "gear"],
        [base / "raw/Midwest_Hive_Data_Files.zip", base / "raw/Taxa_List.csv"],
        "state x year x sampling_method_simple; all available years",
        "One retained row = one recorded fish, including rows expanded from aggregate catch. Map local taxon list to binomial names; merge aliases; exclude missing/coarse/hybrid taxa and either no-catch flag. Groups pool lakes within state and gear, defining an empirical catch distribution, not wild population abundance.")
    cols = ["state", "year", "sampling_method_simple", "species_1", "total_effort_nothing_caught", "sub_effort_nothing_caught"]
    for path in sorted((base / "extracted").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=CHUNK, columns=cols):
            df = batch.to_pandas()
            no_catch = df.total_effort_nothing_caught.fillna(False).astype(bool) | df.sub_effort_nothing_caught.fillna(False).astype(bool)
            species = clean(df.species_1).map(lookup)
            year = pd.to_numeric(df.year, errors="coerce").astype("Int64").astype("string")
            builder.add(df, [df.state, year, df.sampling_method_simple], species,
                        pd.Series(1, index=df.index), ~no_catch)
        print("midfish scanned", path.parent.name, flush=True)
    return builder.finish()


def ltrm():
    base = DATA / "01_fish_counts/ltrm"
    fmt = next((base / "extracted").rglob("ltrm_fish_format.txt"))
    text = fmt.read_text(errors="replace")
    species_section = text[text.index("Following are the valid codes and the common species name"):]
    species_section = species_section[:species_section.index("LENGTH")]
    lookup = {code: name.strip() for code, name in re.findall(r"^([A-Z0-9-]{4})\s*=\s*(.+)$", species_section, re.M)
              if not re.search("hybrid|unidentified|no fish|tiger muskellunge", name, re.I)}
    builder = PopulationBuilder("ltrm", ["pool", "year", "gear"],
        [base / "raw/LTRM_FISH_DATA_ENTIRE.zip"],
        "pool x year(sdate) x gear; years >= 1993 (post design revision)",
        "Sum catch only, never repeated sheet or collection totals. Use documented species-code whitelist; remove unidentified, no-fish, hybrid and unlisted codes. The latter includes valid but undocumented additions: the filtered subset is explicitly narrower than the complete release.")
    path = next((base / "extracted").rglob("ltrm_fish_data_entire.csv"))
    cols = ["pool", "sdate", "gear", "fishcode", "catch"]
    for df in pd.read_csv(path, usecols=cols, dtype=str, chunksize=CHUNK):
        year = pd.to_numeric(df.sdate.str.extract(r"/(\d{4})$", expand=False), errors="coerce").astype("Int64")
        species = clean(df.fishcode).map(lookup)
        builder.add(df, [df.pool, year.astype("string"), df.gear], species, df["catch"], year >= 1993)
    return builder.finish()


def nrmn():
    path = DATA / "02_community_abundance/reef_life_survey/raw/NRMN_global_reef_fish_20261104.csv"
    builder = PopulationBuilder("nrmn", ["program", "ecoregion", "year"], [path],
        "program x marine ecoregion x survey year; method 1 only",
        "Sum integer total across legitimate size-class and survey/block rows. Exclude non-binomial unidentified/hybrid labels. Count unit is fish sightings recorded by underwater visual surveys; repeated observation of individual fish across surveys cannot be excluded; no biomass or density values used.")
    cols = ["program", "ecoregion", "survey_date", "method", "species_name", "total"]
    for df in pd.read_csv(path, usecols=cols, dtype=str, chunksize=CHUNK):
        sp = clean(df.species_name)
        sp = sp.where(sp.str.fullmatch(r"[A-Z][a-z]+ [a-z]+"))
        year = df.survey_date.str.extract(r"^(\d{4})", expand=False)
        builder.add(df, [df.program, df.ecoregion, year], sp, df.total, pd.to_numeric(df.method, errors="coerce") == 1)
    return builder.finish()


def epa():
    base = DATA / "01_fish_counts/finsyncr/raw"
    taxpath = base / "inst__extdata__updateNRSAfishtax.csv"
    tax = pd.read_csv(taxpath, dtype=str, keep_default_na=False)
    genus, epithet = clean(tax.GENUS), clean(tax.SPECIES)
    good = genus.str.fullmatch(r"[A-Z]+") & epithet.str.fullmatch(r"[A-Z]+(?: [A-Z]+)?") & (clean(tax.HERP) == "")
    good &= ~epithet.str.contains(r"\bSPP?\b|\bX\b|UNIDENT", regex=True)
    lookup = dict(zip(tax.loc[good, "TAXA_ID"], genus[good].str.capitalize() + " " + epithet[good].str.split().str[0].str.lower()))
    configs = [("2008_2009", "EPA__fishcts.csv", "FW_ECO9", "FINAL_CT"),
               ("2013_2014", "EPA__nrsa1314_fishcts_04232019.csv", "AG_ECO9_NM", "TOTAL"),
               ("2018_2019", "EPA__nrsa-1819-fish-count-data.csv", "AG_ECO9", "TOTAL")]
    builder = PopulationBuilder("epa_nrsa", ["cycle", "ecoregion"],
        [base / name for _, name, _, _ in configs] + [taxpath],
        "NRSA two-year survey cycle x source ecoregion; VISIT_NO=1; IS_DISTINCT=1",
        "Integer FINAL_CT/TOTAL only. Join pinned NRSA taxon list, retain fish with species-level binomials, merge subspecies/age categories, exclude HERP, hybrids, unresolved taxa. First visit only; pooled site catches represent an empirical distribution, not survey-weighted regional abundance.")
    for cycle, name, eco, count in configs:
        df = pd.read_csv(base / name, dtype=str)
        eligible = (pd.to_numeric(df.VISIT_NO, errors="coerce") == 1) & (pd.to_numeric(df.IS_DISTINCT, errors="coerce") == 1)
        builder.add(df, [pd.Series(cycle, index=df.index), df[eco]], clean(df.TAXA_ID).map(lookup), df[count], eligible)
    return builder.finish()


def bbs():
    base = DATA / "06_birds/bbs/raw"
    taxpath = base / "SpeciesList.csv"
    tax = pd.read_csv(taxpath, dtype=str, encoding="latin1", keep_default_na=False)
    gen, ep = clean(tax.Genus), clean(tax.Species)
    # Retain identified species/forms/subspecies; collapse form/subspecies to binomial.
    good = gen.str.fullmatch(r"[A-Z][a-z]+") & ep.str.match(r"^[a-z]+(?:$| [a-z]+$| \([^)]*\)$)")
    good &= ~clean(tax.English_Common_Name).str.contains(r"unid\.|hybrid", case=False, regex=True)
    good &= ~ep.str.contains(r"\bsp\b|\bx\b|/", regex=True)
    lookup = dict(zip(tax.loc[good, "AOU"].str.zfill(5), gen[good] + " " + ep[good].str.split().str[0]))
    path = base / "States.zip"
    builder = PopulationBuilder("bbs", ["country", "state", "route", "year", "route_data_id"], [path, taxpath],
        "single route survey (RouteDataID), limited a priori to years 2015-2025; RPID=101 standard runs",
        "SpeciesTotal is a bird observation count, not StopTotal (occupied stops). Only States.zip used: 50-StopData is an overlapping representation. Map AOU through supplied SpeciesList, merge forms/subspecies, remove hybrid/unidentified labels. Sightings do not guarantee distinct individual birds; empirical count-distribution benchmark only.")
    cols = ["CountryNum", "StateNum", "Route", "Year", "RouteDataID", "RPID", "AOU", "SpeciesTotal"]
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".csv")):
            with z.open(name) as f:
                for df in pd.read_csv(f, usecols=cols, dtype=str, chunksize=CHUNK):
                    year = pd.to_numeric(df.Year, errors="coerce")
                    eligible = year.between(2015, 2025) & (clean(df.RPID) == "101")
                    builder.add(df, [df.CountryNum, df.StateNum, df.Route, df.Year, df.RouteDataID],
                                clean(df.AOU).str.zfill(5).map(lookup), df.SpeciesTotal, eligible)
    return builder.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["midfish", "ltrm", "nrmn", "epa", "bbs"],
                        default=["midfish", "ltrm", "nrmn", "epa", "bbs"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        globals()[name]()
    pops = []
    audits = []
    for name in ["midfish", "ltrm", "nrmn", "epa_nrsa", "bbs"]:
        path = OUT / f"{name}_populations.json"
        if path.exists():
            pops += json.loads(path.read_text())
            audits.append(json.loads((OUT / f"{name}_audit.json").read_text()))
    for p in pops:
        assert len(p["species"]) == len(set(p["species"])) == p["S"]
        assert len(p["counts"]) == p["S"] and sum(p["counts"]) == p["N"]
        assert all(type(v) is int and v > 0 for v in p["counts"])
    (OUT / "populations.json").write_text(json.dumps(pops, indent=2) + "\n")
    summary = {"selection_rule": __doc__, "scope": "Empirical abundance distributions; no inference about true unobserved ecological species richness.",
               "n_populations": len(pops), "datasets": audits,
               "excluded_sources": {"ebird": "Downloaded tutorial is a single focal species; full EBD unavailable.",
                   "amazonfish_ictio": "Occurrence records not exchangeable fish individual counts.",
                   "midfish_cpue": "Abundance_Data_Files contains CPUE products, not raw counts.",
                   "yrfir_edna": "Barcode/sequence counts are not animal individual counts.",
                   "biotime_rivfishtime_fishglob": "Not used in this fixed five-source benchmark; unit harmonization and additional loaders required."},
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    flat = [{k: p[k] for k in ["id", "dataset", "N", "S", "n_records", "selection", "N_rank_among_eligible"]} |
            {"group": json.dumps(p["group"], sort_keys=True)} for p in pops]
    pd.DataFrame(flat).to_csv(OUT / "population_summary.csv", index=False)
    print(f"Ready: {len(pops)} populations -> {OUT / 'populations.json'}", flush=True)


if __name__ == "__main__":
    main()
