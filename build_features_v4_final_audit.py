"""
Pipeline final v4 pentru licenta - reconstructie DB + feature engineering + dataset clustering.

Ce face:
1. Citeste toate cele 12 foi din Excel.
2. Foloseste headere pe 2 sau 3 randuri, dupa structura reala a fiecarei foi.
3. Creeaza id_pacient anonim si elimina identificatorii directi din tabelele salvate.
4. Salveaza toate foile anonimizate in SQLite (*_clean).
5. Construieste tabele de features pe domenii clinice.
6. Creeaza features_ai: tabel master interpretabil, 1 rand / pacient.
7. Creeaza doua matrici pentru clustering:
   - features_clustering: varianta baseline/conservatoare, fara agregari post-hoc;
   - features_clustering_descriptive: varianta exploratorie, include agregari longitudinale.
8. Adauga missing indicators pentru variabile cu multe valori lipsa si scaleaza doar variabilele continue.

Rulare:
    python3 build_features_v3_final.py
"""

from __future__ import annotations

import re
import sqlite3
import warnings
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

CANDIDATE_EXCEL_PATHS = [
    SCRIPT_DIR / "date_intrare.xlsx",
]
CALE_FISIER = next((p for p in CANDIDATE_EXCEL_PATHS if p.exists()), CANDIDATE_EXCEL_PATHS[0])
# Alegeti intotdeauna directorul proiectului curent ca locatie implicita pentru baza de date,
# astfel incat fisierul generat sa fie gasit in folderul in care il cautati.
CALE_DB = SCRIPT_DIR / "baza_date_licenta.db"

SHEETS = {
    "DATE GENERALE": "pacienti_clean",
    "ETIOLOGIE": "etiologie_clean",
    "VIZITE": "vizite_clean",
    "EXACERBARI": "exacerbari_clean",
    "SIMPTOME": "simptome_clean",
    "PROBE VENTILATORII": "probe_ventilatorii_clean",
    "BIOLOGIC": "biologic_clean",
    "BACTERIOLOGIE": "bacteriologie_clean",
    "HRCT": "hrct_clean",
    "COMORBIDITATI": "comorbiditati_clean",
    "TRATAMENT RESPIRATOR": "tratament_respirator_clean",
    "SCORURI": "scoruri_clean",
}

# Unele foi au header pe 3 randuri; altfel se pierd detalii ca CHISTICE / VARICOASE / categorii BSI.
HEADER_LEVELS = {
    "HRCT": [0, 1, 2],
    "TRATAMENT RESPIRATOR": [0, 1, 2],
    "SCORURI": [0, 1, 2],
}
DEFAULT_HEADER = [0, 1]

IDENTIFIER_PATTERNS = ["cnp", "id", "nume", "prenume", "pacient"]
MAX_MISSING_CLUSTERING = 0.50

# =========================
# HELPERS
# =========================

def normalize_text(s: object) -> str:
    if pd.isna(s):
        return ""
    text = str(s).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    text = re.sub(r"\s+", " ", text)
    return text


def slugify_col(col: object) -> str:
    text = normalize_text(col).lower()
    text = text.replace("/", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "coloana"


def make_unique(cols: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        base = c or "coloana"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for col in df.columns:
            parts = []
            for c in col:
                s = str(c).strip()
                if s and not s.startswith("Unnamed") and s.lower() != "nan":
                    parts.append(s)
            cols.append("_".join(parts))
        df.columns = make_unique(cols)
    else:
        df.columns = make_unique([str(c).strip() for c in df.columns])
    return df


def read_sheet(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(CALE_FISIER, sheet_name=sheet_name, header=HEADER_LEVELS.get(sheet_name, DEFAULT_HEADER))
    df = flatten_columns(df)
    df = df.dropna(how="all").copy()
    return df


def clean_id(val: object) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "":
        return None
    # Excel poate afisa unele ID-uri ca 123.0; pastram partea intreaga.
    s = s.split(".")[0]
    s = re.sub(r"\D", "", s)
    return s if s else None


def find_identifier_col(df: pd.DataFrame) -> str:
    normalized = {c: slugify_col(c) for c in df.columns}
    for target in ["id", "cnp"]:
        for c, nc in normalized.items():
            if nc == target:
                return c
    for c, nc in normalized.items():
        if nc in {"id_pacient", "cnp_pacient", "cod_pacient"}:
            return c
    raise KeyError(f"Nu gasesc coloana ID/CNP. Coloane disponibile: {list(df.columns)}")


def find_col(df: pd.DataFrame, include: list[str], exclude: Optional[list[str]] = None, required: bool = True) -> Optional[str]:
    exclude = exclude or []
    inc = [normalize_text(x) for x in include]
    exc = [normalize_text(x) for x in exclude]
    for c in df.columns:
        nc = normalize_text(c)
        if all(x in nc for x in inc) and not any(x in nc for x in exc):
            return c
    if required:
        raise KeyError(f"Nu gasesc coloana include={include}, exclude={exclude}. Coloane: {list(df.columns)}")
    return None


def find_cols_any(df: pd.DataFrame, patterns: list[str], exclude: Optional[list[str]] = None) -> list[str]:
    exclude = exclude or []
    pats = [normalize_text(x) for x in patterns]
    exc = [normalize_text(x) for x in exclude]
    cols = []
    for c in df.columns:
        nc = normalize_text(c)
        if any(p in nc for p in pats) and not any(e in nc for e in exc):
            cols.append(c)
    return cols


def find_exact_slug(df: pd.DataFrame, slug: str) -> Optional[str]:
    for c in df.columns:
        if slugify_col(c) == slug:
            return c
    return None


def to_num(s: pd.Series) -> pd.Series:
    ser = (
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    ser = ser.mask(ser.isin(["nan", "None", "NaT", ""]))
    return pd.to_numeric(ser, errors="coerce")


def calculate_bmi_series(weight_kg: pd.Series, height_value: pd.Series) -> pd.Series:
    """Calculeaza IMC din greutate si inaltime, cu validari minimale de plauzibilitate.

    Inaltimea poate fi introdusa in centimetri (ex. 163) sau metri (ex. 1.63).
    Rezultatul este pastrat doar pentru valori plauzibile ale IMC.
    Aceasta este o derivare determinista, nu imputare statistica.
    """
    weight = pd.to_numeric(weight_kg, errors="coerce")
    height = pd.to_numeric(height_value, errors="coerce")

    height_m = height.copy()
    height_m = height_m.where(~height_m.between(50, 250), height_m / 100)
    height_m = height_m.where(height_m.between(1.0, 2.5))
    weight = weight.where(weight.between(20, 300))

    bmi = weight / (height_m ** 2)
    return bmi.where(bmi.between(10, 80)).round(2)


def num_col(df: pd.DataFrame, include: list[str], exclude: Optional[list[str]] = None) -> pd.Series:
    col = find_col(df, include, exclude=exclude, required=False)
    return to_num(df[col]) if col else pd.Series(np.nan, index=df.index)


def flag(s: pd.Series) -> pd.Series:
    return to_num(s).fillna(0).clip(0, 1).astype(int)


def max_flag(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0, index=df.index, dtype=int)
    return df[cols].apply(to_num).fillna(0).max(axis=1).clip(0, 1).astype(int)


def sum_flags(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0, index=df.index, dtype=int)
    return df[cols].apply(to_num).fillna(0).sum(axis=1).astype(int)


def attach_id(df: pd.DataFrame, id_map: dict[str, int]) -> pd.DataFrame:
    id_col = find_identifier_col(df)
    out = df.copy()
    out["cnp_key"] = out[id_col].apply(clean_id)
    out["id_pacient"] = out["cnp_key"].map(id_map)
    return out[out["id_pacient"].notna()].copy()


def anonymized_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_cols = []
    for c in out.columns:
        nc = slugify_col(c)
        if c == "cnp_key" or any(p in nc for p in IDENTIFIER_PATTERNS):
            if c != "id_pacient":
                drop_cols.append(c)
    out = out.drop(columns=drop_cols, errors="ignore")
    out.columns = make_unique([slugify_col(c) for c in out.columns])
    cols = [c for c in out.columns if c == "id_pacient"] + [c for c in out.columns if c != "id_pacient"]
    return out[cols]


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    if right.empty:
        return left
    if right["id_pacient"].duplicated().any():
        raise ValueError(f"Tabelul {name} are id_pacient duplicat dupa agregare.")
    return left.merge(right, on="id_pacient", how="left")


def drop_empty_or_constant(df: pd.DataFrame, keep: Optional[set[str]] = None) -> pd.DataFrame:
    keep = keep or set()
    drop = []
    for c in df.columns:
        if c in keep:
            continue
        if df[c].notna().sum() == 0:
            drop.append(c)
        elif df[c].nunique(dropna=True) <= 1:
            # Pentru flags constante zero/unu, nu ajuta nici analiza, nici clusterizarea.
            drop.append(c)
    return df.drop(columns=drop, errors="ignore")

# =========================
# LOAD + ID MAP
# =========================

print("⏳ Lansare pipeline final ...")
print(f"Excel folosit: {CALE_FISIER}")

if not CALE_FISIER.exists():
    raise FileNotFoundError(f"Nu gasesc fisierul Excel. Pune-l la una din locatiile: {CANDIDATE_EXCEL_PATHS}")

raw: dict[str, pd.DataFrame] = {sh: read_sheet(sh) for sh in SHEETS}

# Verificare ca toate foile asteptate exista.
excel_sheets = pd.ExcelFile(CALE_FISIER).sheet_names
missing_sheets = [sh for sh in SHEETS if sh not in excel_sheets]
if missing_sheets:
    raise ValueError(f"Lipsesc foi din Excel: {missing_sheets}")

df_gen = raw["DATE GENERALE"].copy()
id_gen_col = find_identifier_col(df_gen)
df_gen["cnp_key"] = df_gen[id_gen_col].apply(clean_id)
df_gen = df_gen[df_gen["cnp_key"].notna()].drop_duplicates("cnp_key").copy()
df_gen["id_pacient"] = range(1, len(df_gen) + 1)
id_map = dict(zip(df_gen["cnp_key"], df_gen["id_pacient"]))

attached: dict[str, pd.DataFrame] = {"DATE GENERALE": df_gen}
for sh in SHEETS:
    if sh == "DATE GENERALE":
        continue
    attached[sh] = attach_id(raw[sh], id_map)

# =========================
# PACIENTI
# =========================

pac = pd.DataFrame({"id_pacient": df_gen["id_pacient"].astype(int)})
sex_col = find_col(df_gen, ["SEX"], required=False)
age_col = find_col(df_gen, ["VARSTA", "DG"], required=False) or find_col(df_gen, ["VARSTA"], required=False)
an_dg_col = find_col(df_gen, ["AN DG"], required=False)
status_alive_col = find_col(df_gen, ["STATUS VITAL", "IN VIATA"], required=False)
status_dead_col = find_col(df_gen, ["STATUS VITAL", "DECEDAT"], required=False)
weight_col = find_col(df_gen, ["GREUTATE"], required=False)
height_col = find_col(df_gen, ["INALTIME"], required=False)
imc_col = find_col(df_gen, ["IMC", "VALOARE"], required=False) or find_col(df_gen, ["IMC"], required=False)

pac["sex"] = df_gen[sex_col].astype(str).str.strip().replace({"nan": np.nan}) if sex_col else np.nan
pac["varsta"] = to_num(df_gen[age_col]) if age_col else np.nan
pac["an_diagnostic"] = to_num(df_gen[an_dg_col]) if an_dg_col else np.nan
if status_alive_col or status_dead_col:
    alive = flag(df_gen[status_alive_col]) if status_alive_col else pd.Series(0, index=df_gen.index)
    dead = flag(df_gen[status_dead_col]) if status_dead_col else pd.Series(0, index=df_gen.index)
    pac["status_vital"] = np.select([dead.eq(1), alive.eq(1)], [0, 1], default=np.nan)
else:
    pac["status_vital"] = np.nan
pac["greutate"] = to_num(df_gen[weight_col]) if weight_col else np.nan
pac["inaltime"] = to_num(df_gen[height_col]) if height_col else np.nan
pac["imc"] = to_num(df_gen[imc_col]) if imc_col else np.nan

# Curatare valori imposibile/plauzibilitate minima. Inaltimea poate fi in cm sau metri;
# pentru calculul IMC conversia este tratata in calculate_bmi_series().
pac.loc[~pac["greutate"].between(20, 300), "greutate"] = np.nan
pac.loc[~(pac["inaltime"].between(50, 250) | pac["inaltime"].between(1.0, 2.5)), "inaltime"] = np.nan
pac.loc[~pac["imc"].between(10, 80), "imc"] = np.nan

# Daca IMC lipseste, dar greutatea si inaltimea exista, il calculam determinist.
# Nu suprascriem valorile IMC existente in Excel; pastram trasabilitatea sursei.
imc_calculat = calculate_bmi_series(pac["greutate"], pac["inaltime"])
pac["imc_sursa"] = np.select(
    [
        pac["imc"].notna(),
        pac["imc"].isna() & imc_calculat.notna(),
    ],
    [
        "original_excel",
        "calculat_din_greutate_inaltime",
    ],
    default="lipsa",
)
pac["imc"] = pac["imc"].combine_first(imc_calculat)

smoke_never = find_exact_slug(df_gen, "statutul_de_fumator_nefumator")
smoke_current = find_exact_slug(df_gen, "statutul_de_fumator_fumator")
smoke_ex = find_exact_slug(df_gen, "statutul_de_fumator_ex_fumator")
smoke_unknown = find_exact_slug(df_gen, "statutul_de_fumator_necunoscut")
pac["statut_fumator"] = np.select(
    [flag(df_gen[smoke_never]).eq(1) if smoke_never else False,
     flag(df_gen[smoke_current]).eq(1) if smoke_current else False,
     flag(df_gen[smoke_ex]).eq(1) if smoke_ex else False,
     flag(df_gen[smoke_unknown]).eq(1) if smoke_unknown else False],
    ["NEFUMATOR", "FUMATOR", "EX_FUMATOR", "NECUNOSCUT"],
    default="NECUNOSCUT",
)
pa_0_19 = find_col(df_gen, ["PACHETE AN", "0"], required=False)
pa_20_40 = find_col(df_gen, ["PACHETE AN", "20"], required=False)
pa_gt40 = find_col(df_gen, ["PACHETE AN", ">40"], required=False) or find_col(df_gen, ["PACHETE AN", "40"], required=False)
pac["pachete_an_clasa"] = np.select(
    [flag(df_gen[pa_0_19]).eq(1) if pa_0_19 else False,
     flag(df_gen[pa_20_40]).eq(1) if pa_20_40 else False,
     flag(df_gen[pa_gt40]).eq(1) if pa_gt40 else False],
    [1, 2, 3],
    default=0,
)
pac["expunere_noxe"] = max_flag(df_gen, find_cols_any(df_gen, ["EXPUNERE NOXE"], exclude=["NECUNOSCUT"]))

# =========================
# SIMPTOME
# =========================

df_s = attached["SIMPTOME"]
simp = pd.DataFrame({"id_pacient": df_s["id_pacient"].astype(int)})
simp["mmrc_baseline"] = num_col(df_s, ["MMRC", "BASELINE"])
simp["mmrc_max"] = num_col(df_s, ["MMRC", "MAX"])
simp["tuse_cronica"] = flag(df_s[find_col(df_s, ["TUSE CRONICA"])])
simp["expectoratie_cronica"] = flag(df_s[find_col(df_s, ["EXPECTORATIE CRONICA"])])
simp["hemoptizie_vreodata"] = flag(df_s[find_col(df_s, ["HEMOPTIZIE", "VREODATA"])])
simp["hemoptizie_masiva"] = flag(df_s[find_col(df_s, ["HEMOPTIZIE", "MASIVA"])])
simp["raluri_bronsice"] = max_flag(df_s, find_cols_any(df_s, ["RALURI BRONSICE"]))
simp["raluri_bronhoalveolare"] = max_flag(df_s, find_cols_any(df_s, ["BRONHO-ALVEOLARE", "BRONHOALVEOLARE"]))
simp["exacerbari_an_declarate"] = num_col(df_s, ["NR. EXACERBARI/AN"])
simp["exacerbari_total_declarate"] = num_col(df_s, ["NR. EXACERBARI TOTAL"])
simp["timp_intre_2_ex"] = num_col(df_s, ["TIMP INTRE 2 EX"])
simp = simp.groupby("id_pacient", as_index=False).first()

# =========================
# BACTERIOLOGIE
# =========================

df_b = attached["BACTERIOLOGIE"]
bact = pd.DataFrame({"id_pacient": df_b["id_pacient"].astype(int)})
bact["bact_colonizare_pseudomonas"] = flag(df_b[find_col(df_b, ["COLONIZARE", "PSEUDOMONAS"])])
bact["bact_colonizare_mrsa"] = flag(df_b[find_col(df_b, ["COLONIZARE", "MRSA"])])
bact["bact_colonizare_altele"] = flag(df_b[find_col(df_b, ["COLONIZARE", "ALTELE"], required=False)]) if find_col(df_b, ["COLONIZARE", "ALTELE"], required=False) else 0
bact["bact_pseudomonas_vreodata"] = flag(df_b[find_col(df_b, ["PSEUDOMONAS VREODATA"])])
bact["bact_alte_bacterii_vreodata"] = max_flag(df_b, find_cols_any(df_b, ["ALTE BACTERII PREZENTE VREODATA"], exclude=["NTM"]))
bact["bact_ntm"] = max_flag(df_b, find_cols_any(df_b, ["NTM"]))
bact = bact.groupby("id_pacient", as_index=False).max()

# =========================
# ETIOLOGIE
# =========================

df_e = attached["ETIOLOGIE"]
etio = pd.DataFrame({"id_pacient": df_e["id_pacient"].astype(int)})
etio["etio_idiopatica"] = flag(df_e[find_col(df_e, ["IDIOPATICA"])])
etio["etio_postinfectioasa"] = max_flag(df_e, find_cols_any(df_e, ["POSTINFECTIOASA"]))
etio["etio_imunodeficienta"] = max_flag(df_e, find_cols_any(df_e, ["IMUNODEFICIENTA"]))
etio["etio_autoimuna"] = max_flag(df_e, find_cols_any(df_e, ["BOLI DE TESUT CONJUNCTIV", "AUTOIMUNE"]))
etio["etio_congenitala_non_fc"] = max_flag(df_e, find_cols_any(df_e, ["BOLI CONGENITALE NON-FC", "DISKINEZIE", "PID", "ABPA", "ANTITRIPSINA"]))
etio["etio_obstructiva"] = max_flag(df_e, find_cols_any(df_e, ["OBSTRUCTIVA"]))
etio["nr_grupe_etiologie"] = etio.drop(columns=["id_pacient"]).sum(axis=1).astype(int)
etio = etio.groupby("id_pacient", as_index=False).max()

# =========================
# COMORBIDITATI
# =========================

df_c = attached["COMORBIDITATI"]
como = pd.DataFrame({"id_pacient": df_c["id_pacient"].astype(int)})
como["como_pulmonare"] = max_flag(df_c, find_cols_any(df_c, ["PULMONARE"]))
como["como_cardio"] = max_flag(df_c, find_cols_any(df_c, ["CARDIOVASCULARE", "HTA", "HTP", "CARDIACA", "CORONARIANA", "ARITMII"]))
como["como_orl"] = max_flag(df_c, find_cols_any(df_c, ["ORL", "RINITA", "POLIPI"]))
como["como_neurologice"] = max_flag(df_c, find_cols_any(df_c, ["NEUROLOGICE", "AVC"]))
como["como_psihiatrice"] = max_flag(df_c, find_cols_any(df_c, ["PSIHIATRICE", "ANXIETATE", "DEPRESIE"]))
como["como_diabet"] = max_flag(df_c, find_cols_any(df_c, ["DIABET ZAHARAT"]))
como["como_gastro"] = max_flag(df_c, find_cols_any(df_c, ["GASTROENTEROLOGICE", "BRGE", "CIROZA"]))
como["como_osteoporoza"] = max_flag(df_c, find_cols_any(df_c, ["OSTEOPOROZA"]))
como["como_renal"] = max_flag(df_c, find_cols_any(df_c, ["BOALA CR. DE RINICHI", "RINICHI"]))
como["como_neoplazii"] = max_flag(df_c, find_cols_any(df_c, ["NEOPLAZII"]))
como["como_autoimuna"] = max_flag(df_c, find_cols_any(df_c, ["BOLI DE TESUT CONJUNCTIV", "AUTOIMUNE"]))
como["como_imunodeficienta"] = max_flag(df_c, find_cols_any(df_c, ["IMUNODEFICIENTA"]))
como["como_congenitala_non_fc"] = max_flag(df_c, find_cols_any(df_c, ["BOLI CONGENITALE NON-FC", "ANTITRIPSINA", "DISKINEZIE"]))
como["nr_grupe_comorbiditati"] = como.drop(columns=["id_pacient"]).sum(axis=1).astype(int)
como = como.groupby("id_pacient", as_index=False).max()

# =========================
# HRCT
# =========================

df_h = attached["HRCT"]
hrct = pd.DataFrame({"id_pacient": df_h["id_pacient"].astype(int)})
hrct["hrct_lobi_1"] = max_flag(df_h, find_cols_any(df_h, ["NR. LOBI AFECTATI_1"]))
hrct["hrct_lobi_2_3"] = max_flag(df_h, find_cols_any(df_h, ["2 SAU 3"]))
hrct["hrct_lobi_sever"] = max_flag(df_h, find_cols_any(df_h, [">3"]))
hrct["hrct_unilateral"] = max_flag(df_h, find_cols_any(df_h, ["UNILATERALA"]))
hrct["hrct_bilateral"] = max_flag(df_h, find_cols_any(df_h, ["BILATERALA"]))
hrct["hrct_tractiune"] = max_flag(df_h, find_cols_any(df_h, ["TRACTIUNE"]))
hrct["hrct_emfizem"] = max_flag(df_h, find_cols_any(df_h, ["EMFIZEM"]))
hrct["hrct_fibroza"] = max_flag(df_h, find_cols_any(df_h, ["FIBROZA", "REMANIERI"]))
hrct["hrct_micetom"] = max_flag(df_h, find_cols_any(df_h, ["MICETOM"]))
cols_cil = find_cols_any(df_h, ["CILINDRICE"])
cols_var = find_cols_any(df_h, ["VARICOASE"])
cols_chis = find_cols_any(df_h, ["CHISTICE"])
hrct["hrct_nr_lobi_cilindrice"] = sum_flags(df_h, cols_cil)
hrct["hrct_nr_lobi_varicoase"] = sum_flags(df_h, cols_var)
hrct["hrct_nr_lobi_chistice"] = sum_flags(df_h, cols_chis)
hrct["hrct_tip_chistic"] = max_flag(df_h, cols_chis)
hrct["hrct_nr_lobi_morfologici"] = hrct[["hrct_nr_lobi_cilindrice", "hrct_nr_lobi_varicoase", "hrct_nr_lobi_chistice"]].sum(axis=1)
hrct = hrct.groupby("id_pacient", as_index=False).max()

# =========================
# TRATAMENT RESPIRATOR
# =========================

df_t = attached["TRATAMENT RESPIRATOR"]
trat = pd.DataFrame({"id_pacient": df_t["id_pacient"].astype(int)})
trat["trat_old"] = max_flag(df_t, find_cols_any(df_t, ["OLD"]))
trat["trat_vni_cpap"] = max_flag(df_t, find_cols_any(df_t, ["VNI", "CPAP"]))
trat["trat_csi"] = max_flag(df_t, find_cols_any(df_t, ["CSI"], exclude=["CSI+LABA", "LAMA+LABA+CSI"]))
trat["trat_csi_laba"] = max_flag(df_t, find_cols_any(df_t, ["CSI+LABA"]))
trat["trat_lama"] = max_flag(df_t, find_cols_any(df_t, ["LAMA"], exclude=["LABA+LAMA", "LAMA+LABA+CSI"]))
trat["trat_laba_lama"] = max_flag(df_t, find_cols_any(df_t, ["LABA+LAMA"]))
trat["trat_saba"] = max_flag(df_t, find_cols_any(df_t, ["SABA"]))
trat["trat_terapie_tripla"] = max_flag(df_t, find_cols_any(df_t, ["LAMA+LABA+CSI"]))
trat["trat_antileucotriene"] = max_flag(df_t, find_cols_any(df_t, ["ANTILEUCOTRIENE"]))
trat["trat_mucolitice"] = max_flag(df_t, find_cols_any(df_t, ["MUCOLITICE", "EXPECTORANTE"]))
trat["trat_atb_nebulizare"] = max_flag(df_t, find_cols_any(df_t, ["ANTIBIOTERAPIE CRONICA_NEBULIZARE", "NEBULIZARE"]))
trat["trat_atb_oral"] = max_flag(df_t, find_cols_any(df_t, ["ANTIBIOTERAPIE CRONICA_ORAL", "_ORAL_"]))
trat["trat_atb_cronic"] = trat[["trat_atb_nebulizare", "trat_atb_oral"]].max(axis=1)
trat = trat.groupby("id_pacient", as_index=False).max()

# =========================
# VIZITE - longitudinal
# =========================

df_v = attached["VIZITE"].copy()
df_v["tip_vizita_ex"] = num_col(df_v, ["TIP VIZITA"])
df_v["vizita_spitalizare"] = max_flag(df_v, find_cols_any(df_v, ["SPITALIZARE"]))
df_v["mmrc_vizita"] = num_col(df_v, ["SIMPTOMATOLOGIE", "MMRC"])
df_v["tuse_vizita"] = max_flag(df_v, find_cols_any(df_v, ["SIMPTOMATOLOGIE", "TUSE"]))
df_v["sputa_vizita"] = max_flag(df_v, find_cols_any(df_v, ["SIMPTOMATOLOGIE", "SPUTA"]))
df_v["hemoptizie_vizita"] = max_flag(df_v, find_cols_any(df_v, ["SIMPTOMATOLOGIE", "HEMOPTIZIE"]))
df_v["murray_mucos"] = max_flag(df_v, find_cols_any(df_v, ["MURRAY", "MUCOS"], exclude=["MUCO-PURULENT"]))
df_v["murray_mucopurulent"] = max_flag(df_v, find_cols_any(df_v, ["MUCO-PURULENT"]))
df_v["murray_purulent"] = max_flag(df_v, find_cols_any(df_v, ["PURULENT"], exclude=["MUCO-PURULENT"]))
df_v["stetacustic_normal"] = max_flag(df_v, find_cols_any(df_v, ["STETACUSTIC", "NORMAL"]))
df_v["stetacustic_r_bronsice"] = max_flag(df_v, find_cols_any(df_v, ["R.BRONSICE", "BRONSICE"]))
df_v["stetacustic_r_bronhoalveolare"] = max_flag(df_v, find_cols_any(df_v, ["BRONHO-ALVEOLARE"]))
df_v["leucocite_vizita"] = num_col(df_v, ["LEUCOCITE"])
df_v["pcr_vizita"] = num_col(df_v, ["PCR"])
df_v["vsh_vizita"] = num_col(df_v, ["VSH"])
df_v["fvc_vizita"] = num_col(df_v, ["FVC_PRE_%"])
df_v["fev1_vizita"] = num_col(df_v, ["FEV1_PRE_%"])
df_v["ipb_vizita"] = num_col(df_v, ["IPB"])
for agent in ["PSEUDOMONAS", "MSSA", "MRSA", "KLEBSIELLA", "E.COLI", "ACINETOBACTER", "S.PNEUMONIAE", "CANDIDA"]:
    df_v[f"viz_bact_{slugify_col(agent)}"] = max_flag(df_v, find_cols_any(df_v, ["BACTERIOLOGIC", agent]))

viz_agg_spec = {
    "nr_total_vizite": ("id_pacient", "count"),
    "nr_vizite_exacerbare": ("tip_vizita_ex", lambda x: int((x == 1).sum())),
    "nr_vizite_stabile": ("tip_vizita_ex", lambda x: int((x == 0).sum())),
    "nr_spitalizari_vizite": ("vizita_spitalizare", "sum"),
    "mmrc_max_vizite": ("mmrc_vizita", "max"),
    "mmrc_medie_vizite": ("mmrc_vizita", "mean"),
    "tuse_vizite": ("tuse_vizita", "max"),
    "sputa_vizite": ("sputa_vizita", "max"),
    "hemoptizie_vizite": ("hemoptizie_vizita", "max"),
    "murray_mucos_vizite": ("murray_mucos", "max"),
    "murray_mucopurulent_vizite": ("murray_mucopurulent", "max"),
    "murray_purulent_vizite": ("murray_purulent", "max"),
    "stetacustic_normal_vizite": ("stetacustic_normal", "max"),
    "stetacustic_r_bronsice_vizite": ("stetacustic_r_bronsice", "max"),
    "stetacustic_r_bronhoalveolare_vizite": ("stetacustic_r_bronhoalveolare", "max"),
    "leucocite_max_vizite": ("leucocite_vizita", "max"),
    "leucocite_medie_vizite": ("leucocite_vizita", "mean"),
    "max_pcr_vizite": ("pcr_vizita", "max"),
    "pcr_medie_vizite": ("pcr_vizita", "mean"),
    "max_vsh_vizite": ("vsh_vizita", "max"),
    "vsh_medie_vizite": ("vsh_vizita", "mean"),
    "min_fev1_procent_vizite": ("fev1_vizita", "min"),
    "medie_fev1_procent_vizite": ("fev1_vizita", "mean"),
    "min_fvc_procent_vizite": ("fvc_vizita", "min"),
    "medie_fvc_procent_vizite": ("fvc_vizita", "mean"),
    "min_ipb_vizite": ("ipb_vizita", "min"),
}
for agent in ["PSEUDOMONAS", "MSSA", "MRSA", "KLEBSIELLA", "E.COLI", "ACINETOBACTER", "S.PNEUMONIAE", "CANDIDA"]:
    col = f"viz_bact_{slugify_col(agent)}"
    viz_agg_spec[f"{col}_vreodata"] = (col, "max")
viz = df_v.groupby("id_pacient").agg(**viz_agg_spec).reset_index()

# =========================
# EXACERBARI
# =========================

df_x = attached["EXACERBARI"]
ex = pd.DataFrame({"id_pacient": df_x["id_pacient"].astype(int)})
ex["ex_usor"] = max_flag(df_x, find_cols_any(df_x, ["USOARA"]))
ex["ex_moderat"] = max_flag(df_x, find_cols_any(df_x, ["MODERATA"]))
ex["ex_sever"] = max_flag(df_x, find_cols_any(df_x, ["SEVERA", "SPITALIZARE"]))
ex["ex_pseudomonas"] = max_flag(df_x, find_cols_any(df_x, ["PSEUDOMONAS"]))
ex["ex_agent_altele"] = max_flag(df_x, find_cols_any(df_x, ["AGENT", "ALTELE"]))
ex["ex_atb_oral"] = max_flag(df_x, find_cols_any(df_x, ["ANTIBIOTIC", "ORAL"]))
ex["ex_atb_inhalator"] = max_flag(df_x, find_cols_any(df_x, ["INHALATOR"]))
ex["ex_atb_iv"] = max_flag(df_x, find_cols_any(df_x, ["ANTIBIOTIC", "IV"]))
ex["ex_atb_5_7"] = max_flag(df_x, find_cols_any(df_x, ["5-7 ZILE"]))
ex["ex_atb_8_14"] = max_flag(df_x, find_cols_any(df_x, ["8-14 ZILE"]))
ex["ex_atb_gt_14"] = max_flag(df_x, find_cols_any(df_x, [">14 ZILE"]))
ex["ex_corticoterapie"] = max_flag(df_x, find_cols_any(df_x, ["CORTICOTERAPIE"]))
ex_agg = ex.groupby("id_pacient").agg(
    nr_exacerbari_inregistrate=("id_pacient", "count"),
    nr_exacerbari_usoare=("ex_usor", "sum"),
    nr_exacerbari_moderate=("ex_moderat", "sum"),
    nr_exacerbari_severe=("ex_sever", "sum"),
    exacerbari_pseudomonas=("ex_pseudomonas", "max"),
    exacerbari_agent_altele=("ex_agent_altele", "max"),
    exacerbari_atb_oral=("ex_atb_oral", "max"),
    exacerbari_atb_inhalator=("ex_atb_inhalator", "max"),
    exacerbari_atb_iv=("ex_atb_iv", "max"),
    exacerbari_atb_5_7=("ex_atb_5_7", "sum"),
    exacerbari_atb_8_14=("ex_atb_8_14", "sum"),
    exacerbari_atb_gt_14=("ex_atb_gt_14", "sum"),
    exacerbari_corticoterapie=("ex_corticoterapie", "max"),
).reset_index()

# =========================
# PROBE VENTILATORII
# =========================

df_pv = attached["PROBE VENTILATORII"]
pv = pd.DataFrame({"id_pacient": df_pv["id_pacient"].astype(int)})
for out_col, terms in {
    "pv_test_bd_baseline": ["SPIROMETRIE BASELINE", "TEST BD"],
    "pv_fev1_baseline_pct": ["SPIROMETRIE BASELINE", "FEV1_PRE_%"],
    "pv_fvc_baseline_pct": ["SPIROMETRIE BASELINE", "FVC_PRE_%"],
    "pv_ipb_baseline": ["SPIROMETRIE BASELINE", "IPB"],
    "pv_test_bd_final": ["SPIROMETRIE FINAL", "TEST BD"],
    "pv_fev1_final_pct": ["SPIROMETRIE FINAL", "FEV1_PRE_%"],
    "pv_fvc_final_pct": ["SPIROMETRIE FINAL", "FVC_PRE_%"],
    "pv_ipb_final": ["SPIROMETRIE FINAL", "IPB"],
    "pv_delta_fev1_pct": ["DELTA_FEV1_PRE_%"],
    "pv_delta_fvc_pct": ["DELTA_FVC_PRE_%"],
}.items():
    col = find_col(df_pv, terms, required=False) or find_col(df_pv, [t.replace("_", " ") for t in terms], required=False)
    pv[out_col] = to_num(df_pv[col]) if col else np.nan
pv["pv_interp_normal"] = max_flag(df_pv, find_cols_any(df_pv, ["INTERPRETARE", "NORMAL"]))
pv["pv_interp_dvo"] = max_flag(df_pv, find_cols_any(df_pv, ["INTERPRETARE", "DVO"]))
pv["pv_interp_dvr"] = max_flag(df_pv, find_cols_any(df_pv, ["INTERPRETARE", "DVR"]))
pv["pv_interp_dvm"] = max_flag(df_pv, find_cols_any(df_pv, ["INTERPRETARE", "DVM"]))
pv["pv_disfunctie_usoara"] = max_flag(df_pv, find_cols_any(df_pv, ["DISFUNCTIE", "USOARA"]))
pv["pv_disfunctie_moderata"] = max_flag(df_pv, find_cols_any(df_pv, ["DISFUNCTIE", "MODERATA"], exclude=["MODERAT-SEVERA"]))
pv["pv_disfunctie_moderat_severa"] = max_flag(df_pv, find_cols_any(df_pv, ["MODERAT-SEVERA"]))
pv["pv_disfunctie_severa"] = max_flag(df_pv, find_cols_any(df_pv, ["DISFUNCTIE", "SEVERA"], exclude=["MODERAT-SEVERA", "F. SEVERA"]))
pv["pv_disfunctie_foarte_severa"] = max_flag(df_pv, find_cols_any(df_pv, ["F. SEVERA"]))
pv["pv_dlco"] = df_pv[find_cols_any(df_pv, ["DLCO"])].apply(to_num).max(axis=1) if find_cols_any(df_pv, ["DLCO"]) else np.nan
pv["pv_6mwt"] = df_pv[find_cols_any(df_pv, ["6MWT"])].apply(to_num).max(axis=1) if find_cols_any(df_pv, ["6MWT"]) else np.nan
# Workbook-ul actual are o singura linie per pacient pentru PROBE VENTILATORII,
# cu baseline si final pe coloane separate. Totusi, pentru robustete la versiuni viitoare:
# - baseline se pastreaza ca prima valoare disponibila;
# - final se pastreaza ca ultima valoare disponibila;
# - interpretarea/disfunctia se combina prin max;
# - delta se mediaza daca apar duplicate.
pv_agg_rules = {}
for c in pv.columns:
    if c == "id_pacient":
        continue
    if c.startswith("pv_interp") or c.startswith("pv_disfunctie") or c.startswith("pv_test"):
        pv_agg_rules[c] = "max"
    elif "baseline" in c:
        pv_agg_rules[c] = "first"
    elif "final" in c:
        pv_agg_rules[c] = "last"
    elif "delta" in c:
        pv_agg_rules[c] = "mean"
    else:
        pv_agg_rules[c] = "mean"
pv = pv.groupby("id_pacient", as_index=False).agg(pv_agg_rules)

# =========================
# BIOLOGIC
# =========================

df_bio = attached["BIOLOGIC"]
bio = pd.DataFrame({"id_pacient": df_bio["id_pacient"].astype(int)})
for out_col, terms in {
    "bio_albumine_pct": ["ALBUMINE"],
    "bio_a1_globuline_pct": ["A1 GLOBULINE"],
    "bio_a2_globuline_pct": ["A2 GLOBULINE"],
    "bio_b_globuline_pct": ["BGLOBULINE"],
    "bio_g_globuline_pct": ["G GLOBULINE"],
    "bio_iga": ["IMUNOGRAMA", "IGA"],
    "bio_igg": ["IMUNOGRAMA", "IGG"],
    "bio_igm": ["IMUNOGRAMA", "IGM"],
    "bio_ige_total": ["IGE TOTAL"],
    "bio_ige_aspergillus": ["ASPERGILLUS"],
}.items():
    col = find_col(df_bio, terms, required=False)
    bio[out_col] = to_num(df_bio[col]) if col else np.nan
bio["bio_factor_reumatoid_pozitiv"] = max_flag(df_bio, find_cols_any(df_bio, ["FACTOR REUMATOID", "POZITIV"]))
bio["bio_alfa1_antitripsina_pozitiv"] = max_flag(df_bio, find_cols_any(df_bio, ["ALFA 1 ANTITRIPSINA", "POZITIV"]))
bio["bio_test_sudorii_pozitiv"] = max_flag(df_bio, find_cols_any(df_bio, ["TESTUL SUDORII", "POZITIV"]))
bio = bio.groupby("id_pacient", as_index=False).first()

# =========================
# SCORURI EXISTENTE
# =========================
# In workbook-ul primit, foaia SCORURI are coloane definite, dar valorile sunt goale.
# O salvam ca *_clean pentru trasabilitate; nu o folosim ca input AI pentru ca ar fi inutila sau redundant cu scorurile calculate.

df_sc = attached["SCORURI"]
scor = pd.DataFrame({"id_pacient": df_sc["id_pacient"].astype(int)})
for out_col, patterns in {
    "scor_qol_b_original": ["QOL"],
    "scor_eq5d_original": ["EQ-5D"],
    "scor_sgrq_original": ["SGRQ"],
    "scor_cough_nrs_original": ["COUGH"],
    "scor_snot22_original": ["SNOT"],
    "scor_caat_original": ["CAAT"],
    "scor_baci_original": ["BACI"],
}.items():
    cols = find_cols_any(df_sc, patterns)
    scor[out_col] = df_sc[cols].apply(to_num).max(axis=1) if cols else np.nan
scor = scor.groupby("id_pacient", as_index=False).first()
scor_nonempty = scor[["id_pacient"] + [c for c in scor.columns if c != "id_pacient" and scor[c].notna().sum() > 0]]

# =========================
# MASTER FEATURES
# =========================

features = pac.copy()
for name, table in [
    ("simptome", simp),
    ("bacteriologie", bact),
    ("etiologie", etio),
    ("comorbiditati", como),
    ("hrct", hrct),
    ("tratament", trat),
    ("vizite", viz),
    ("exacerbari", ex_agg),
    ("probe_ventilatorii", pv),
    ("biologic", bio),
]:
    features = safe_merge(features, table, name)

# Daca SCORURI are valori reale in versiuni viitoare, le pastram in features_ai ca original/reference.
if len(scor_nonempty.columns) > 1:
    features = safe_merge(features, scor_nonempty, "scoruri")

zero_fill_prefixes = (
    "bact_", "etio_", "como_", "hrct_", "trat_", "nr_total_vizite", "nr_vizite_",
    "nr_spitalizari_", "nr_exacerbari_", "exacerbari_", "pv_interp_", "pv_disfunctie_",
    "bio_factor_", "bio_alfa", "bio_test", "expunere_noxe", "raluri_", "hemoptizie_",
    "tuse_", "expectoratie_", "murray_", "stetacustic_", "viz_bact_", "sputa_",
)
for c in features.columns:
    if c.startswith(zero_fill_prefixes):
        features[c] = features[c].fillna(0)

# Variabile canonice, ca sa nu duplicam inutil aceeasi informatie.
features["fev1_min_final"] = features["pv_fev1_baseline_pct"].combine_first(features["min_fev1_procent_vizite"])
features["fvc_min_final"] = features["pv_fvc_baseline_pct"].combine_first(features["min_fvc_procent_vizite"])
features["ipb_min_final"] = features["pv_ipb_baseline"].combine_first(features["min_ipb_vizite"])
features["mmrc_max_final"] = features["mmrc_max"].combine_first(features["mmrc_max_vizite"])
features["nr_spitalizari_final"] = features["nr_exacerbari_severe"].combine_first(features["nr_spitalizari_vizite"]).fillna(0)
features["exacerbari_an_final"] = features["exacerbari_an_declarate"].combine_first(features["nr_exacerbari_inregistrate"])
features["pseudomonas_final"] = features[["bact_pseudomonas_vreodata", "bact_colonizare_pseudomonas", "exacerbari_pseudomonas", "viz_bact_pseudomonas_vreodata"]].max(axis=1)

# Calcul scoruri clinice - rezultat derivat, NU input pentru clustering.
# Pastram doua variante:
# - *_partial: calculeaza cu informatia disponibila, fara sa penalizeze lipsurile; util exploratoriu.
# - *_calculat: varianta stricta; devine NaN daca lipsesc componente esentiale.
def calc_faced_partial(row: pd.Series) -> int:
    pts = 0
    fev = row.get("fev1_min_final")
    if pd.notna(fev) and fev < 50:
        pts += 2
    if pd.notna(row.get("varsta")) and row["varsta"] > 70:
        pts += 1
    if row.get("pseudomonas_final", 0) == 1:
        pts += 1
    if row.get("hrct_lobi_sever", 0) == 1:
        pts += 1
    mmrc = row.get("mmrc_max_final")
    if pd.notna(mmrc) and mmrc >= 3:
        pts += 1
    return pts


def calc_bsi_partial(row: pd.Series) -> int:
    pts = 0
    v = row.get("varsta")
    if pd.notna(v):
        if 50 <= v <= 69:
            pts += 2
        elif 70 <= v <= 79:
            pts += 4
        elif v >= 80:
            pts += 6
    imc = row.get("imc")
    if pd.notna(imc) and imc < 18.5:
        pts += 2
    fev = row.get("fev1_min_final")
    if pd.notna(fev):
        if 50 <= fev <= 80:
            pts += 1
        elif 30 <= fev < 50:
            pts += 2
        elif fev < 30:
            pts += 3
    if row.get("nr_spitalizari_final", 0) > 0:
        pts += 4
    ex_an = row.get("exacerbari_an_final")
    if pd.notna(ex_an) and ex_an >= 3:
        pts += 2
    if row.get("pseudomonas_final", 0) == 1:
        pts += 3
    if row.get("hrct_lobi_sever", 0) == 1 or row.get("hrct_tip_chistic", 0) == 1:
        pts += 1
    return pts


def missing_count(row: pd.Series, cols: list[str]) -> int:
    return int(sum(pd.isna(row.get(c)) for c in cols))

features["faced_missing_components"] = features.apply(
    lambda r: missing_count(r, ["fev1_min_final", "varsta", "mmrc_max_final"]), axis=1
)
features["bsi_missing_components"] = features.apply(
    lambda r: missing_count(r, ["varsta", "imc", "fev1_min_final", "exacerbari_an_final"]), axis=1
)
features["faced_score_partial"] = features.apply(calc_faced_partial, axis=1)
features["bsi_score_partial"] = features.apply(calc_bsi_partial, axis=1)
features["faced_score_calculat"] = features["faced_score_partial"].where(features["faced_missing_components"].eq(0), np.nan)
features["bsi_score_calculat"] = features["bsi_score_partial"].where(features["bsi_missing_components"].eq(0), np.nan)
features["faced_calculabil_complet"] = features["faced_missing_components"].eq(0).astype(int)
features["bsi_calculabil_complet"] = features["bsi_missing_components"].eq(0).astype(int)
features["severitate_faced_calculata"] = pd.cut(
    features["faced_score_calculat"], bins=[-1, 2, 4, 99], labels=["USOARA", "MODERATA", "SEVERA"]
).astype("object")
features["severitate_bsi_calculata"] = pd.cut(
    features["bsi_score_calculat"], bins=[-1, 4, 8, 99], labels=["USOARA", "MODERATA", "SEVERA"]
).astype("object")
# Curatam master-ul de coloane complet goale/constante, dar pastram id si scorurile calculate.
features = drop_empty_or_constant(features, keep={"id_pacient", "sex", "statut_fumator", "faced_score_calculat", "bsi_score_calculat", "severitate_faced_calculata", "severitate_bsi_calculata"})

# Verificari anti-leakage identificatori.
assert features["id_pacient"].is_unique, "features_ai trebuie sa aiba o singura linie per pacient."
for forbidden in ["cnp", "nume", "prenume"]:
    bad_cols = [c for c in features.columns if forbidden in c.lower()]
    assert not bad_cols, f"Coloane interzise in features_ai: {bad_cols}"

# =========================
# DATASET CLUSTERING
# =========================

# Alegem coloane canonice. Nu includem: id ca feature, BSI/FACED, severitati, scoruri originale.
# Separat construim doua matrici:
# 1) baseline/conservatoare: evita agregarile post-hoc din urmarirea pacientului;
# 2) descriptiva: include si agregari longitudinale, utila pentru fenotipare retrospectiva.
preferred_cluster_cols = [
    "sex", "varsta", "imc", "statut_fumator", "pachete_an_clasa", "expunere_noxe",
    "mmrc_max_final", "tuse_cronica", "expectoratie_cronica", "hemoptizie_vreodata", "raluri_bronsice", "raluri_bronhoalveolare",
    "exacerbari_an_final", "nr_exacerbari_inregistrate", "nr_exacerbari_severe", "nr_spitalizari_final",
    "pseudomonas_final", "bact_colonizare_mrsa", "bact_ntm", "bact_alte_bacterii_vreodata",
    "etio_idiopatica", "etio_postinfectioasa", "etio_imunodeficienta", "etio_autoimuna", "etio_congenitala_non_fc", "etio_obstructiva", "nr_grupe_etiologie",
    "como_pulmonare", "como_cardio", "como_orl", "como_neurologice", "como_psihiatrice", "como_diabet", "como_gastro", "como_renal", "como_neoplazii", "como_autoimuna", "nr_grupe_comorbiditati",
    "hrct_lobi_sever", "hrct_bilateral", "hrct_tractiune", "hrct_emfizem", "hrct_tip_chistic", "hrct_nr_lobi_cilindrice", "hrct_nr_lobi_varicoase", "hrct_nr_lobi_chistice",
    "trat_old", "trat_vni_cpap", "trat_csi", "trat_csi_laba", "trat_lama", "trat_laba_lama", "trat_saba", "trat_terapie_tripla", "trat_mucolitice", "trat_atb_cronic",
    "nr_total_vizite", "nr_vizite_exacerbare", "leucocite_max_vizite", "max_pcr_vizite", "max_vsh_vizite", "fev1_min_final", "fvc_min_final", "ipb_min_final",
    "pv_interp_dvo", "pv_interp_dvr", "pv_interp_dvm", "pv_disfunctie_moderata", "pv_disfunctie_moderat_severa", "pv_disfunctie_severa", "pv_disfunctie_foarte_severa",
]

POSTHOC_CLUSTER_COLS = {
    # rezultate acumulate pe perioada de urmarire; bune pentru clustering descriptiv, riscante pentru predictie baseline
    "nr_total_vizite", "nr_vizite_exacerbare", "nr_exacerbari_inregistrate", "nr_exacerbari_severe",
    "nr_spitalizari_final", "leucocite_max_vizite", "max_pcr_vizite", "max_vsh_vizite",
}


def is_binary_like(s: pd.Series) -> bool:
    vals = set(pd.to_numeric(s.dropna(), errors="coerce").dropna().unique().tolist())
    return len(vals) > 0 and vals.issubset({0, 1})


def build_cluster_table(source: pd.DataFrame, cols: list[str], dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construieste o matrice numerica pentru clustering si un raport de preprocesare.

    Principii:
    - categoriile sunt one-hot encoded;
    - pentru variabile cu missing_rate > 20%, adaugam indicator *_missing;
    - binarele/counturile se impută cu 0 doar după ce am păstrat indicatorul de lipsă;
    - variabilele continue se impută cu mediana;
    - standardizam doar variabilele continue, nu dummy/flag-urile 0/1.
    """
    base = source[["id_pacient"] + [c for c in cols if c in source.columns]].copy()
    raw_input = base.drop(columns=["id_pacient"]).copy()
    raw_input = pd.get_dummies(raw_input, columns=[c for c in ["sex", "statut_fumator"] if c in raw_input.columns], dummy_na=True, dtype=int)

    report_rows = []
    prepared = raw_input.copy()

    # Adaugam missing indicators inainte de imputare pentru variabile informative cu lipsuri multe.
    for c in list(prepared.columns):
        miss = float(prepared[c].isna().mean())
        if miss > 0.20:
            prepared[f"{c}_missing"] = prepared[c].isna().astype(int)

    selected_cols = []
    for c in prepared.columns:
        miss = float(prepared[c].isna().mean())
        nun = int(prepared[c].nunique(dropna=True))
        binary = is_binary_like(prepared[c])
        reason = "selectat"
        if miss > MAX_MISSING_CLUSTERING and not c.endswith("_missing"):
            reason = f"exclus_missing_{miss:.0%}"
        elif nun <= 1:
            reason = "exclus_constanta"
        else:
            selected_cols.append(c)
        report_rows.append({
            "dataset": dataset_name,
            "feature": c,
            "missing_rate": miss,
            "unique_values": nun,
            "binary_like": int(binary),
            "status": reason,
        })

    numeric = prepared[selected_cols].copy()
    imputation_rows = []
    for c in numeric.columns:
        binary = is_binary_like(numeric[c]) or c.endswith("_missing")
        if binary or c.startswith(("nr_", "como_", "etio_", "hrct_", "trat_", "bact_", "pv_interp_", "pv_disfunctie_", "tuse_", "expectoratie_", "hemoptizie_", "raluri_", "expunere_", "pseudomonas_")):
            numeric[c] = numeric[c].fillna(0)
            imputation = "zero_after_missing_indicator"
        else:
            median = numeric[c].median(skipna=True)
            numeric[c] = numeric[c].fillna(median)
            imputation = f"median_{median}"
        imputation_rows.append({"dataset": dataset_name, "feature": c, "imputation": imputation})

    # Standardizam doar continuele; binarele/dummy-urile raman 0/1.
    scaled = numeric.copy()
    scaled_rows = []
    for c in scaled.columns:
        binary = is_binary_like(scaled[c]) or c.endswith("_missing")
        if binary:
            scaled_rows.append({"dataset": dataset_name, "feature": c, "scaled": 0})
            continue
        std = scaled[c].std(ddof=0)
        mean = scaled[c].mean()
        if pd.notna(std) and std != 0:
            scaled[c] = (scaled[c] - mean) / std
            scaled_rows.append({"dataset": dataset_name, "feature": c, "scaled": 1})
        else:
            scaled[c] = 0
            scaled_rows.append({"dataset": dataset_name, "feature": c, "scaled": 0})

    table = pd.concat([base[["id_pacient"]].reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
    assert not table.drop(columns=["id_pacient"]).isna().any().any(), f"{dataset_name} nu trebuie sa contina NaN."

    report = pd.DataFrame(report_rows).merge(pd.DataFrame(imputation_rows), on=["dataset", "feature"], how="left")
    report = report.merge(pd.DataFrame(scaled_rows), on=["dataset", "feature"], how="left")
    raw_table = pd.concat([base[["id_pacient"]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)
    return table, raw_table, report

baseline_cluster_cols = [c for c in preferred_cluster_cols if c not in POSTHOC_CLUSTER_COLS]
descriptive_cluster_cols = preferred_cluster_cols.copy()

features_clustering, features_clustering_raw, clustering_report_baseline = build_cluster_table(features, baseline_cluster_cols, "baseline")
features_clustering_descriptive, features_clustering_descriptive_raw, clustering_report_descriptive = build_cluster_table(features, descriptive_cluster_cols, "descriptive")
clustering_feature_report = pd.concat([clustering_report_baseline, clustering_report_descriptive], ignore_index=True)
# =========================
# RAPOARTE
# =========================

report_rows = []
for sh, df in attached.items():
    report_rows.append({
        "sheet": sh,
        "randuri_dupa_mapare": len(df),
        "pacienti_unici_dupa_mapare": int(df["id_pacient"].nunique()),
        "coloane_initiale": len(raw[sh].columns),
        "header_rows": len(HEADER_LEVELS.get(sh, DEFAULT_HEADER)),
        "tabel_sql": SHEETS[sh],
    })
quality_report = pd.DataFrame(report_rows)
quality_report.loc[len(quality_report)] = {
    "sheet": "features_ai",
    "randuri_dupa_mapare": len(features),
    "pacienti_unici_dupa_mapare": int(features["id_pacient"].nunique()),
    "coloane_initiale": len(features.columns),
    "header_rows": np.nan,
    "tabel_sql": "features_ai",
}
quality_report.loc[len(quality_report)] = {
    "sheet": "features_clustering_baseline",
    "randuri_dupa_mapare": len(features_clustering),
    "pacienti_unici_dupa_mapare": int(features_clustering["id_pacient"].nunique()),
    "coloane_initiale": len(features_clustering.columns),
    "header_rows": np.nan,
    "tabel_sql": "features_clustering",
}
quality_report.loc[len(quality_report)] = {
    "sheet": "features_clustering_descriptive",
    "randuri_dupa_mapare": len(features_clustering_descriptive),
    "pacienti_unici_dupa_mapare": int(features_clustering_descriptive["id_pacient"].nunique()),
    "coloane_initiale": len(features_clustering_descriptive.columns),
    "header_rows": np.nan,
    "tabel_sql": "features_clustering_descriptive",
}

feature_quality_report = pd.DataFrame({
    "feature": features.columns,
    "missing_rate": [float(features[c].isna().mean()) for c in features.columns],
    "unique_values": [int(features[c].nunique(dropna=True)) for c in features.columns],
})
feature_quality_report["used_in_clustering"] = feature_quality_report["feature"].isin(features_clustering.columns)

# =========================
# SAVE
# =========================

CALE_DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(CALE_DB) as conn:
    for sh, table_name in SHEETS.items():
        anonymized_table(attached[sh]).to_sql(table_name, conn, if_exists="replace", index=False)

    pac.to_sql("pacienti", conn, if_exists="replace", index=False)
    simp.to_sql("simptome_features", conn, if_exists="replace", index=False)
    bact.to_sql("bacteriologie_features", conn, if_exists="replace", index=False)
    etio.to_sql("etiologie_features", conn, if_exists="replace", index=False)
    como.to_sql("comorbiditati_features", conn, if_exists="replace", index=False)
    hrct.to_sql("hrct_features", conn, if_exists="replace", index=False)
    trat.to_sql("tratament_features", conn, if_exists="replace", index=False)
    viz.to_sql("vizite_agg", conn, if_exists="replace", index=False)
    ex_agg.to_sql("exacerbari_agg", conn, if_exists="replace", index=False)
    pv.to_sql("probe_ventilatorii_features", conn, if_exists="replace", index=False)
    bio.to_sql("biologic_features", conn, if_exists="replace", index=False)
    scor.to_sql("scoruri_features", conn, if_exists="replace", index=False)
    features.to_sql("features_ai", conn, if_exists="replace", index=False)
    features_clustering.to_sql("features_clustering", conn, if_exists="replace", index=False)
    features_clustering_raw.to_sql("features_clustering_raw", conn, if_exists="replace", index=False)
    features_clustering_descriptive.to_sql("features_clustering_descriptive", conn, if_exists="replace", index=False)
    features_clustering_descriptive_raw.to_sql("features_clustering_descriptive_raw", conn, if_exists="replace", index=False)
    quality_report.to_sql("data_quality_report", conn, if_exists="replace", index=False)
    feature_quality_report.to_sql("feature_quality_report", conn, if_exists="replace", index=False)
    clustering_feature_report.to_sql("clustering_feature_report", conn, if_exists="replace", index=False)

print("\n✅ Pipeline finalizat.")
print(f"DB salvata la: {CALE_DB}")
print(f"features_ai: {features.shape[0]} pacienti x {features.shape[1]} coloane")
print(f"features_clustering baseline: {features_clustering.shape[0]} pacienti x {features_clustering.shape[1]} coloane (include id_pacient pentru mapare)")
print(f"features_clustering_descriptive: {features_clustering_descriptive.shape[0]} pacienti x {features_clustering_descriptive.shape[1]} coloane")
print("\nTabele integrate:")
print(quality_report.to_string(index=False))
print("\nPrimele 3 randuri din features_clustering:")
print(features_clustering.head(3).to_string(index=False))
