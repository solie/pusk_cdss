"""Shared data preparation for the ICD-10 comparison.

One definition of the dataset, the label set, the folds and the sample, so the
TF-IDF + LinearSVC baseline and the Jeff zero-shot model are scored on exactly
the same rows. Import it from the other scripts; do not fork the logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
TARGET_COVERAGE = 0.75

NUMERIC_FIELDS = ["bb", "tb", "sistole", "diastole", "age"]
CATEGORICAL_FIELDS = ["jkl"]
TEXT_FIELD = "anamnesa"
FEATURE_FIELDS = [TEXT_FIELD] + NUMERIC_FIELDS + CATEGORICAL_FIELDS

# Clinical plausibility ranges, in the units the file uses. Values outside are
# set to missing and then imputed; the notebook kept a 258,500 kg weight and a
# systolic pressure of 1157 as training features.
PLAUSIBLE = {
    "bb": (0.5, 400.0),        # kg: newborn to extreme adult
    "tb": (20.0, 250.0),       # cm
    "sistole": (40.0, 300.0),  # mmHg
    "diastole": (20.0, 200.0),  # mmHg
    "age": (0.0, 120.0),       # years
}


@dataclass
class Dataset:
    """Modelling table plus the fold assignment every model must reuse."""

    frame: pd.DataFrame
    labels: list[str]
    fold_of_row: np.ndarray
    n_splits: int
    report: dict

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[FEATURE_FIELDS]

    @property
    def y(self) -> pd.Series:
        return self.frame["label"]

    @property
    def groups(self) -> pd.Series:
        return self.frame["group"]

    def fold(self, index: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train and validation frames for one fold, as X_train, y_train, X_val, y_val."""
        mask = self.fold_of_row == index
        train, valid = self.frame[~mask], self.frame[mask]
        return train[FEATURE_FIELDS], train["label"], valid[FEATURE_FIELDS], valid["label"]


def clean_icd_code(value: object) -> str | float:
    """Primary diagnosis code of one `kode_penyakit` cell, upper-cased."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip().upper()
    except (ValueError, TypeError):
        pass
    stripped = text.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
    first = stripped.split(",")[0].strip().upper()
    return first if first else np.nan


def target_codes(counts: pd.Series, coverage: float = TARGET_COVERAGE) -> list[str]:
    """The most frequent codes covering at least `coverage` of the visits."""
    share = counts / counts.sum()
    cumulative = share.cumsum()
    below = cumulative[cumulative < coverage]
    keep = int(len(below)) + 1  # the code that crosses the threshold is included
    return list(counts.index[:keep])


def load_raw(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    frame["icd"] = frame["kode_penyakit"].apply(clean_icd_code)
    return frame


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Age, numeric coercion, plausibility clipping and text cleaning."""
    frame = frame.copy()
    seen = pd.to_datetime(frame["tanggal"], errors="coerce")
    born = pd.to_datetime(frame["tg_lahir"], errors="coerce")
    frame["age"] = (seen - born).dt.days / 365.25
    for column in ["bb", "tb", "sistole", "diastole"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column, (low, high) in PLAUSIBLE.items():
        frame.loc[~frame[column].between(low, high), column] = np.nan
    frame["anamnesa"] = frame["anamnesa"].fillna("").astype(str).str.strip()
    frame["jkl"] = frame["jkl"].fillna("unknown").astype(str)
    return frame


def group_key(frame: pd.DataFrame, group_field: str) -> pd.Series:
    """Rows that must stay in one fold.

    patient: the same person's visits, keyed by registration number (no_reg).
    village: the administrative kelurahan (kdesa), enabling spatial/geographical validation.
    text: the same complaint wording, which catches template records shared by
          different patients (536 complaint texts appear under more than one
          registration number).
    """
    if group_field == "patient":
        return frame["no_reg"].astype(str)
    if group_field == "village":
        return frame["kdesa"].fillna("UNKNOWN").astype(str)
    if group_field == "text":
        return frame["anamnesa"] + "||" + frame["jkl"] + "||" + frame["age"].round(2).astype(str)
    raise ValueError(f"Unknown group field {group_field!r}; choose 'patient', 'village', or 'text'")


def subsample_groups(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Keep whole groups, in a seeded random order, until about `n_rows` rows.

    Sampling by group rather than by row keeps every patient's visits together,
    so the fold split stays honest on the reduced dataset too.
    """
    sizes = frame.groupby("group", sort=False).size()
    order = sizes.sample(frac=1.0, random_state=seed).index
    keep, total = [], 0
    for name in order:
        size = int(sizes[name])
        if total + size > n_rows and total:
            continue
        keep.append(name)
        total += size
        if total >= n_rows:
            break
    return frame[frame["group"].isin(set(keep))].copy()


def build_dataset(
    csv_path: str,
    *,
    n_splits: int = 5,
    coverage: float = TARGET_COVERAGE,
    sample: int = 0,
    group_field: str = "patient",
    seed: int = RANDOM_STATE,
) -> Dataset:
    """The modelling table: labels, features, folds. `sample=0` keeps every row."""
    raw = load_raw(csv_path)
    counts = raw["icd"].value_counts()
    kept = target_codes(counts, coverage)
    covered = float(counts[kept].sum() / counts.sum())

    frame = raw[raw["icd"].isin(kept)].copy()
    frame = engineer(frame)
    frame["label"] = frame["icd"]
    frame["group"] = group_key(frame, group_field)
    frame = frame[frame["group"].isin(frame["group"].dropna().unique())].reset_index(drop=True)

    # Impute with statistics of the modelling table only, before splitting, so
    # no fold sees another fold's distribution through a fitted scaler.
    for column in NUMERIC_FIELDS:
        frame[column] = frame[column].fillna(frame[column].median())

    if sample:
        frame = subsample_groups(frame, sample, seed).reset_index(drop=True)

    missing = sorted(set(frame["label"].unique()) - set(kept))
    if missing:
        raise ValueError(f"Labels missing after sampling: {missing}")
    labels = [code for code in kept if code in set(frame["label"])]
    unlabelled = [code for code in labels if code not in ICD_LABELS]
    if unlabelled:
        raise ValueError(
            f"No option text for {unlabelled}: add it to ICD_LABELS in pusk_data.py "
            "(raise --coverage to keep fewer classes, or lower it to keep more)"
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(frame), -1, dtype=int)
    for index, (_, valid) in enumerate(splitter.split(frame, frame["label"], frame["group"])):
        fold_of_row[valid] = index
    if (fold_of_row < 0).any():
        raise RuntimeError("Every row must land in exactly one fold")

    report = {
        "csv": csv_path,
        "rows_in_file": int(len(raw)),
        "unique_codes_in_file": int(counts.size),
        "target_coverage_requested": coverage,
        "target_coverage_reached": round(covered, 4),
        "labels": labels,
        "n_labels": len(labels),
        "rows_modelled": int(len(frame)),
        "groups": int(frame["group"].nunique()),
        "group_field": group_field,
        "n_splits": n_splits,
        "seed": seed,
        "sample_requested": sample,
        "label_counts": {code: int((frame["label"] == code).sum()) for code in labels},
        "dropped_rows": int(len(raw) - len(frame)),
    }
    return Dataset(frame=frame, labels=labels, fold_of_row=fold_of_row, n_splits=n_splits, report=report)


def prompt_for_row(row: pd.Series, language: str = "en") -> str:
    """Clinical note in plain words; the same fields the baseline sees."""
    if language.lower() in ("id", "indonesia", "indonesian"):
        return (
            f"Pasien: jenis kelamin {row['jkl']}, usia {row['age']:.1f} tahun, "
            f"berat badan {row['bb']:.0f} kg, tinggi badan {row['tb']:.0f} cm, "
            f"tensi / tekanan darah {row['sistole']:.0f}/{row['diastole']:.0f} mmHg.\n"
            f"Keluhan utama (anamnesa): {row['anamnesa']}"
        )
    return (
        f"Patient: {row['jkl']}, {row['age']:.1f} years old, "
        f"weight {row['bb']:.0f} kg, height {row['tb']:.0f} cm, "
        f"blood pressure {row['sistole']:.0f}/{row['diastole']:.0f} mmHg.\n"
        f"Chief complaint: {row['anamnesa']}"
    )


# 9 Kelurahan in Puskesmas Catchment Area
CATCHMENT_VILLAGES = {
    "KD01": "Kelurahan 1",
    "KD02": "Kelurahan 2",
    "KD03": "Kelurahan 3",
    "KD04": "Kelurahan 4",
    "KD05": "Kelurahan 5",
    "KD06": "Kelurahan 6",
    "KD07": "Kelurahan 7",
    "KD08": "Kelurahan 8",
    "KD09": "Kelurahan 9",
}

# Short option text per ICD-10 code in English (standard WHO/ICD-10)
ICD_LABELS_EN = {
    "A09": "Infectious gastroenteritis and colitis, unspecified (diarrhoea and vomiting)",
    "A15": "Respiratory tuberculosis",
    "B26": "Mumps",
    "E11": "Type 2 diabetes mellitus",
    "H10": "Conjunctivitis",
    "H81.3": "Other peripheral vertigo (dizziness from the inner ear)",
    "I10": "Essential (primary) hypertension",
    "J00": "Acute nasopharyngitis (common cold)",
    "J02": "Acute pharyngitis (sore throat)",
    "J06": "Acute upper respiratory infection of multiple or unspecified sites",
    "K04.0": "Pulpitis (inflamed dental pulp)",
    "K04.1": "Necrosis of pulp (dead dental pulp)",
    "K29": "Gastritis and duodenitis",
    "K29.7": "Gastritis, unspecified",
    "K30": "Functional dyspepsia (indigestion)",
    "L30": "Other and unspecified dermatitis (skin rash)",
    "M13": "Other arthritis",
    "M79.1": "Myalgia (muscle pain)",
    "R10.4": "Other and unspecified abdominal pain",
    "R11": "Nausea and vomiting",
    "R50": "Fever of other and unknown origin",
    "R51": "Headache",
    "Z00": "General medical examination without complaint or reported diagnosis",
    "Z21": "Asymptomatic human immunodeficiency virus (HIV) infection status",
    "Z27.3": "Need for immunization against other combinations of infectious diseases",
    "Z30.4": "Surveillance of contraceptive drugs",
    "Z34": "Supervision of normal pregnancy",
    "Z35": "Supervision of high-risk pregnancy",
    "Z48": "Other surgical follow-up care (dressing and wound care)",
}

# Clinical option text per ICD-10 code in Indonesian (Bahasa Indonesia)
ICD_LABELS_ID = {
    "A09": "Gastroenteritis dan kolitis infeksius (diare cair, mencret, muntah)",
    "A15": "Tuberkulosis paru / TBC (batuk lama berdahak, batuk darah, flek)",
    "B26": "Gondongan / parotitis (bengkak nyeri kelenjar ludah dan leher)",
    "E11": "Diabetes melitus tipe 2 (kencing manis, kadar gula darah tinggi)",
    "H10": "Konjungtivitis (mata merah, perih, belekan, gatal)",
    "H81.3": "Vertigo perifer (pusing berputar, gangguan keseimbangan telinga)",
    "I10": "Hipertensi esensial primer (tekanan darah tinggi, tensi naik)",
    "J00": "Nasofaringitis akut / selesma (batuk pilek biasa, hidung tersumbat)",
    "J02": "Faringitis akut (radang tenggorokan, nyeri menelan)",
    "J06": "Infeksi saluran pernapasan akut atas / ISPA (batuk, pilek, demam, gregesi)",
    "K04.0": "Pulpitis (sakit gigi berlubang nyut-nyutan, radang pulpa gigi)",
    "K04.1": "Nekrosis pulpa (gangren pulpa, gigi mati berbau, saraf gigi rusak)",
    "K29": "Gastritis dan duodenitis (sakit maag, radang lambung, nyeri lambung)",
    "K29.7": "Gastritis tidak spesifik (perih lambung, mual, sakit ulu hati)",
    "K30": "Dispepsia fungsional (gangguan pencernaan, begah, kembung, perih)",
    "L30": "Dermatitis / eksim (gatal-gatal kulit kemerahan, alergi kulit)",
    "M13": "Artritis lainnya (radang sendi, nyeri persendian kaku)",
    "M79.1": "Mialgia (pegal-pegal linu, nyeri otot seluruh tubuh)",
    "R10.4": "Nyeri perut lainnya dan tidak spesifik (sakit perut melilit)",
    "R11": "Mual dan muntah",
    "R50": "Demam / panas badan tanpa sebab jelas (meriang, gregesi)",
    "R51": "Sakit kepala / pusing cekot-cekot",
    "Z00": "Pemeriksaan kesehatan umum / general check up rutin (tanpa keluhan sakit)",
    "Z21": "Status infeksi HIV tanpa gejala (kontrol rutin VCT / terapi obat ARV)",
    "Z27.3": "Imunisasi kombinasi lanjutan (vaksin DPT / Polio / campak anak)",
    "Z30.4": "Pelayanan dan pengawasan kontrasepsi keluarga berencana (KB suntik, pil, IUD)",
    "Z34": "Pemeriksaan kehamilan normal (kontrol antenatal care / ANC rutin ibu hamil)",
    "Z35": "Pengawasan kehamilan risiko tinggi (KRT, hamil usia risiko atau ada penyulit)",
    "Z48": "Perawatan pasca bedah / tindakan (rawat luka jahitan, ganti perban)",
}

# Default export retains backwards compatibility
ICD_LABELS = ICD_LABELS_EN

def get_icd_labels(language: str = "en") -> dict[str, str]:
    """Return ICD descriptions in either 'en' (English) or 'id' (Indonesian)."""
    if language.lower() in ("id", "indonesia", "indonesian"):
        return ICD_LABELS_ID
    return ICD_LABELS_EN
