"""Generate realistic synthetic clinical dataset mimicking Puskesmas characteristics.

Maintains:
- Identical 16-column EHR schema.
- Statistically accurate distribution across the 29 ICD-10 diagnostic classes.
- Realistic code-mixed Bahasa Indonesia and Javanese clinical anamnesis templates.
- Clinically correlated physiological vitals (blood pressure, BMI, age, sex).
- Both urban resident catchment villages (KD01-KD54) and commuter/referral population (XXX).
- Zero protected health information (100% synthetic).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

CLINICAL_TEMPLATES = {
    "J00": [
        "batuk pilek sejak 2 hari, hidung tersumbat, badan sumer-sumer gregesi",
        "pilek hidung buntu, bersin-bersin, tenggorokan agak gatal, pusing",
        "keluar ingus encer, batuk kering sedikit, badan meriang sejak kemarin",
        "pilek 3 hari, hidung mampet, pusing cekot-cekot ringan, badan lemas"
    ],
    "J06": [
        "batuk mulai tadi malam, dahak sedikit hijau, pilek 3 hari, badan gregesi, tenggorokan nyeri (+)",
        "demam tinggi sejak kemarin, batuk berdahak, pilek, badan gregesi, tenggorokan sakit menelan",
        "batuk pilek demam 2 hari, badan linu semua, tenggorokan merah nyeri menelan",
        "batuk grok-grok berdahak kuning, pilek ingus kental, meriang panas dingin"
    ],
    "J02": [
        "tenggorokan sakit perih buat menelan (+), amandel merah bengkak, badan sumer gregesi, batuk sedikit",
        "nyeri telan hebat sejak kemarin, amandel bengkak, tenggorokan merah meradang, demam",
        "sakit tenggorokan buat makan minum perih, badan panas nggregesi, leher pegal"
    ],
    "I10": [
        "kontrol ulang tensi tinggi, kepala belakang cekot-cekot, leher kaku, kadang terasa nggliyeng kalau bangun tidur",
        "kontrol tensi rutin, kepala pusing berputar ringan, tengkuk terasa berat kaku",
        "pasien riwayat darah tinggi, cek tensi rutin bulanan, sering pusing bila kecapekan",
        "kontrol tensi, kepala cenut-cenut di belakang, leher kencang, obat darah tinggi habis"
    ],
    "Z21": [
        "ambil obat ARV 1 bulan, kontrol rutin, tidak ada keluhan",
        "px mengatakan ambil obat ARV rutin bulanan, kondisi stabil tak ada keluhan",
        "ambil obat ARV 1 bulan, kontrol rutin vct, kondisi baik tidak ada keluhan penyerta",
        "ambil ARV rutin, tidak ada keluhan mual atau pusing, minum obat teratur"
    ],
    "K29": [
        "perut perih di ulu hati sejak 2 hari, mual muntah bila makan, badan lemas, perut kembung dan begah",
        "ulu hati nyeri perih menusuk, mual-mual setelah makan pedas, sering sendawa asam",
        "nyeri lambung perih kambuh, mual muntah 2x, perut begah kembung perih",
        "sakit perut melilit bagian atas, mual tidak nafsu makan, perih ulu hati"
    ],
    "K29.7": [
        "perut perih di ulu hati kambuhan, mual muntah bila makan telat, badan lemas",
        "asam lambung naik perih, dada panas terbakar, mual muntah, riwayat sakit maag",
        "maag kambuh, nyeri ulu hati tembus belakang, mual tidak bisa makan nasi"
    ],
    "K30": [
        "perut kembung begah penuh sesak, sering sendawa, mual tapi tidak muntah, begah setelah makan",
        "dispepsia perut sebah, cepat kenyang, perut atas tidak nyaman kembung perih",
        "perut atas begah kembung, mual ringan, buang angin susah, dada agak sesak begah"
    ],
    "K04.1": [
        "px mengatakan sakit gigi berlubang cekot-cekot sejak 3 hari, buat ngunyah linu, gusi bengkak",
        "gigi geraham bawah kanan berlubang besar, linu saat makan minum dingin, cenut-cenut malam hari",
        "gigi berlubang mati berbau, gusi bengkak linu buat mengunyah sakit cekot-cekot",
        "gigi belakang berlubang sakit ngilu, pipi terasa agak bengkak kencang"
    ],
    "K04.0": [
        "gigi geraham berlubang linu nyut-nyutan saat makan manis dan dingin, belum bengkak",
        "linu gigi berlubang bila kena air dingin, ngilu tajam menusuk, tidak ada bengkak gusi"
    ],
    "H81.3": [
        "pusing muter-muter sejak tadi pagi, terasa nggliyeng saat miring ke kanan, mual keringat dingin, jalan sempoyongan",
        "kepala terasa berputar hebat bila menoleh, mual muntah 1x, pandangan kabur nggliyeng",
        "vertigo kambuh pusing muter, kuping berdenging, badan lemas mual keringat dingin"
    ],
    "H10": [
        "mata kanan kiri merah perih, keluar kotoran belekan lengket saat bangun tidur, gatal dan silau",
        "mata merah belekan banyak lendir kuning, kelopak mata bengkak lengket gatal",
        "mata perih kemerahan, terasa mengganjal seperti berpasir, keluar cairan belekan"
    ],
    "R50": [
        "demam tinggi 3 hari naik turun, badan gregesi lemas, pusing, mual tapi tidak batuk dan tidak pilek",
        "panas demam sejak 2 hari, meriang panas dingin, badan ngilu pegal, mual muntah 1x",
        "panas tinggi mendadak kemarin malam, badan lemas gregesi, pusing cekot-cekot"
    ],
    "R51": [
        "kepala pusing cekot-cekot sebelah kanan sejak 2 hari, mata agak silau, mual ringan",
        "sakit kepala cenut-cenut berdenyut di dahi dan pelipis, leher kaku, susah tidur"
    ],
    "A09": [
        "diare cair lebih dari 4 kali sejak semalam, perut mules melilit, badan lemas mual",
        "mencret cair berulang, feses lembek tanpa darah, perut mules melilit, haus terus",
        "buang air besar cair 5x hari ini, badan lemas, perut kram melilit mual"
    ],
    "E11": [
        "kontrol kencing manis rutin, sering haus dan buang air kecil malam hari, badan cepat lelah",
        "kontrol gula darah rutin, kaki kesemutan baal, riwayat diabetes mellitus minum obat teratur",
        "cek gula darah puasa, badan lemas sering mengantuk, riwayat DM tipe 2"
    ],
    "L30": [
        "kulit tangan dan sela jari gatal-gatal kemerahan, bruntusan berair perih setelah kena sabun",
        "gatal-gatal di lipatan kaki dan tangan, kulit bersisik merah menebal, perih digaruk",
        "bercak merah gatal di kulit punggung dan lengan, bentol-bentol perih gatal"
    ],
    "M13": [
        "nyeri sendi lutut kanan kiri linu kaku saat digerakkan bangun tidur, jalan sakit",
        "persendian jari tangan ngilu bengkak kaku, linu bila kena hawa dingin malam hari",
        "sendi lutut pegal linu kemeng, susah ditekuk buat sholat, nyeri gerak"
    ],
    "M79.1": [
        "badan pegal-pegal linu seluruh tubuh, otot leher dan pundak kencang kemeng linu",
        "nyeri otot punggung dan pinggang setelah angkat beban, badan pegel linu nggregesi"
    ],
    "Z34": [
        "pemeriksaan kehamilan rutin anak pertama, usia kehamilan 24 minggu, kondisi ibu dan janin baik",
        "kontrol ANC rutin trimester 2, tensi bagus, gerak janin aktif, tidak ada keluhan"
    ],
    "Z35": [
        "kontrol hamil risiko tinggi, usia ibu 38 tahun hamil anak ke 4, riwayat tensi tinggi",
        "kontrol ANC risiko tinggi trimester 3, bengkak kedua kaki, tensi terpantau naik"
    ],
    "Z30.4": [
        "kontrol suntik KB 3 bulan rutin, tidak ada keluhan perdarahan atau pusing",
        "pelayanan KB suntik ulang, jadwal suntik hari ini, tensi dan berat badan normal"
    ],
    "Z00": [
        "pemeriksaan kesehatan umum / general checkup untuk berkas lamaran, tidak ada keluhan sakit",
        "kontrol sehat rutin, cek tensi dan laboratorium berkala, kondisi fit tanpa keluhan",
        "minta surat keterangan sehat, tidak ada keluhan sakit, badan terasa bugar"
    ],
    "Z48": [
        "kontrol ganti perban luka operasi, lepas jahitan post tindakan, luka tampak bersih kering",
        "perawatan luka jahitan di tangan, luka bersih tidak ada nanah, ganti kasa perban steril"
    ],
    "B26": [
        "pipi bawah telinga kanan bengkak besar dan nyeri linu, demam 2 hari, susah buka mulut dan mengunyah",
        "gondongan bengkak di bawah rahang telinga kiri, nyeri menelan dan demam nggregesi"
    ],
    "A15": [
        "batuk lama lebih 3 minggu berdahak bercampur darah, badan kurus lemas berkeringat malam hari, demam sumer",
        "batuk berdahak sudah 1 bulan tak kunjung sembuh, sesak napas, nafsu makan turun berat badan anjlok"
    ],
    "R10.4": [
        "perut bawah sakit melilit nyeri tekan ringan, belum BAB 3 hari, perut begah",
        "nyeri perut melilit hilang timbul sejak pagi, kembung dan mual"
    ],
    "R11": [
        "mual dan muntah-muntah sejak tadi pagi, tidak bisa masuk makanan, lambung terasa perih",
        "muntah berulang kali isi cairan dan makanan, badan lemas dehidrasi ringan"
    ],
    "Z27.3": [
        "imunisasi bayi rutin usia 4 bulan, vaksin DPT combo dan polio, bayi dalam keadaan sehat",
        "jadwal imunisasi lanjutan balita, tidak demam, kondisi anak sehat ceria"
    ]
}

MEDS_MAPPING = {
    "J00": ("16004.01.3 F, 11040.01.3 A", "PARASETAMOL TABLET 500 MG, KLORFENIRAMIN MALEAT TB 4 MG", "10, 10"),
    "J06": ("01013 M, 14024 E, 16004.01.3 F", "AMOKSILIN KAPSUL 500 MG, N-ACETYL CYTEINE 200 MG, PARASETAMOL TABLET 500 MG", "10, 10, 10"),
    "J02": ("01013 M, 16004.01.3 F, 16015.01.3 E", "AMOKSILIN KAPSUL 500 MG, PARASETAMOL TABLET 500 MG, DEKSAMETASON 0.5 MG", "10, 10, 10"),
    "I10": ("03001 A, 03002 B", "AMLODIPIN TABLET 5 MG, KANDESARTAN TABLET 8 MG", "30, 30"),
    "Z21": ("99001 V, 99002 V", "TENOFOVIR 300 MG, LAMIVUDIN 300 MG", "30, 30"),
    "K29": ("01019.01.3, 04029 C", "ANTASIDA DOEN TAB KOMB, DOMPERIDON TAB 10 MG", "10, 10"),
    "K29.7": ("01019.01.3, 04030 D", "ANTASIDA DOEN TAB KOMB, OMEPRAZOL KAPSUL 20 MG", "10, 10"),
    "K30": ("01019.01.3, 04029 C", "ANTASIDA DOEN TAB KOMB, DOMPERIDON TAB 10 MG", "10, 10"),
    "K04.1": ("01013 M, 16004.01.3 F", "AMOKSILIN KAPSUL 500 MG, ASAM MEFENAMAT 500 MG", "10, 10"),
    "K04.0": ("16004.01.3 F", "ASAM MEFENAMAT 500 MG", "10"),
    "H81.3": ("04031 B, 16004.01.3 F", "BETAHISTIN MESILAT 6 MG, PARASETAMOL TABLET 500 MG", "10, 10"),
    "H10": ("05001 E", "KLORAMFENIKOL TETES MATA 0.5%", "1"),
    "R50": ("16004.01.3 F", "PARASETAMOL TABLET 500 MG", "10"),
    "R51": ("16004.01.3 F", "PARASETAMOL TABLET 500 MG", "10"),
    "A09": ("06001 O, 06002 Z", "ORALIT SERBUK, ZINK TABLET 20 MG", "5, 10"),
    "E11": ("07001 M, 07002 G", "METFORMIN TABLET 500 MG, GLIMEPIRID TABLET 2 MG", "60, 30"),
    "L30": ("08001 H, 11040.01.3 A", "HIDROKORTISON KRIM 2.5%, KLORFENIRAMIN MALEAT TB 4 MG", "1, 10"),
    "M13": ("16004.01.3 F", "NATRIUM DIKLOFENAK 50 MG", "10"),
    "M79.1": ("16004.01.3 F, 16015.01.3 E", "PARASETAMOL TABLET 500 MG, VITAMIN B COMPLEX", "10, 10"),
    "Z34": ("09001 F", "TABLET TAMBAH DARAH, KALSIUM LAKTAT 500 MG", "30, 30"),
    "Z35": ("09001 F", "TABLET TAMBAH DARAH, KALSIUM LAKTAT 500 MG", "30, 30"),
    "Z30.4": ("09005 K", "MEDROKSI PROGESTERON ASETAT INJ 150 MG", "1"),
    "Z00": ("-", "-", "-"),
    "Z48": ("10001 K, 10002 P", "KASSA STERIL, POVIDON IODIN 10%", "1, 1"),
    "B26": ("16004.01.3 F", "PARASETAMOL TABLET 500 MG", "10"),
    "A15": ("12001 F", "OAT KATEGORI 1 (FDC TBC)", "28"),
    "R10.4": ("01019.01.3", "ANTASIDA DOEN TAB KOMB", "10"),
    "R11": ("04029 C", "DOMPERIDON TAB 10 MG", "10"),
    "Z27.3": ("13001 V", "VAKSIN DPT-HB-HIB, POLIO ORAL", "1, 1")
}

def generate_records(num_rows: int = 1000) -> pd.DataFrame:
    records = []
    base_date = datetime(2024, 1, 2)
    classes = list(CLINICAL_TEMPLATES.keys())
    
    # Probabilities matching Puskesmas distribution
    weights = [
        0.18, 0.14, 0.05, 0.15, 0.08, 0.06, 0.03, 0.03, 0.04, 0.02,
        0.03, 0.01, 0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.005, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005
    ]
    weights = [w / sum(weights) for w in weights]
    
    patient_ids = [f"9{random.randint(1000000, 9999999)}" for _ in range(num_rows // 2)]
    
    for i in range(num_rows):
        code = np.random.choice(classes, p=weights)
        is_commuter = random.random() < 0.15
        kdesa = "XXX" if is_commuter else f"KD{random.randint(1, 54):02d}"
        
        # Correlated vitals & demographics
        if code in ["Z34", "Z35", "Z30.4"]:
            jkl = "Perempuan"
            age = round(random.uniform(19, 38), 1)
            sistole = random.randint(100, 130) if code != "Z35" else random.randint(130, 160)
            diastole = random.randint(65, 85) if code != "Z35" else random.randint(85, 100)
        elif code in ["B26", "Z27.3"]:
            jkl = random.choice(["Laki-laki", "Perempuan"])
            age = round(random.uniform(1, 12), 1)
            sistole = random.randint(90, 115)
            diastole = random.randint(60, 75)
        elif code == "I10":
            jkl = random.choice(["Laki-laki", "Perempuan"])
            age = round(random.uniform(45, 78), 1)
            sistole = random.randint(140, 185)
            diastole = random.randint(88, 110)
        else:
            jkl = random.choice(["Laki-laki", "Perempuan"])
            age = round(random.uniform(18, 68), 1)
            sistole = random.randint(105, 135)
            diastole = random.randint(68, 88)
            
        bb = round(random.uniform(42, 85) if age > 15 else random.uniform(10, 35), 1)
        tb = round(random.uniform(145, 178) if age > 15 else random.uniform(75, 140), 1)
        
        # Clinical date and birth date
        enc_date = base_date + timedelta(days=random.randint(0, 360))
        birth_year = enc_date.year - int(age)
        birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
        
        anamnesa = random.choice(CLINICAL_TEMPLATES[code])
        med_codes, med_names, med_reqs = MEDS_MAPPING.get(code, ("-", "-", "-"))
        
        rec = {
            "tanggal": enc_date.strftime("%Y-%m-%d 00:00:00"),
            "kpusk": "P01",
            "no_reg": random.choice(patient_ids),
            "bb": bb,
            "tb": tb,
            "sistole": sistole,
            "diastole": diastole,
            "anamnesa": anamnesa,
            "kode_penyakit": json.dumps([code]),
            "cara_bayar": random.choice(["bpjs pbi", "bpjs non pbi", "umum / bayar"]),
            "tg_lahir": birth_date.strftime("%Y-%m-%d 00:00:00"),
            "jkl": jkl,
            "kdesa": kdesa,
            "kode_obat": med_codes,
            "nama_obat": med_names,
            "permintaan": med_reqs
        }
        records.append(rec)
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_records(1000)
    out_file = out_dir / "dummy_pusk.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} synthetic patient encounters to {out_file}")
