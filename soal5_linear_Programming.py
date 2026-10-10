"""
=============================================================================
 UTS — PEMODELAN DAN SIMULASI
 Soal No. 5 : Media Tanam Kultur Jaringan (Laboratorium Bioteknologi)
-----------------------------------------------------------------------------
 Variabel keputusan :
   x1 = liter Media Agar Cokelat  (keuntungan Rp 25.000 / liter)
   x2 = liter Media Agar Hijau    (keuntungan Rp 30.000 / liter)

 Model (Program Linear) :
   Maksimumkan   Z = 25*x1 + 30*x2            (ribu rupiah)
   Batasan       2*x1 +  1*x2 <= 16           (nutrisi organik, gram)
                  1*x1 +  3*x2 <= 18           (hormon pertumbuhan, gram)
                  x1, x2 >= 0  dan bulat       (jumlah produksi disakarikan)

 Strategi verifikasi silang (6 pendekatan independen):
   (1) Enumerasi titik sudut      — metode KKM / titik sudut (analitik)
   (2) Brute force bilangan bulat — pemeriksaan menyeluruh di kisi integer
   (3) Grid halus                 — pencarian kasar kontinu (sanity check)
   (4) SciPy linprog              — LP kontinu via simplex HiGHS
   (5) PuLP (CBC)                 — integer linear program nyata
   (6) Metode garis objektif      — pemeriksaan geometri grafis

 Program ini TIDAK memakai library pemecah LP dalam metode (1); seluruh
 perhitungan pokok dikerjakan sendiri secara eksak dengan aljabar murni,
 sehingga hasilnya dapat ditelusuri kembali dengan tangan di laporan.
=============================================================================
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

NAMA = {
    "x1": "Media Agar Cokelat",
    "x2": "Media Agar Hijau",
}

C = np.array([25.0, 30.0])

A = np.array([
    [2.0, 1.0],
    [1.0, 3.0],
])
B = np.array([16.0, 18.0])

KETERANGAN_BATAS = [
    "Nutrisi organik (maks 16 gram)",
    "Hormon pertumbuhan (maks 18 gram)",
]


def Z(x) -> float:
    """Fungsi tujuan dalam ribu rupiah."""
    return float(C[0] * x[0] + C[1] * x[1])


def ke_rupiah(z_ribu: float) -> str:
    v = int(round(z_ribu * 1000))
    return "Rp " + f"{v:,}".replace(",", ".")


def layak(x, A=A, B=B, tol=1e-9) -> bool:
    """Uji kelayakan: non-negatif + seluruh batasan terpenuhi."""
    x = np.asarray(x, dtype=float)
    if x[0] < -tol or x[1] < -tol:
        return False
    return bool(np.all(A @ x - B <= tol))


def pemakaian(x):
    """Pemakaian tiap sumber daya terhadap batasannya."""
    x = np.asarray(x, dtype=float)
    dipakai = A @ x
    return [
        {
            "sumber_daya": KETERANGAN_BATAS[i],
            "dipakai": float(dipakai[i]),
            "tersedia": float(B[i]),
            "sisa": float(B[i] - dipakai[i]),
        }
        for i in range(len(B))
    ]


def semua_interseksi(A, B):
    """
    Kumpulkan semua kandidat titik batas:
      - irisan sepasang garis batasan (2 persamaan, 2 unknown)
      - irisan tiap garis batasan dengan sumbu koordinat (x1=0 / x2=0)
      - titik asal (0,0)
    """
    titik = []
    n = len(A)
    for i, j in itertools.combinations(range(n), 2):
        M = np.array([A[i], A[j]])
        d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(d) < 1e-12:
            continue
        x1 = (B[i] * M[1, 1] - B[j] * M[0, 1]) / d
        x2 = (M[0, 0] * B[j] - M[1, 0] * B[i]) / d
        titik.append((x1, x2))
    for i in range(n):
        if A[i, 1] != 0:
            titik.append((0.0, B[i] / A[i, 1]))
        if A[i, 0] != 0:
            titik.append((B[i] / A[i, 0], 0.0))
    titik.append((0.0, 0.0))

    uniq = []
    for t in titik:
        if not any(abs(t[0] - u[0]) < 1e-9 and abs(t[1] - u[1]) < 1e-9 for u in uniq):
            uniq.append(t)
    return uniq


def metode_titik_sudut():
    """Semua titik layak beserta nilai Z, diurutkan dari Z terbesar."""
    hasil = []
    for t in semua_interseksi(A, B):
        if layak(t):
            hasil.append((t[0], t[1], Z(t)))
    return sorted(hasil, key=lambda h: -h[2])


def urut_sudut(daftar):
    """Urutkan titik sudut searah sudut terhadap pusat poligon (untuk grafik)."""
    cx = sum(d[0] for d in daftar) / len(daftar)
    cy = sum(d[1] for d in daftar) / len(daftar)
    return sorted(daftar, key=lambda d: np.arctan2(d[1] - cy, d[0] - cx))


def brute_force_bulat():
    daftar = []
    for x1 in range(0, 9):
        for x2 in range(0, 7):
            if layak((x1, x2)):
                daftar.append((x1, x2, Z((x1, x2))))
    best = max(daftar, key=lambda d: d[2])
    n_opt = sum(1 for d in daftar if abs(d[2] - best[2]) < 1e-9)
    return best, len(daftar), n_opt, daftar


def grid_cari(n=4001):
    xs = np.linspace(0, 16, n)
    ys = np.linspace(0, 18, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ok = (2 * X + Y <= 16 + 1e-9) & (X + 3 * Y <= 18 + 1e-9)
    zz = 25 * X + 30 * Y
    zz[~ok] = -np.inf
    i = np.unravel_index(np.argmax(zz), zz.shape)
    return float(xs[i[0]]), float(ys[i[1]]), float(zz[i])


def dengan_scipy():
    try:
        from scipy.optimize import linprog
    except Exception as e:
        return {"tersedia": False, "alasan": str(e)}
    r = linprog([-25, -30], A_ub=A, b_ub=B, bounds=[(0, None), (0, None)],
                method="highs")
    return {
        "tersedia": bool(r.success),
        "pesan": str(r.message),
        "x1": round(float(r.x[0]), 6),
        "x2": round(float(r.x[1]), 6),
        "Z_ribu": round(float(-r.fun), 6),
        "slack_variabel": [round(float(v), 6) for v in r.ineqlin.residual],
        "harga_bayangan": [round(float(-m), 6) for m in r.ineqlin.marginals],
    }


def dengan_pulp(integral=True):
    try:
        import pulp
    except Exception as e:
        return {"tersedia": False, "alasan": str(e)}
    cat = "Integer" if integral else "Continuous"
    prob = pulp.LpProblem("UTS_Soal5", pulp.LpMaximize)
    x1 = pulp.LpVariable("x_Agar_Cokelat", lowBound=0, cat=cat)
    x2 = pulp.LpVariable("x_Agar_Hijau", lowBound=0, cat=cat)
    prob += 25 * x1 + 30 * x2, "Keuntungan"
    prob += 2 * x1 + 1 * x2 <= 16, "Batas_Nutrisi_Organik"
    prob += 1 * x1 + 3 * x2 <= 18, "Batas_Hormon"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {
        "tersedia": True,
        "tipe": cat,
        "status": pulp.LpStatus[prob.status],
        "x1": int(pulp.value(x1)) if integral else float(pulp.value(x1)),
        "x2": int(pulp.value(x2)) if integral else float(pulp.value(x2)),
        "Z_ribu": int(pulp.value(prob.objective)) if integral else float(pulp.value(prob.objective)),
        "slack_batasan": {
            "nutrisi_organik": round(float(prob.constraints["Batas_Nutrisi_Organik"].slack), 6),
            "hormon": round(float(prob.constraints["Batas_Hormon"].slack), 6),
        },
    }


def garis_objektif(titik_sudut):
    """
    Gradien garis objektif: -c1/c2 = -25/30 = -5/6.
    Gradien batas nutrisi  : -2/1 = -2
    Gradien batas hormon   : -1/3
    Karena -2 < -5/6 < -1/3, garis objektif lebih landai daripada kedua
    garis batas -> pergeseran garis objektif ke arah utara-timur pertama
    kali menyentuh poligon pada irisan kedua garis batas.
    """
    gr_obj = -C[0] / C[1]
    gr_batas = [-A[i, 0] / A[i, 1] for i in range(len(A))]
    urutan = sorted(list(gr_batas) + [gr_obj])
    posisi = urutan.index(gr_obj)
    return {
        "gradien_objektif": gr_obj,
        "gradien_batas": dict(zip(KETERANGAN_BATAS, gr_batas)),
        "urutan_kemiringan": urutan,
        "indeks_objektif": posisi,
        "interpretasi": (
            "Gradien objektif berada DI ANTARA gradien kedua garis batas, "
            "sehingga garis objektif menyentuh daerah layak terakhir pada "
            "titik potong kedua garis batas (6, 4)."
        ),
    }


def _opt_lp(A_, B_):
    """Optimum LP kontinu (bukan integer) untuk data A_, B_ tertentu."""
    best = None
    for t in semua_interseksi(np.asarray(A_, float), np.asarray(B_, float)):
        if layak(t, np.asarray(A_, float), np.asarray(B_, float)):
            v = Z(t)
            if best is None or v > best[2] + 1e-9:
                best = (t[0], t[1], v)
    return best


def analisis_sensitivitas():
    dasar = _opt_lp(A, B)
    out = []
    for k in range(len(B)):
        BP = B.copy(); BP[k] += 1
        BM = B.copy(); BM[k] -= 1
        z_naik = _opt_lp(A, BP)[2]
        z_turun = _opt_lp(A, BM)[2]
        out.append({
            "batasan": KETERANGAN_BATAS[k],
            "RHS": float(B[k]),
            "Z_dasar": dasar[2],
            "Z_naik_1": z_naik,
            "Z_turun_1": z_turun,
            "harga_bayangan": round(z_naik - dasar[2], 6),
            "harga_bayangan_negatif": round(z_turun - dasar[2], 6),
        })
    return out, dasar


def rentang_keabadian_basis(i, j, b_i=None, b_j=None):
    """
    Rentang RHS batasan `i` agar titik potong garis batasan (i) dan (j)
    tetap NON-NEGATIF (syarat dasar tetap sah sebagai solusi optium).
    Hanya RHS batasan i yang diubah; RHS batasan j tetap.

      M = [a_i ; a_j]        M.[x1;x2] = [b_i'; b_j]
      x1 = (b_i'*a_j2 - b_j*a_i2) / det
      x2 = (a_i1*b_j - a_j1*b_i') / det

    Syarat: x1 >= 0  dan  x2 >= 0  ->  interval [lo, hi].
    """
    a_i, a_j = A[i], A[j]
    det = a_i[0] * a_j[1] - a_i[1] * a_j[0]
    bj = B[j] if b_j is None else float(b_j)
    b0 = B[i] if b_i is None else float(b_i)

    def titik(v):
        return ((v * a_j[1] - bj * a_i[1]) / det,
                (a_i[0] * bj - a_j[0] * v) / det)

    lo, hi = None, None
    for v in np.linspace(b0 - 80, b0 + 80, 32001):
        x1, x2 = titik(v)
        if x1 >= -1e-9 and x2 >= -1e-9:
            lo = v if lo is None else lo
            hi = v
    kandidat = []
    for v in np.linspace(b0 - 80, b0 + 80, 32001):
        x1, x2 = titik(v)
        if abs(x1) < 1e-12 or abs(x2) < 1e-12:
            kandidat.append(v)
    return {
        "RHS_sekarang": b0,
        "rentang_keabadian_dasar": [round(float(lo), 6), round(float(hi), 6)],
        "titik_potong_di_batas": [tuple(round(c, 6) for c in titik(v)) for v in kandidat],
        "solusi_optimum": tuple(round(c, 6) for c in titik(b0)),
    }


def grafik(titik_sudut, path_png, tanda_tangan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    poligon = urut_sudut(titik_sudut)
    koordinat = [(p[0], p[1]) for p in poligon]

    ax.add_patch(Polygon(koordinat, closed=True, facecolor="#74c0fc", alpha=0.40,
                         edgecolor="#1864ab", lw=2.2, zorder=1,
                         label="Daerah layak (feasible region)"))

    xs = np.linspace(0, 18, 500)
    ax.plot(xs, (16 - 2 * xs) / 1, color="#e03131", lw=2, ls="--",
            label=r"$2x_1+x_2=16$  (nutrisi organik)")
    ax.plot(xs, (18 - xs) / 3, color="#2f9e44", lw=2, ls="--",
            label=r"$x_1+3x_2=18$  (hormon pertumbuhan)")

    z_opt = max(t[2] for t in titik_sudut)
    x_opt, y_opt, _ = max(titik_sudut, key=lambda t: t[2])
    ax.plot(xs, (z_opt - 25 * xs) / 30, color="#9c36b5", lw=2.4, ls=":",
            label=rf"$25x_1+30x_2={z_opt:.0f}$ (garis objektif optimal)")

    for (a, b, val) in poligon:
        ax.plot(a, b, "o", ms=9, color="#111827", zorder=5)
        dx, dy = (0.3, 0.45)
        if a > 7.5:   dx = -1.75
        if b > 15:    dy = -1.15
        ax.annotate(f"({a:g}, {b:g})\nZ = {val:.0f}", (a, b),
                    textcoords="offset points", xytext=(dx * 10, dy * 10),
                    fontsize=9.2, fontweight="bold", color="#111827")

    ax.plot(x_opt, y_opt, "*", ms=23, color="#ffb703", mec="#7d3c00", zorder=6,
            label=rf"Optimum ({x_opt:g}, {y_opt:g})  $Z$ = {ke_rupiah(z_opt)}")

    ax.set_xlabel(r"$x_1$  —  liter Media Agar Cokelat", fontsize=11.5)
    ax.set_ylabel(r"$x_2$  —  liter Media Agar Hijau", fontsize=11.5)
    ax.set_title("Soal No. 5 — Daerah Layak, Titik Sudut, dan Garis Objektif",
                 fontsize=13.5, fontweight="bold")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.set_xticks(range(0, 19, 2))
    ax.set_yticks(range(0, 19, 2))
    ax.grid(True, ls=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.96)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return path_png


def main():
    out = Path(__file__).resolve().parent
    hasil = {}

    ts = metode_titik_sudut()
    hasil["1_titik_sudut"] = [{"x1": a, "x2": b, "Z_ribu": z} for a, b, z in ts]
    opt = ts[0]

    best, n_layak, n_opt, _ = brute_force_bulat()
    hasil["2_brute_force_bulat"] = {
        "jumlah_solusi_bulat_layak": n_layak,
        "jumlah_solusi_optimum": n_opt,
        "optimum": {"x1": best[0], "x2": best[1], "Z_ribu": best[2]},
    }

    g = grid_cari()
    step_x1 = 16.0 / 4000
    step_x2 = 18.0 / 4000
    tol_grid = abs(C[0]) * step_x1 + abs(C[1]) * step_x2
    hasil["3_grid_halus"] = {"x1": round(g[0], 4), "x2": round(g[1], 4), "Z_ribu": round(g[2], 4),
                             "step_x1": step_x1, "step_x2": step_x2,
                             "toleransi_Z": round(tol_grid, 6),
                             "selisih_Z": round(opt[2] - g[2], 6)}

    hasil["4_scipy_linprog"] = dengan_scipy()

    hasil["5_pulp_integer"] = dengan_pulp(integral=True)
    hasil["5b_pulp_relaksasi"] = dengan_pulp(integral=False)

    hasil["6_garis_objektif"] = garis_objektif(ts)

    sens, _ = analisis_sensitivitas()
    hasil["7_analisis_sensitivitas"] = sens
    hasil["7b_rentang_keabadian_basis"] = {
        "nutrisi_organik_16": rentang_keabadian_basis(0, 1),
        "hormon_18": rentang_keabadian_basis(1, 0),
    }

    hasil["pemakaian_sumber_daya"] = pemakaian(opt[:2])

    hasil["grafik"] = str(grafik(ts, out / "daerah_layak_no5.png", None))

    hasil["KESIMPULAN"] = {
        "x1_Agar_Cokelat": opt[0],
        "x2_Agar_Hijau": opt[1],
        "Z_ribu": opt[2],
        "Z_rupiah": ke_rupiah(opt[2]),
        "pemakaian_nutrisi": 2 * opt[0] + opt[1],
        "pemakaian_hormon": opt[0] + 3 * opt[1],
        "sisa_nutrisi": 16 - (2 * opt[0] + opt[1]),
        "sisa_hormon": 18 - (opt[0] + 3 * opt[1]),
        "keduanya_batasan_ketat": True,
    }

    cek = []
    cek.append(("titik_sudut == brute_force", abs(opt[2] - best[2]) < 1e-9))
    cek.append(("titik_sudut == grid", abs(opt[2] - g[2]) <= tol_grid + 1e-9))
    if hasil["4_scipy_linprog"]["tersedia"]:
        cek.append(("titik_sudut == scipy", abs(opt[2] - hasil["4_scipy_linprog"]["Z_ribu"]) < 1e-6))
    if hasil["5_pulp_integer"]["tersedia"]:
        cek.append(("titik_sudut == pulp ILP", opt[2] == hasil["5_pulp_integer"]["Z_ribu"]))
    if hasil["5b_pulp_relaksasi"]["tersedia"]:
        cek.append(("ILP == relaksasi kontinu (tanpa pecahan)", abs(opt[2] - hasil["5b_pulp_relaksasi"]["Z_ribu"]) < 1e-6))
    cek.append(("solusi bulat unik", n_opt == 1))
    cek.append(("kedua batasan tepat terpakai",
                abs(2 * opt[0] + opt[1] - 16) < 1e-9 and abs(opt[0] + 3 * opt[1] - 18) < 1e-9))

    rb = hasil["7b_rentang_keabadian_basis"]
    for nama, info in rb.items():
        lo, hi = info["rentang_keabadian_dasar"]
        cek.append((f"solusi dalam rentang basis [{nama}]", lo - 1e-9 <= info["RHS_sekarang"] <= hi + 1e-9))

    cek.append(("harga bayangan positif",
                all(s["harga_bayangan"] > 0 for s in hasil["7_analisis_sensitivitas"])))

    if hasil["4_scipy_linprog"]["tersedia"] and hasil["4_scipy_linprog"].get("harga_bayangan"):
        for k, hb in enumerate(hasil["4_scipy_linprog"]["harga_bayangan"]):
            cek.append((f"harga bayangan scipy == enumerasi [{KETERANGAN_BATAS[k]}]",
                        abs(hb - hasil["7_analisis_sensitivitas"][k]["harga_bayangan"]) < 1e-6))
    hasil["PEMERIKSAAN_KONSEISTENSI"] = [{"tes": t, "lolos": bool(v)} for t, v in cek]

    teks = json.dumps(hasil, indent=2, ensure_ascii=False)
    print(teks)
    (out / "hasil_verifikasi_no5.json").write_text(teks, encoding="utf-8")

    gagal = [t for t, v in cek if not v]
    if gagal:
        raise SystemExit("GAGAL: " + ", ".join(gagal))
    print("\n==> SEMUA PEMERIKSAAN KONSEISTEN  |  Optimum: "
          f"x1={opt[0]:g}, x2={opt[1]:g}, Z={ke_rupiah(opt[2])}")


if __name__ == "__main__":
    main()
