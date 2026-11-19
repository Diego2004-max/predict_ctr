
from pathlib import Path
import random

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, list_repo_files

REPOSITORIO = "criteo/CriteoClickLogs"
SEMILLA = 42

FECHAS = {
    "train": [
        "2015-02-15",
        "2015-02-16",
        "2015-02-17",
        "2015-02-18",
        "2015-02-19",
    ],
    "validation": ["2015-02-20"],
    "test": ["2015-02-21"],
}

FILAS_POR_DIA = 20_000
ARCHIVOS_POR_DIA = 5
LOTE = 10_000

SALIDA = Path("data/processed")
SALIDA.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEMILLA)
fs = HfFileSystem()

print("Consultando archivos disponibles...")

archivos = list_repo_files(
    repo_id=REPOSITORIO,
    repo_type="dataset"
)

resumen = []

for particion, fechas in FECHAS.items():
    for fecha in fechas:
        print(f"\nProcesando {fecha} ({particion})")

        candidatos = sorted(
            archivo for archivo in archivos
            if archivo.startswith(f"data/day={fecha}/")
            and archivo.endswith(".parquet")
        )

        if len(candidatos) < ARCHIVOS_POR_DIA:
            raise RuntimeError(
                f"No hay suficientes archivos para {fecha}"
            )

        seleccionados = rng.sample(
            candidatos, ARCHIVOS_POR_DIA
        )

        muestras = []
        total = 0

        for archivo in seleccionados:
            if total >= FILAS_POR_DIA:
                break

            ruta_remota = f"datasets/{REPOSITORIO}/{archivo}"

            with fs.open(ruta_remota, "rb") as remoto:
                parquet = pq.ParquetFile(remoto)

                for lote in parquet.iter_batches(batch_size=LOTE):
                    df_lote = lote.to_pandas()

                    faltan = FILAS_POR_DIA - total
                    n = min(len(df_lote), faltan)

                    # Selección reproducible dentro del lote
                    muestra = df_lote.sample(
                        n=n,
                        random_state=SEMILLA + total
                    )

                    muestras.append(muestra)
                    total += len(muestra)

                    if total >= FILAS_POR_DIA:
                        break

        if not muestras:
            raise RuntimeError(f"No se obtuvieron datos de {fecha}")

        df_dia = pd.concat(muestras, ignore_index=True)
        df_dia["fecha"] = fecha

        ruta_salida = SALIDA / f"{particion}_{fecha}.parquet"
        df_dia.to_parquet(ruta_salida, index=False)

        clics = int(df_dia["label"].sum())
        ctr = df_dia["label"].mean() * 100

        resumen.append({
            "particion": particion,
            "fecha": fecha,
            "filas": len(df_dia),
            "clics": clics,
            "ctr_porcentaje": round(ctr, 2),
        })

        print(f"Guardado: {ruta_salida}")
        print(f"Filas: {len(df_dia):,} | CTR: {ctr:.2f}%")

df_resumen = pd.DataFrame(resumen)
df_resumen.to_csv(
    SALIDA / "resumen_muestra.csv",
    index=False
)

print("\n===== RESUMEN FINAL =====")
print(df_resumen.to_string(index=False))
print("\nPreparación finalizada.")
