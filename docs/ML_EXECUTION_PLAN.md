# Инженерный план исполнения ML/DL-подсистемы магистерской диссертации

**Тема диссертации:** *«Recognition of ionospheric and geomagnetic events using wavelet analysis and deep learning methods»*  
**Целевой документ:** `docs/ML_EXECUTION_PLAN.md` (рабочая спецификация для поэтапной реализации последующими ИИ-агентами)  
**Статус:** Утверждённый инженерный план исполнения (код обучения на данном этапе не пишется)  
**Базис верификации:** Прямой аудит репозитория `c:\Programming\Projects\ONU_Thesis`, БД `data/processed/catalog.sqlite`, всех 7 Parquet-файлов `data/processed/`, модулей `backend/geomag_core/` и `backend/geomag_etl/`, а также локального железа.

---

## Содержание

1. [Верифицированное состояние репозитория и данных (Ground Truth)](#1-верифицированное-состояние-репозитория-и-данных-ground-truth)
2. [Жёсткие ограничения среды, изоляции, железа и переносимости](#2-жёсткие-ограничения-среды-изоляции-железа-и-переносимости)
3. [Архитектурная концепция: две ветки (A и B) на общем фундаменте](#3-архитектурная-концепция-две-ветки-a-и-b-на-общем-фундаменте)
4. [Полная структура каталога `ml_research/`](#4-полная-структура-каталога-ml_research)
5. [Контракты данных, тензоров, конфигов и артефактов](#5-контракты-данных-тензоров-конфигов-и-артефактов)
6. [Спецификации моделей, функций потерь и стратегий обучения (Фазы 1–4)](#6-спецификации-моделей-функций-потерь-и-стратегий-обучения-фазы-14)
7. [Пошаговый план атомарных задач (`T0.1` – `T6.3`)](#7-пошаговый-план-атомарных-задач-t01--t63)
8. [Схема запуска: локально (CPU) vs Google Colab / Kaggle (GPU T4)](#8-схема-запуска-локально-cpu-vs-google-colab--kaggle-gpu-t4)
9. [Матрица рисков и инженерные меры защиты в коде](#9-матрица-рисков-и-инженерные-меры-защиты-в-коде)

---

## 1. Верифицированное состояние репозитория и данных (Ground Truth)

Все числа, схемы, имена столбцов и интерфейсы ниже проверены напрямую по файлам репозитория `c:\Programming\Projects\ONU_Thesis`. Любые последующие агенты **обязаны** опираться именно на эти контракты.

### 1.1. Нормализованный контур `data/processed/` (строго `read-only` для ML)

| Файл | Строк / размер | Реальная Arrow/SQLite схема | Критические факты |
|---|---:|---|---|
| `catalog.sqlite` | 5 таблиц (`stations`: 50, `episodes`: 4, `raw_manifest`: 483, `availability`: 435, `dataset_metadata`: 8) | `stations(iaga_code, station_name, station_type, region, lat_deg, lon_deg, elev_m, orig_format, has_baseline, mean_vec_norm_nT, days_present, source_of_data, cutoff_rigidity_gv, barometric_coeff)` | 49 магнитометров (`station_type='magnetometer'`: 37 Europe, 12 Canada) + 1 `OULU` (`station_type='neutron_monitor'`). Долготы уже приведены к `[-180, 180]`. |
| `geomag/storm_20261004_05.parquet` | 135 360 / 2.26 МБ | `timestamp_utc: timestamp[us, tz=UTC]`, `station_iaga: large_string`, `x_nT, y_nT, z_nT, f_scalar_nT: double`, `is_valid_xyz, is_interpolated, was_despiked: bool` | 2 сут. (2 880 мин), 47 ст., `min_valid_st=46`, `is_interpolated=53`, `was_despiked=0`. |
| `geomag/storm_20261009_11.parquet` | 194 400 / 3.03 МБ | То же | 3 сут. (4 320 мин), 45 ст., `min_valid_st=44`, `is_interpolated=19`, `was_despiked=3`. |
| `geomag/storm_20261021_23.parquet` | 194 400 / 2.80 МБ | То же | 3 сут. (4 320 мин), 45 ст., `min_valid_st=40`, `is_interpolated=5`, `was_despiked=7`. |
| `geomag/storm_20261030_31.parquet` | 118 080 / 1.81 МБ | То же | 2 сут. (2 880 мин), 41 ст., `min_valid_st=37`, `is_interpolated=23`, `was_despiked=0`. |
| `solar_wind/solar1_l1_1min.parquet` | 14 400 / 2.09 МБ | `timestamp_utc`, `episode_id`, `time_frame` (`"L1"`), `bx/by/bz_{gse,gsm}_nT`, `b_mag_gsm_nT`, `proton_n_cm3`, `proton_T_K`, `vx/vy/vz_{gse,gsm}_kms`, `pdyn_nPa`, `swips_flags: int16`, `sweep_offset_sec: double`, `is_swips_valid: bool` | Временная шкала **L1** (нужен баллистический сдвиг `L1 → BSN`). Валидно: MAG `14 160/14 400`, SWiPS `13 045/14 400`. Обрыв `2026-10-31 ~23:15 UTC`. Диапазоны: `Bz [-28.89, +34.29] нТл`, `Vx [-886.1, -294.4] км/с`, `Pdyn [0.16, 42.38] нПа`. |
| `solar_wind/omni_bsn_1min.parquet` | 7 200 / 0.43 МБ | 49 столбцов (`timestamp_utc`, `episode_id`, `time_frame="BSN"`, IMF/плазма, `ae_nT`, `al_nT`, `au_nT`, `sym_d_nT`, `sym_h_nT`, `asy_d_nT`, `asy_h_nT`, `pc_n_index`, `electric_field_mVm`, и др.) | Только **2 из 4 эпизодов** (`Oct 04–05` и `Oct 09–11`). `ae/al/au` валидны `7200/7200` (`AE: 16..3169 нТл`, `AL: -3044..+25 нТл`), `bz_gsm`: `6085/7200`, `flow_speed/pdyn`: `4718/7200`. **`sym_h_nT`, `sym_d_nT`, `asy_d_nT`, `asy_h_nT`, `pc_n_index` = `0/7200` (100% NaN)** до догрузки обновлённого OMNI! |
| `cosmic_rays/oulu_nm_1min.parquet` | 14 400 / 0.27 МБ | `timestamp_utc`, `episode_id`, `fractional_date`, `uncorrected_cpm`, `corrected_cpm`, `pressure_mbar`, `delta_n_pct` | 0 пропусков, 60 с, `delta_n_pct ∈ [-7.43%, +9.77%]` (Форбуш-понижение относительно первых 6 ч эпизода). |

> **Критическая семантика масок в `geomag/storm_*.parquet`:** в `x_nT, y_nT, z_nT` короткие внутренние пропуски `≤ 3 мин` (всего 100 станция-минут по всем 4 бурям) **уже линейно интерполированы**, тогда как `is_valid_xyz` хранит исходную валидность *до* интерполяции. То есть `isfinite(x_nT) == (is_valid_xyz | is_interpolated)` точно до каждой строки.

### 1.2. Геомагнитные пояса, покрытия станций и расстояния (IGRF-14, эпоха `2026.78`)

Дипольные магнитные координаты `(MLat, MLon)` вычислены напрямую через `backend/geomag_core/mlt.py::geo_to_magnetic_dipole` (полюс IGRF-14: `80.86°N, -72.83°E`):

| Пояс (внутренний код) | Критерий | Кол-во | Станции (`IAGA`) | Медианная $\sigma(\Delta X, \Delta Y, \Delta Z)$ [нТл] | Медиана $d_{\text{NN}}$ [км] / соседи $\le 500$ км |
|---|---|---:|---|---|---|
| **Средние широты Европы** (`EU_mid`) | `region='Europe'`, `MLat < 60°` | **31** | `BDV, BEL, BFO, CLF, DOU, DUR, EBR, ESK, FUR, GCK, GUI, HAD, HLP, HRB, IZN, LON, LVV, MAB, NCK, NGK, NUR, PAG, PEG, SFS, SPT, SUA, THY, UPS, VAL, WIC, WNG` | `[22.1, 20.4, 9.3]` | `72..511 км` (`GUI`: `1355.5 км`), обычно `1..8` соседей в 500 км |
| **Авроральная Европа** (`EU_aur`) | `region='Europe'`, `MLat ≥ 60°` | **6** | `ABK (66.39°), HRN (74.37°), KIR (65.64°), LER (61.77°), LYC (62.88°), SOD (64.35°)` | `[105.1, 43.6, 62.3]` | `92..962 км` (`ABK–KIR`: `91.6 км`) |
| **Средние широты Канады** (`CA_mid`) | `region='Canada'`, `MLat < 60°` | **5** | `BRD (57.78°), OTT (54.53°), SBL (52.84°), STJ (56.06°), VIC (53.75°)` | `[21.5, 20.2, 15.3]` | **`763..1161 км` (`0` соседей в 500 км!)** |
| **Авроральная Канада** (`CA_aur`) | `region='Canada'`, `MLat ≥ 60°` | **7** | `BLC (72.37°), CBB (76.02°), FCC (67.06°), IQA (72.85°), MEA (61.03°), RES (82.37°), YKC (68.50°)` | `[93.9, 50.3, 78.3]` | **`626..1330 км` (`0` соседей в 500 км!)** |

**Три канонических отчётных пояса (по требованиям диссертации):**
- `europe_mid` = `EU_mid` (31 ст.);
- `auroral` = `EU_aur ∪ CA_aur` (все 13 ст. с `MLat ≥ 60°`, с дополнительной разбивкой `EU_aur` / `CA_aur`);
- `canada` = `CA_mid ∪ CA_aur` (все 12 ст. Канады, с разбивкой `CA_mid` / `CA_aur`);
- плюс агрегат `global` (все валидные станции).

### 1.3. Эмпирически выявленные дефекты и аномалии (подтверждено кодом)

1. **Автокорреляция минутного ряда:** медианная автокорреляция $\Delta X$ по всем станциям-эпизодам составляет **`r(1 мин) = 0.9975`**, `r(15 мин) = 0.9119`, `r(60 мин) = 0.6828` — случайное минутное разбиение `train/test` математически недопустимо.
2. **Неполное покрытие дат (14 из 49 станций имеют `< 10` сут.):**
   - **Критическая краевая ситуация — `SUA`:** присутствует **ТОЛЬКО в `storm_20261030_31`** (2 880 мин; `0` мин в эпизодах 1–3!). В 4-fold LOOE-CV при тесте на эпизоде 4 станция `SUA` имеет **0 обучающих точек** — пер-станционные скейлеры и фиксированные веса MLP для неё в Fold 3 необучаемы!
   - Остальные неполные: `LVV` (9 сут., только 512 мин в эп. 4 — уходит в разрыв 928 мин до главной фазы!), `RES` (8 сут., нет эп. 4), `SOD` (8 сут., 1 440 мин в эп. 3, смена кода `v → p`), `DOU`/`MAB` (7 сут., нет эп. 2), `CLF`/`GUI`/`SFS`/`SPT` (6 сут., обрыв в эп. 3 ~890 мин, нет эп. 4), `DUR`/`EBR`/`HRB` (5 сут., только эп. 1–2), `NCK` (5 сут., только эп. 3–4).
3. **Безбазисные вариационные станции (`has_baseline = 0`):** `GCK` (`‖meanXYZ‖ = 469.3 нТл`, все 4 бури) и `NCK` (`‖meanXYZ‖ = 2343.9 нТл`, только эп. 3–4). При этом внутри октября 2026 их межэпизодные медианы относительно стабильны (`GCK`: ±5 нТл, `NCK`: ±4 нТл), но на многомесячных данных возможны произвольные перекалибровки базы.
4. **Межэпизодные скачки базы у `Variation`-станций (`quality_code='v'`) с полным полем:** прямой аудит дневных и эпизодных медиан выявил негеофизические ступеньки `> 30 нТл` на **`LON`** (`Y`: `+72.1 нТл` относительно эп. 1 / размах **`76.0 нТл`** в `storm_20261030_31`, при `0 нТл` сдвиге у соседей `HAD`/`DOU`) и **`SFS`** (`(X,Y,Z)`: **`(+60.8, +34.8, +36.5) нТл`** в `storm_20261009_11`, с возвратом на прежний уровень в `storm_20261021_23`). Без обработки эти два эпизода создают фиктивный неустранимый RMSE `45–75 нТл` в `EU_mid`.
5. **10 одиночных 1-минутных спайков, очищенных в ETL (`was_despiked = True`):**
   - `VAL` (`2026-10-09 09:23, 10:39, 10:40 UTC` — до `-1231.3 нТл` по `Z`, `-333.7 нТл` по `Y`);
   - `OTT` (`2026-10-22 13:48, 13:53, 14:00 UTC` — до `-353.3 нТл` по `Z`);
   - `MEA` (`2026-10-23 17:33, 17:55, 17:56 UTC` — до `-86.8 нТл` по `Z`);
   - `IZN` (`2026-10-23 07:11 UTC` — `-192.8 нТл` по `Z`, `-153.3 нТл` по `Y`).
6. **Многоминутные приборные аномалии, ПРОШЕДШИЕ через одноточечный деспайкер ETL (подтверждено аудитом `|‖XYZ‖ - F|` и кросс-станционных невязок):**
   - **`VAL` (`2026-10-09 09:19 .. 10:42 UTC`, 84 минуты):** падение `Z` до `-1036 нТл` (`44986 → 43956 нТл`, уже `-25 нТл` в `09:19`, `-45 нТл` в `09:20`, `-79 нТл` в `10:41`, `-22 нТл` в `10:42`) и смещение `X/Y` на `±300 нТл` — одноточечный фильтр Хампеля (`d_in * d_out < 0`) снял только 3 точки (`09:23, 10:39, 10:40`), оставив всё плато в `storm_20261009_11.parquet`!
   - **`OTT` (`2026-10-22 13:45 .. 14:15 UTC`, 31 минута):** окно обслуживания/калибровки (совпадает с пропаданием скаляра `F` в `13:38..14:15` и скачком `|Δ(‖XYZ‖−F)| = 70.6 нТл`): после снятия 3 точек (`13:48, 13:53, 14:00`) в `storm_20261021_23.parquet` остались неочищенные приборные выбросы до `-113 нТл` по `Z` (`13:52`, `14:06`, `14:09`).
   - **`MEA` (`2026-10-23 17:54 .. 18:43 UTC`, 50 минут):** крупный приборный сбой до **`-320 нТл` по `Z`** (`|Δ(‖XYZ‖−F)| = 261.3 нТл` при изменениях `< 10 нТл` у `YKC`/`BRD`)! Критично: **`MEA` входит в тройку Zero-Shot станций (`LVV, BEL, MEA`)** — без исключения этого окна из `M_qc_clean` Zero-Shot оценка в Fold 2 искажается приборным дефектом!
   - **`IZN` (`2026-10-23 07:10 .. 07:13 UTC`, 4 минуты):** провал `-53.6 нТл` по `Z` / `-40 нТл` по `Y` в `07:10` непосредственно перед деспайкнутой точкой `07:11` (исказивший её линейную замену).
   - **`IZN` (`2026-10-30 07:16 .. 07:19 UTC`, 4 минуты):** локальный провал `ΔX = -157, ΔY = -217, ΔZ = -289 нТл`, тогда как ближайшие `PAG`/`SUA` изменились `< 2.2 нТл`.
   - Объединение всех 10 деспайкнутых точек и 5 многоминутных окон (`173` мин + изолированный спайк `MEA 17:33`) составляет ровно **174 станция-минуты**. Они имеют двойную ценность: (а) исключаются через `M_qc_clean` из `train`-лосса и чистой оценки качества, и (б) на Фазе 5 (`T5.4`) служат **реальным научным доказательством превосходства «Виртуального магнитометра» (Ветка B)** над классическим 1-минутным деспайкером!

### 1.4. Переиспользуемые модули `backend/` (строго `read-only` импорт)

Чтобы не дублировать верифицированную математику, `ml_research` импортирует следующие чистые функции из `backend/` (через адаптер `ml_research/geomag_ml/core_bridge.py`, без единого изменения в `backend/`):

- **`geomag_core.mlt`:**
  - `geo_to_magnetic_dipole(lat_deg, lon_deg) -> (mlat_deg, mlon_deg)` (IGRF-14 `2026.78`)
  - `compute_mlt_array(lat_deg, lon_deg, utc_timestamps) -> np.ndarray[float64]` (векторизованный `MLT ∈ [0, 24)`)
- **`geomag_core.conventions`:**
  - `haversine_km(lat1, lon1, lat2, lon2) -> float`
  - `has_valid_baseline(xyz, threshold_nT=10000.0) -> bool`
  - `interpolate_short_gaps`, `is_window_valid`, `get_band_spec`
- **`geomag_core.spectral` & `geomag_core.polarization` (Фаза 5 — проверка сохранения `Pc5/Pc6`):**
  - `compute_welch_spectral_matrix(xyz, dt=60.0, band="Pc5"|"Pc6", L=128|256, M=64, ...)`
  - `lookup_welch_effective_n(...)` / `welch_effective_dof(...)`
  - `stokes_from_spectral_matrix_2d(S, n_eff=..., alpha=0.05) -> Stokes2D(I, Q, U, V, p, psi_deg, chi_deg, epsilon, gamma2, ...)`
  - `polarization_3d(S) -> Polarization3D(P, theta_deg, b_over_a, ...)`
  - `find_valid_subsegments(...)`, `run_variant_b_pipeline(...)` из `geomag_core.analytic`
- **`geomag_core.cross_wavelet` (Фаза 5 — вейвлет-признаки и XWT/WTC):**
  - `build_morlet_scales(n_samples, dt=60.0, period_min_sec=120.0, period_max_sec=..., dj=1/12, omega0=6.0)`
  - `morlet_cwt_single(x_1d, scales, dt=60.0, omega0=6.0) -> (W_mat, coi_periods)` *(внимание: возвращает COI сразу в **периодах [с]**!)*
  - `compute_gap_coi_scales(valid_mask, dt=60.0)` + `MORLET_FOURIER_FACTOR` *(внимание: возвращает **масштабы [с]**, для сравнения с `periods_sec` обязательно умножать на `MORLET_FOURIER_FACTOR ≈ 1.033`!)*
  - `compute_xwt_wtc(x_i, x_j, ..., gap_policy="subsegments"|"gap_coi", n_surrogates=300, seed=42) -> XWTWTCResult`
- **`geomag_etl.intermagnet` & `geomag_etl.omni` (Фаза 0 — сборка недеспайкнутой тестовой версии и догрузка новых сырых данных):**
  - `parse_iaga2002_file(path) -> IagaDayRecord`
  - `parse_omni_hro_file(path) -> pd.DataFrame` *(внимание для `T0.5`: столбец `episode_id` внутри этого парсера захардкожен на 10 дат октября 2026 и даёт `<NA>` на новых датах — при расширенной догрузке слияние выполнять строго по `timestamp_utc`!)*
- **`geomag_api.data_store` (Фаза 0 — баллистический сдвиг `SOLAR-1 L1 → BSN`):**
  - `compute_ballistic_shift(timestamps_utc, vx_kms, ref_distance_km=1_400_000.0, smooth_window_min=31)` и `_resample_monotonic_series` (для полной автономности в Colab можно либо воспроизвести эту 25-строчную функцию в `ml_research`, либо запечь уже сдвинутые массивы `solar1_*_bsn` прямо в `.npz`-датасет на Фазе 0 — **принято запекать на Фазе 0**).

---

## 2. Жёсткие ограничения среды, изоляции, железа и переносимости

1. **Строгая изоляция от `GeoMag Analyzer`:**
   - Весь новый код, конфиги, ноутбуки, тесты, кэш датасета и артефакты живут **исключительно** в `c:\Programming\Projects\ONU_Thesis\ml_research\`.
   - Отдельное виртуальное окружение: `c:\Programming\Projects\ONU_Thesis\.venv_ml` на базе локального Python 3.12 (`C:\Program Files\Python312\python.exe`, версия `3.12.2`).
   - **ЗАПРЕЩЕНО** модифицировать любые файлы в `backend/`, `frontend/`, `tests/`, `.venv/`, `GeoMag_Storms_Dataset_Oct-2026/`, `data/processed/`, корневой `pytest.ini`.
   - `ml_research/` имеет собственные `ml_research/pyproject.toml`, `ml_research/pytest.ini` и `ml_research/.gitignore` (исключающий `data_cache/`, `artifacts/`, `data_external/`, `__pycache__/`).
2. **Учёт фактического локального железа:**
   - **CPU:** AMD Ryzen 5 6600HS Creator Edition (**6 физических ядер / 12 потоков**).
   - **RAM:** `14 702 МБ` (~13.69 ГиБ доступной ОЗУ; ~2 ГБ резервирует встроенная AMD Radeon).
   - **GPU:** выделенной NVIDIA GPU нет; встроенная AMD Radeon **не используется** (локально ставится CPU-сборка PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
   - **Ограничения на локальный CPU:** `torch.set_num_threads(6)` (по числу физических ядер, избегая треш-оверхеда SMT в OpenMP), `num_workers=0` в `DataLoader` на Windows (нулевой спавн-оверхед и минимальное потребление ОЗУ, датасет целиком ~15 МБ в памяти!).
   - Локально выполняются: **Фазы 0, 1, 2 полностью**, все `pytest`, smoke-прогоны Фаз 3–4 (`2–5 эпох`, `1 фолд`), Фаза 5 (анализ и расчёты `geomag_core`), Фаза 6 (агрегация, графики, ONNX-проверка).
3. **Переносимость в Google Colab / Kaggle (NVIDIA Tesla T4, 15–16 ГБ VRAM):**
   - Фазы 0–4 не зависят от `backend/` во время обучения: все координаты (`MLat`, `MLT`, матрица расстояний `station_dist_km`), флаги и сдвинутый `SOLAR-1` предвычислены в **одном компактном файле** `geomag_ml_<tag>.npz` (~12–18 МБ).
   - Единый CLI `python scripts/train.py --config <yaml> --dataset <npz> --device cuda|cpu` (или `.venv_ml\Scripts\python.exe ml_research/scripts/train.py ...` из корня репозитория) работает без единой правки локально и в облаке.
4. **Совместимость с будущим экспортом в `ONNX` (вне UI-интеграции сейчас):**
   - Все PyTorch-модели (`forward`) принимают и возвращают строго фиксированный кортеж плоских/регулярных `torch.Tensor` (`float32`/`bool`) без Python-словарей и без динамических циклов по элементам батча.
   - Каждый запуск сохраняет самодостаточный бандл в `ml_research/artifacts/<run_id>/` (контракт в §5.4).

---

## 3. Архитектурная концепция: две ветки (A и B) на общем фундаменте

Мы сознательно разделяем две физически конфликтующие задачи на **две независимые ветки моделей**, работающие поверх **единого датасета, единых сплитов, единого генератора масок, единых метрик и общего пространственного энкодера**:

```text
                       data/processed/ (read-only) + geomag_core.mlt
                                            │
                       [Фаза 0] Единый компактный датасет (.npz)
                     X[T,N,3], M[T,N], MLat/MLT, UT/DOY, OMNI/SOLAR1/OULU
                                            │
                       [Фаза 1] Сплиты (4-fold LOOE / block≥1d)
                             + Генератор масок (M1, M2, M3, M4)
                             + Единый модуль метрик (по поясам & X,Y,Z)
                      ┌─────────────────────┴──────────────────────┐
                      ▼                                            ▼
        ВЕТКА A — «НЕЙРО-ИНДЕКС»                    ВЕТКА B — «ВИРТУАЛЬНЫЙ МАГНИТОМЕТР»
   Глобальное сжатие: N×3 ──► z(t) ∈ R^(3..5)       Без узкого горлышка: S_obs ──► ΔB̂(j)
   Цель: глобальные режимы, связь с                 Цель: макс. точность в нТл, заполнение
   AE/SYM-H/SOLAR-1/OULU, фазы бури,                6–12ч пропусков (Pc5/Pc6), детектор
   нейро-ULF-индекс (вейвлеты)                      аномалий r = факт − прогноз (спайки/затм.)
                      │                                            │
   [Фаза 1] PCA (k=3,4,5) + Sq-detrend              [Фаза 1] Ridge (Schur) + IDW / Kriging (ZS)
   [Фаза 2] Masked Cond. MLP-AE / β-VAE (4N→k→3N)   [Фаза 2] Masked Wide MLP (4N→512×3→3N)
   [Фаза 3] Spatial Set Encoder + Perceiver-AE      [Фаза 3] Spatial Set Encoder + Cross-Attn VM
   [Фаза 4] Causal 1D-CNN + Perceiver (go/no-go)    [Фаза 4] Masked 1D-CNN + Cross-Attn (приоритет)
                      └─────────────────────┬──────────────────────┘
                                            ▼
           [Фаза 5] Научные эксперименты & Синтез A+B: ΔB = global(A) + meso(B−A) + local(fact−B)
                                            │
           [Фаза 6] Сводные таблицы, публикационные графики (300 DPI/SVG), ONNX-валидация
```

---

## 4. Полная структура каталога `ml_research/`

```text
ml_research/
├── .gitignore                          # data_cache/, artifacts/, data_external/, *.pt, *.onnx
├── pyproject.toml                      # Пакет `geomag_ml` (editable install в .venv_ml)
├── pytest.ini                          # testpaths = tests; изолировано от корневого pytest.ini
├── requirements_local_cpu.txt          # Закреплённые версии под Windows CPU (PyTorch CPU wheel)
├── requirements_colab_gpu.txt          # Лёгкий список доустановки для Colab / Kaggle T4
├── README.md                           # Краткая памятка запуска CLI локально и в облаке
│
├── configs/                            # Декларативные YAML-конфиги всех фаз и абляций
│   ├── dataset_oct2026.yaml            # Сборка текущего 10-сут. датасета (clean + undespiked)
│   ├── dataset_extended_template.yaml  # Шаблон для будущей догрузки (quiet Kp≤2, бури, OMNI, события)
│   ├── events_template.yaml            # Реестр интервалов quiet / storm / eclipse / spe
│   ├── phase1_baselines.yaml           # PCA (k=3,4,5, ±Sq), Ridge, IDW, Kriging
│   ├── phase2_branch_a_mlp.yaml        # Masked Cond. MLP-AE / β-VAE + сетка абляций
│   ├── phase2_branch_b_mlp.yaml        # Masked Wide MLP (единая сеть на все N станций)
│   ├── phase3_branch_a_perceiver.yaml  # Spatial Set Transformer + Perceiver-Bottleneck AE
│   ├── phase3_branch_b_cross_attn.yaml # Spatial Set Transformer + Cross-Attention VM (+ Zero-Shot)
│   ├── phase4_branch_b_temporal.yaml   # Shared Masked 1D-CNN → Spatial Transformer (causal/centered)
│   ├── phase4_branch_a_temporal.yaml   # Causal 1D-CNN → Perceiver-AE (проверка go/no-go)
│   └── phase5_experiments.yaml         # Параметры ULF-вейвлетов, 6–12ч пропусков, спайков, затмений
│
├── geomag_ml/                          # Основной Python-пакет
│   ├── __init__.py
│   ├── core_bridge.py                  # Read-only импорт backend/geomag_core и geomag_etl
│   ├── utils/
│   │   ├── seed.py                     # Детерминированный seed (random, numpy, torch, cuda)
│   │   ├── io.py                       # Атомарное чтение/запись JSON, YAML, NPZ, SHA-256
│   │   └── device.py                   # Выбор cpu/cuda, настройка torch threads (6 на Ryzen)
│   ├── data/
│   │   ├── schema.py                   # Константы контрактов, поясов, индексов, валидатор .npz
│   │   ├── build_dataset.py            # Сборка geomag_ml_*.npz из data/processed/ + raw (undespiked)
│   │   ├── ingest_extended.py          # Подгрузка будущих данных (quiet Kp≤2, бури, OMNI, Kp, события)
│   │   ├── preprocessing.py            # BaselineEstimator (3 режима) и FieldScaler (zscore / asinh)
│   │   ├── splits.py                   # 4-fold LOOE, block_temporal (gap ≥ 1d), Zero-Shot, window guard
│   │   ├── masks.py                    # Единый генератор масок M1, M2, M3 (dist-aware), M4 и смесей
│   │   └── datasets.py                 # PyTorch Dataset: SingleStepDataset и TemporalWindowDataset
│   ├── models/
│   │   ├── base.py                     # Общий интерфейс + декларация ONNX-сигнатуры
│   │   ├── baselines.py                # EM-PCA (±Sq), Ridge-Schur (M1–M4), IDW, Ordinary Kriging
│   │   ├── mlp_ae.py                   # Фаза 2 (A): Masked Conditional MLP-AE / β-VAE
│   │   ├── mlp_vm.py                   # Фаза 2 (B): Masked Wide Residual MLP
│   │   ├── spatial_encoder.py          # Общий энкодер Фаз 3–4: Set Transformer + SpatialBiasMLP
│   │   ├── perceiver_ae.py             # Фаза 3 (A): Perceiver-Bottleneck AE + Coordinate QueryDecoder
│   │   ├── cross_attn_vm.py            # Фаза 3 (B): Cross-Attention Virtual Magnetometer
│   │   ├── temporal_cnn.py             # Фаза 4: Shared Masked Dilated 1D-CNN (causal / centered)
│   │   └── temporal_models.py          # Фаза 4 (A и B): обёртки 1D-CNN → Spatial Transformer
│   ├── train/
│   │   ├── losses.py                   # Masked Weighted Huber, β-VAE KL, Cov(z)≈I, Mask-Consistency
│   │   ├── trainer.py                  # Единый цикл обучения (CPU/T4), early stopping, чекпоинты
│   │   └── artifacts.py                # Сохранение/загрузка artifacts/<run_id>/ + onnx_export_spec
│   ├── eval/
│   │   ├── metrics.py                  # RMSE/MAE/R² по поясам и (X,Y,Z), стабильность z(t), OMNI/SOLAR-1
│   │   ├── evaluator.py                # Унифицированный прогон модели по M1, M2, M3, M4 и Zero-Shot
│   │   └── stokes_wavelet_eval.py      # Обёртка над geomag_core: Stokes (I,p,ψ,ε), 3D, XWT/WTC
│   └── experiments/
│       ├── branch_a_physics.py         # Сверка z(t) с OMNI/SOLAR-1/OULU, траектории, фазы бурь (GMM/HMM)
│       ├── neuro_ulf_index.py          # Вейвлет-признаки (Pc5, Pc6, 30–120m, >2h) → нейро-ULF-индекс
│       ├── gap_filling_stokes.py       # 6–12ч пропуски на LVV/BEL/ABK/OTT + проверка Stokes & WTC
│       ├── anomaly_detection.py        # Детектор fact−pred: реальные VAL/OTT/IZN + синтетика 5..50 нТл
│       ├── eclipse_spe_protocol.py     # Готовый протокол для реальных дат затмений (M3) и SPE (M4)
│       └── synthesis_decomposition.py  # Разложение ΔB = global(A) + meso(B−A) + local(fact−B)
│
├── scripts/                            # CLI точки входа (одинаковые локально и в Colab/Kaggle)
│   ├── build_dataset.py                # Фаза 0: сборка clean + undespiked .npz
│   ├── run_baselines.py                # Фаза 1: прогон PCA, Ridge, IDW, Kriging
│   ├── train.py                        # Фазы 2–4: обучение одной конфигурации / фолда / абляции
│   ├── evaluate.py                     # Повторная оценка любого artifacts/<run_id>/
│   ├── check_phase2_gate.py            # Фаза 2: сравнение с PCA/Ridge + отчёт «цены сжатия» (A vs B)
│   ├── check_phase4_branch_a_gate.py   # Фаза 4 (A): формальный тест Go/No-Go с блочным бутстрепом
│   ├── run_phase5.py                   # Фаза 5: запуск научных экспериментов
│   ├── aggregate_benchmarks.py         # Фаза 6: сводная таблица всех моделей (CSV/MD/LaTeX)
│   ├── make_thesis_figures.py          # Фаза 6: публикационные графики (PNG 300 DPI + SVG)
│   ├── export_onnx.py                  # Фаза 6: экспорт в ONNX + проверка паритета onnxruntime
│   └── pack_for_cloud.py               # Сборка компактного архива кода + .npz для Colab/Kaggle
│
├── notebooks/                          # Тонкие обёртки (0 дублирования логики моделей!)
│   ├── colab_runner.ipynb              # Mount Drive → install → CLI train/eval → zip artifacts
│   └── kaggle_runner.ipynb             # Attach dataset → install → CLI train/eval → save outputs
│
├── tests/                              # Автотесты pytest (строго изолированы от корневых tests/)
│   ├── conftest.py                     # Синтетический мини-датасет (T=600, N=12) + фикстуры
│   ├── test_dataset_contract.py        # Фаза 0: формы, dtype, NaN, MLat/MLT, undespiked, extended
│   ├── test_preprocessing_and_splits.py# Фазы 0–1: отсутствие утечек B0/scaler/split/window/events
│   ├── test_masks_and_metrics.py       # Фаза 1: дизъюнктность M_obs∩M_tgt=∅, M1–M4, метрики в нТл
│   ├── test_baselines.py               # Фаза 1: PCA (±Sq), Ridge-Schur, IDW, Kriging, Zero-Shot
│   ├── test_phase2_mlps.py             # Фаза 2: формы 4N→3N, zero-leakage, NaN-safe, seed parity
│   ├── test_phase3_transformers.py     # Фаза 3: adversarial mask leakage (1e6→0), Zero-Shot, Perceiver
│   ├── test_phase4_temporal_cnn.py     # Фаза 4: Masked Conv1d на пропусках, causal leakage test!
│   ├── test_phase5_experiments.py      # Фаза 5: CWT-полосы, Stokes/WTC-пайплайн, ROC/PR, синтез
│   └── test_onnx_export.py             # Фаза 6: ONNX экспорт и max|y_pt - y_onnx| < 1e-4
│
├── data_cache/                         # Локальный кэш .npz (git-ignored)
├── data_external/                      # Место для будущих догрузок от соруководителя (git-ignored)
└── artifacts/                          # Результаты всех запусков artifacts/<run_id>/ (git-ignored)
```

---

## 5. Контракты данных, тензоров, конфигов и артефактов

### 5.1. Контракт файла датасета `ml_research/data_cache/geomag_ml_<tag>.npz`

Единый самодостаточный архив (сжатый `np.savez_compressed`, ~12–18 МБ для октября 2026) + сопутствующий `geomag_ml_<tag>.meta.json`:

| Ключ в `.npz` | Форма | Dtype | Семантика и инварианты |
|---|---|---|---|
| `X_raw` | `(T, N, 3)` | `float32` | Абсолютное (или вариационное для `GCK`/`NCK`) поле `(X, Y, Z)` в нТл. `NaN` везде, где `~M_valid`. Для октября 2026: `T = 14400`, `N = 49` (станции отсортированы строго лексикографически по `iaga_code`). |
| `M_valid` | `(T, N)` | `bool` | Основная маска валидности: `True`, если все 3 компоненты `(X, Y, Z)` конечны (`is_valid_xyz \| is_interpolated` в режиме по умолчанию; строго `is_valid_xyz` при `include_short_interp=False`). |
| `M_raw_valid` | `(T, N)` | `bool` | Исходная маска `is_valid_xyz` до интерполяции коротких пропусков `≤ 3 мин`. |
| `M_interpolated` | `(T, N)` | `bool` | Флаг `is_interpolated` (100 станция-минут в октябре 2026). |
| `M_despiked` | `(T, N)` | `bool` | Флаг `was_despiked` (10 станция-минут в октябре 2026). |
| `M_qc_clean` | `(T, N)` | `bool` | `M_valid` за вычетом подтверждённых многоминутных приборных сбоев (`VAL 2026-10-09 09:19..10:42`, `OTT 2026-10-22 13:45..14:15`, `MEA 2026-10-23 17:54..18:43`, `IZN 2026-10-23 07:10..07:13`, `IZN 2026-10-30 07:16..07:19 UTC`), эпизодов скачков базы `Variation` (`LON@ep3`, `SFS@ep1`, управляемо флагом `variation_step_policy`) и будущих записей из `qc_exclusions`. Используется в `train`-лоссе и чистой оценке; на Фазе 5 (`T5.4`) эти окна служат реальными позитивными метками детектора аномалий! |
| `M_known_spikes` | `(T, N)` | `bool` | Только в `*_undespiked_*.npz` (в clean — всё `False`): точное объединение (`174` станция-минуты) всех 10 реальных 1-мин спайков (`VAL, OTT, MEA, IZN`) и всех 5 многоминутных приборных окон (`VAL` 84м, `OTT` 31м, `MEA` 50м, `IZN` 4м+4м). |
| `timestamps_unix_s` | `(T,)` | `int64` | Метки времени UTC в секундах Unix, строго возрастающие внутри каждого эпизода с шагом `60`. |
| `episode_idx` | `(T,)` | `int16` | Индекс эпизода `0 .. E-1` (`E = 4` для октября 2026). |
| `episode_bounds` | `(E, 2)` | `int64` | Полуоткрытые границы `[start_idx, end_idx)` каждого эпизода. |
| `station_coords` | `(N, 4)` | `float32` | `[lat_deg, lon_deg, mlat_deg, mlon_deg]` (через `geomag_core.mlt.geo_to_magnetic_dipole`). |
| `station_dist_km` | `(N, N)` | `float32` | Симметричная матрица ортодромических расстояний (через `geomag_core.conventions.haversine_km`), диагональ `= 0.0`. |
| `station_has_baseline` | `(N,)` | `bool` | `False` для `GCK`, `NCK`; `True` для остальных 47 станций. |
| `station_belt_id` | `(N,)` | `int8` | `0=EU_mid`, `1=EU_aur`, `2=CA_mid`, `3=CA_aur`. |
| `mlt_hours` | `(T, N)` | `float32` | Магнитное местное время `MLT ∈ [0, 24)` ч (через `geomag_core.mlt.compute_mlt_array`). |
| `time_features` | `(T, 4)` | `float32` | `[sin(2π·UT/24), cos(2π·UT/24), sin(2π·DOY/365.25), cos(2π·DOY/365.25)]`. |
| `sw_omni` | `(T, 10)` | `float32` | Столбцы (порядок фиксирован в `schema.py`): `[ae, al, au, sym_h, pc_n, kp, bz_gsm, flow_speed, pdyn, electric_field]`. `NaN`, где нет покрытия. |
| `sw_solar1_bsn` | `(T, 6)` | `float32` | `[bz_gsm, b_mag_gsm, vx_gse, proton_n, pdyn_nPa, ey_mVm]`, где **`ey_mVm = +vx_gse * bz_gsm * 1e-3 = -|vx_gse| * bz_gsm * 1e-3`** (т.к. `vx_gse < 0` в GSE, при южном ММП $B_z < 0$ получаем $E_y > 0$ — точное совпадение с `OMNI electric_field_mVm`, $r = +0.99998$, MAE $= 0.014$ мВ/м!), сдвинутые `L1 → BSN` баллистически (`compute_ballistic_shift`, 1.4 млн км, медианное окно 31 мин). |
| `sw_solar1_l1` | `(T, 5)` | `float32` | Исходные (несдвинутые L1) `[bz_gsm, b_mag_gsm, vx_gse, proton_n, pdyn_nPa]` для лаг-анализа на Фазе 5. |
| `oulu_delta_n_pct` | `(T,)` | `float32` | Вариации нейтронного монитора Оулу `ΔN/N₀ [%]`. |
| `event_flags` | `(T,)` | `uint8` | Битовая маска тегов: `1=quiet_kp_le2`, `2=storm`, `4=eclipse`, `8=spe` (для октября 2026 все `2=storm`; расширяется через `events.yaml`). |

Сопутствующий `geomag_ml_<tag>.meta.json` хранит: `dataset_sha256`, `created_utc`, `station_codes: list[str]`, `station_regions: list[str]`, `episode_ids: list[str]`, `omni_cols`, `solar1_cols`, `calendar_span_days`, `recommended_use_doy: bool` (`True` только если `calendar_span_days >= 60`), `qc_exclusions`.

### 5.2. Контракты базовой линии $B_0$ и масштабирования (строго без утечки!)

#### Режимы базовой линии $B_0$ (`BaselineEstimator`, fit строго по `train`-индексам!)

1. **`temp_train_quiet_proxy` (временный режим по умолчанию для текущих 10 штормовых суток октября 2026):**
   - Для каждой станции $i$, имеющей `>= 60` валидных минут (`M_qc_clean == True`) в `train`: выбираются ночные (`MLT ∈ [22, 24) ∪ [0, 4]`) и квазиспокойные минуты внутри `train` — нижние 30% по сетевой медиане скорости вариации $v_{\text{net}}(t) = \operatorname{median}_{j \in \text{in\_train}} \sqrt{(\Delta_1 X_{t,j})^2 + (\Delta_1 Y_{t,j})^2}$ (агрегация строго по станциям `station_train_mask`, чтобы Zero-Shot станции не участвовали даже в оценке $v_{\text{net}}$!), где первая разность $\Delta_1$ вычисляется **строго внутри каждого эпизода** (`NaN` на первой минуте эпизода и после пропусков; 30-й перцентиль берётся по конечным $v_{\text{net}}(t)$ внутри `train`, чтобы многодневные стыки не искажали $v_{\text{net}}$!).
   - $B_0^{(i)} = \operatorname{median}_{t \in \mathcal{T}_{\text{quiet,train}}(i)} B_{\text{raw}}(t, i) \in \mathbb{R}^3$ (фолбэк: ночная медиана по `train`, если `< 60` точек).
   - **Почему не по-эпизодная медиана:** в октябре 2026 все 4 эпизода укладываются в 28 дней (вековой ход `< 3 нТл`), поэтому единая train-база $B_0^{(i)}$ **сохраняет реальный буревой сдвиг кольцевого тока (`Dst/SYM-H` ~ -100 нТл)**, имеет **0% утечки из `test`** и не ломается на неполных эпизодах вроде `LVV` (512 мин в эп. 4).
   - **Обработка `has_baseline=False` (`GCK`, `NCK`) и межэпизодных скачков базы `Variation` (`LON@ep3`, `SFS@ep1`):**
     - `baseless_policy ∈ {"train_quiet_proxy", "episode_initial_6h", "exclude"}` (для `GCK`/`NCK`, с обязательной абляцией `with_baseless` vs `without_baseless`);
     - `variation_step_policy ∈ {"flag_qc_exclude", "causal_initial_6h_reanchor", "keep_raw"}` (по умолчанию `"flag_qc_exclude"`: подтверждённые эпизоды скачков базы `LON@storm_20261030_31` и `SFS@storm_20261009_11` исключаются из `M_qc_clean` при обучении и чистом бенчмарке, но сохраняются в `T5.4` как дополнительный тест обнаружения базовых ступенек; детектор скачка `|med_{e1} - med_{e2}| > 30 нТл` автоматически пишет предупреждение в `.meta.json`).
   - **Базовая линия $B_0^{(j)}$ для Zero-Shot станций (`LVV`, `BEL`, `MEA` и одноэпизодной `SUA`):** чёткое разделение понятий! В Zero-Shot эти станции полностью исключены из обучения нейросети и `FieldScaler` (`station_train_mask[j] = False`, скейлер использует поясной фолбэк `asinh_global` / belt-median), но чтобы оценить ошибку вариаций $\Delta\hat{B} - \Delta B$ на тестовом эпизоде (из абсолютного поля ~50 000 нТл), требуется статическая обсерваторская константа $B_0^{(j)}$: для `LVV`, `BEL`, `MEA` она оценивается **строго по нетестовым эпизодам (`train_idx`, 0% касания `test`!)**, а для одноэпизодной `SUA` (Fold 3, где `0` минут вне `ep3`) — строго по каузальным первым 6 часам `ep3` до активной фазы (с явной пометкой в `metrics.json`).
2. **`target_kp_quiet_night` (целевой режим после догрузки спокойных дней `Kp ≤ 2` и полного `OMNI`):**
   - $B_0^{(i)}$ вычисляется как покомпонентная медиана по всем валидным минутам `train` (или выделенных калибровочных спокойных суток, не пересекающихся с `val`/`test`), удовлетворяющим одновременно: `Kp ≤ 2.0`, `MLT ∈ [22, 24) ∪ [0, 4]` и (при наличии `SYM-H`) `|SYM-H| ≤ 15 нТл`.
   - При многомесячных рядах (`> 60` дней) дополнительно включается плавная линейная/кусочно-постоянная коррекция векового хода по спокойным дням `train` и отдельная перепривязка `GCK`/`NCK` по ближайшим предшествующим спокойным часам (обязательная абляция `with_baseless` vs `without_baseless`).
3. **`per_episode_median_ablation` (только для абляции/сравнения):** вычитание медианы каждого эпизода — используется исключительно как контрольный эксперимент для количественной демонстрации потери буревого сдвига `Dst/SYM-H`.

#### Два режима масштабирования (`FieldScaler`, fit строго по `train` с `M_qc_clean == True`!)

1. **`zscore_per_station` (покомпонентный z-score):**
   $$\tilde{x}_{t,i,c} = \frac{\Delta B_{t,i,c} - \mu_{i,c}}{\max(\sigma_{i,c},\, \sigma_{\min})}, \qquad \sigma_{\min} = 5.0\text{ нТл}$$
   - $\mu_{i,c}, \sigma_{i,c}$ оцениваются робастно (либо стандартные `mean/std` с клиппингом выбросов `> 10·MAD` при фите, чтобы аномалии не раздували $\sigma$).
   - Для станций, отсутствующих в `train` (Zero-Shot `LVV, BEL, MEA` или `SUA` в Fold 3): фолбэк на поясную медиану $(\mu_{\text{belt}(i), c}, \sigma_{\text{belt}(i), c})$ — **тестовые данные никогда не используются при фите скейлера!**
2. **`asinh_global` (мягкое гиперболическое сжатие — рекомендовано по умолчанию для Фаз 3–4 и Zero-Shot, и обязательно в абляции Фазы 2!):**
   $$\tilde{x}_{t,i,c} = \operatorname{asinh}\!\left(\frac{\Delta B_{t,i,c}}{s_c}\right), \qquad \Delta\hat{B}_{t,i,c} = s_c \sinh(\hat{\tilde{x}}_{t,i,c})$$
   - Единый на всю сеть 3-вектор масштаба $s = (s_X, s_Y, s_Z)$ вычисляется по `train`-фолду как медиана станционных `IQR/1.349` среднеширотного пояса `EU_mid` (ориентировочно `s ≈ [22, 20, 10] нТл`, с нижним порогом `s_min = 10.0 нТл`, либо фиксируемая константа `s = [25.0, 25.0, 15.0] нТл`).
   - Свойства: линейна при $|\Delta B| \ll s$ (сохраняет малые `Pc5/Pc6`), логарифмически сжимает авроральные пики `500–1500 нТл` (устраняет 25-кратный перекос дисперсий в лоссе), гладкая, нечётная, строго обратимая и **не зависит от ID станции** (идеальна для Zero-Shot).

### 5.3. Точные формы тензоров на границах модулей

| Уровень / модуль | Входной тензор | Форма | Dtype | Обработка `NaN` и масок | Выходной тензор |
|---|---|---|---|---|---|
| **Фаза 2 (MLP A и B)** | `x_flat` | `(B, 4*N)` | `float32` | `[ΔB̃_obs_flat (B, 3*N), m_obs (B, N)]`; где `m_obs==0` строго `0.0` через `where/nan_to_num` (никаких `NaN` — в IEEE-754 `0 * NaN == NaN`!) | `y_hat`: `(B, 3*N)` → `(B, N, 3)`; для Ветки A также `z`: `(B, k)` |
| **Фаза 2 (Cond A)** | `c_time` | `(B, d_t)` | `float32` | `d_t ∈ {0, 2, 4}` (`[sin_UT, cos_UT]` ± `[sin_DOY, cos_DOY]`); при `d_t == 0` передаётся `None` (опускается из входов ONNX) | Подаётся только в декодер Ветки A |
| **Фаза 3 (Spatial Set Enc)** | `obs_tokens`, `obs_mask`, `pair_geom` | `(B, N, d_in)`, `(B, N)`, `(B\|1, N, N, 5)` | `float32`, `bool`, `float32` | Конфигурируемый `d_in` (в Фазе 3 `d_in = 9`: `[ΔX̃, ΔỸ, ΔZ̃, MLat/90, sin_MLT, cos_MLT, lat/90, sin_lon, cos_lon]`; в Фазе 4 `d_in = 105`; в `T5.2` `d_in = 18`). Безопасное зануление `torch.where(obs_mask[..., None], torch.nan_to_num(obs_tokens, 0.0), 0.0)` + `fp32/finfo.min` маска внимания `~obs_mask` | `H_obs`: `(B, N, d_model)` (`d_model=128`) |
| **Фаза 3 (A: Perceiver)** | `H_obs`, `obs_mask`, `q_coords` | `(B, N, 128)`, `(B, N)`, `(B, N_q, d_q)` | `float32`, `bool`, `float32` | `d_q = 6 + d_t` (`[MLat/90, sin_MLT, cos_MLT, lat/90, sin_lon, cos_lon]` + `c_time`). `N_q` может быть любым; конфигурируемый выход `d_out` (`3` в Фазах 3–4, `8` в `T5.2`) | `y_hat`: `(B, N_q, d_out)`, `z`: `(B, k)` |
| **Фаза 3 (B: Cross-Attn)** | `H_obs`, `obs_mask`, `q_coords`, `q_pair_geom`, `attn_forbidden` | `(B, N, 128)`, `(B, N)`, `(B, N_q, d_q)`, `(B\|1, N_q, N, 5)`, `(B, N_q, N)` | `float32`, `bool` | Явная маска запрещённых ключей `attn_forbidden[b, q, k] = (~obs_mask[b, k]) \| (q_id[q] == k_id[k])` **в дополнение к обязательному `obs_mask[b, j] == False` до `SpatialSetEncoder`** для всех целевых станций `M_target` (предотвращает multi-hop утечку через Self-Attention!) | `y_hat`: `(B, N_q, 3)` |
| **Фаза 4 (Temporal CNN)** | `X_win`, `obs_mask` | `(B, N, 4, W)`, `(B, N)` | `float32`, `bool` | `4` канала = `[ΔX̃, ΔỸ, ΔZ̃, m_step]`, `W ∈ {32, 64}`. Если `obs_mask[b, i] == False` или `m_step == 0`, первые 3 канала строго `0.0` через `torch.where`/`nan_to_num` и `m_step = 0.0`! | `h_temp`: `(B, N, 96)` → конкат `[h_temp, ΔB̃(t), coords]` (`d_in=105`) → `SpatialSetEncoder` |
| **Фаза 5 (Neuro-ULF)** | `W_pow`, `M_coi` | `(T, N, 8)`, `(T, N, 4)` | `float32`, `bool` | 4 полосы (`Pc5`, `Pc6`, `30–120m`, `>2h`) × 2 (`log10(1 + P_H)`, `log10(1 + P_Z)`, где `P` по Torrence & Compo Eq. 24 [нТл²]); на вход энкодера подаётся `[W_pow_norm_masked (8), M_coi (4), coords (6)]` (`d_in=18`, `d_out=8`), `obs_mask = M_coi.any(-1)` | `z_ulf`: `(T, k)`, `W_pow_hat`: `(T, N, 8)` |

### 5.4. Контракт артефактов запуска `ml_research/artifacts/<run_id>/`

Формат имени: `<phase>_<branch>_<model>_<dataset_tag>_f<fold>_s<seed>_<YYYYMMDD_HHMMSS>` (например, `p3_B_cross_attn_oct2026_f0_s42_20261201_142011`).

Обязательные файлы в каждой папке `<run_id>/`:
1. **`model.pt`** — `{"model_class": str, "model_kwargs": dict, "state_dict": OrderedDict, "epoch": int, "best_val_metric_nT": float}` (для линейных бейзлайнов Фазы 1: `model.npz` с матрицами весов/компонент).
2. **`config.json`** — полностью разрешённый конфиг эксперимента + `git_commit`, `dataset_path`, `dataset_sha256`, `fold_id`, `seed`, `device`, `elapsed_sec`, `n_params`.
3. **`scaler.json`** — `baseline_mode`, `baseless_policy`, `B0_nT` (`[N, 3]`), `in_train_stations` (`[N]`), `scaler_mode` (`zscore_per_station` | `asinh_global`), параметры `mean/std` или `scale_s` (`[3]`), `fit_split_hash`.
4. **`stations.json`** — упорядоченный список из `N` записей: `{idx, iaga_code, region, belt, lat_deg, lon_deg, mlat_deg, mlon_deg, has_baseline, in_train, is_zero_shot}`.
5. **`metrics.json`** — стандартизованный JSON:
   - `eval_masks`: `{"M1": {...}, "M2": {...}, "M3_500km": {...}, "M3_1000km": {...}, "M4": {...}, "zero_shot": {"heldout_zero_shot": {...}, "single_episode_unseen": {...}, "per_station": {...}}}`, где внутри каждого режима лежат `global`, `europe_mid`, `auroral` (`europe_auroral`, `canada_auroral`), `canada` (`canada_mid`, `canada_auroral`), и внутри каждого пояса — `X`, `Y`, `Z`, `vector_3d` с полями `{rmse_nT, mae_nT, r2, pearson_r, n_points}`. Разделение `heldout_zero_shot` (`is_zero_shot=True`: `LVV, BEL, MEA` с базой из `train`) и `single_episode_unseen` (`SUA` в Fold 3 с каузальной базой первых 6 ч `ep3`) исключает смешивание одноэпизодной `SUA` с 4-фолдовым Zero-Shot средним!
   - Для Ветки A дополнительно: `latent_stability` (`M2_20`, `M2_30`, `M2_40`: `cos_sim`, `rel_l2_drift`, `axis_corr_mean`), `axis_decorr_frob`, `space_weather_alignment` (корреляции и линейный $R^2$ на `OMNI`/`SOLAR-1 BSN`/`OULU`).
6. **`onnx_export_spec.json`** — `{opset: 17, input_names: [...], output_names: [...], dynamic_axes: {...}, sample_shapes: {...}, verified_max_abs_diff: float | null}`.
7. **`history.csv`** — лог по эпохам (`epoch, train_loss, val_loss, val_rmse_global_nT, val_rmse_eu_mid_nT, val_rmse_auroral_nT, lr, epoch_sec`).
8. **`val_outputs.npz`** (опционально для лучшего чекпоинта) — компактные `float32` массивы `z_test` (Ветка A) и `y_pred_m1` / `y_pred_zs` (Ветка B) для быстрого запуска Фазы 5 без переинференса.

---

## 6. Спецификации моделей, функций потерь и стратегий обучения (Фазы 1–4)

### 6.1. Фаза 1 — Линейные бейзлайны (A и B)

| Ветка | Модель | Математическая спецификация | Гиперпараметры и обучение |
|---|---|---|---|
| **A** | **`PCA` (`k = 3, 4, 5`)** | Вектор $x_t \in \mathbb{R}^{3N_{\text{tr}}}$ по станциям `in_train` (в масштабированном пространстве, по `M_qc_clean`, без утечки `test`). Обучение: оценка среднего $\mu_{\text{tr}}$ + итеративный **EM-PCA** (20 итераций) на центрированной матрице $x_t - \mu_{\text{tr}}$ с реальными пропусками (SVD на каждом шаге, ортонормированный базис $W \in \mathbb{R}^{3N_{\text{tr}} \times k}$). Инференс под любой маской $O$ (вкл. `M1–M4`): точная проекция МНК по наблюдаемым координатам $\hat{z}_t = (W_O^\top W_O + 10^{-4} I_k)^{-1} W_O^\top (x_{t,O} - \mu_O)$, восстановление $\hat{x}_t = \mu + W \hat{z}_t$. | `k ∈ {3, 4, 5}`, `scaling ∈ {zscore_per_station, asinh_global}`. Время: `< 5 с` на CPU. |
| **A** | **`Sq-detrended PCA` (`k = 3, 4, 5`)** | Предварительно по `train` (`M_qc_clean`) для каждой станции-компоненты оценивается гармонический суточный ход $S_q(\text{UT}) = a_0 + \sum_{m=1}^{M_h} [a_m \cos(\frac{2\pi m}{24}\text{UT}) + b_m \sin(\frac{2\pi m}{24}\text{UT})]$ с **масштабированной по числу наблюдений $\ell_2$-регуляризацией $\lambda = 0.01 \cdot n_{\text{obs}}$ на гармоники $m \ge 1$** (`0` на $a_0$) и **понижением порядка $M_h = 1$ при охвате `< 20/24` часовых бинов UT** (иначе `M_h = 4`) *(критично: в Fold 0 у `SFS` только `885` чистых мин = `15/24` ч UT в `ep2`, и немасштабированный $\ell_2=10^{-2}$ при $M_h=4$ экстраполирует на `+261 нТл` в ненаблюдаемых часах UT, давая RMSE `112 нТл`, тогда как адаптивный $M_h=1, \lambda=0.01\cdot n_{\text{obs}}$ ограничивает $|S_q| \le 23$ нТл и снижает RMSE до `26.3 нТл`!)*. Затем центрированный EM-PCA на остатках $x_t - S_q(\text{UT}_t)$; при реконструкции $S_q(\text{UT}_t)$ прибавляется обратно. | Те же `k ∈ {3, 4, 5}`. Показывает долю дисперсии, приходящуюся на регулярный $S_q$. |
| **B** | **`Ridge-Schur` (`M1–M4`)** | По `train` (`M_qc_clean`, станции `in_train`) оцениваются $\mu$ и ковариационная матрица $\Sigma \in \mathbb{R}^{3N_{\text{tr}} \times 3N_{\text{tr}}}$ методом **итеративного EM-Gaussian (15–20 итераций) + усадка $(1-\gamma)\Sigma + \gamma\operatorname{diag}(\Sigma)$** *(критично: в 5 из 6 пар эпизодов есть `3–32` пары станций с `0` совместно валидных минут, например `DOU↔DUR` или `CLF↔NCK` — наивная попарная ковариация даёт `NaN` и роняет `eigh`, тогда как EM-ковариация гарантированно конечна, строго PD и транзитивно восстанавливает корреляции через общие 35 станций!)*. Для любой маски $(O, U)$: $\hat{x}_{t,U} = \mu_U + \Sigma_{UO}(\Sigma_{OO} + \alpha I)^{-1}(x_{t,O} - \mu_O)$. | Подбор $\alpha \in \{10^{-2}, 10^{-1}, 1, 10, 10^2\}$ на `inner-val`. Время: `< 20 с` на CPU. |
| **B** | **`IDW` (Zero-Shot)** | Взвешенная интерполяция (строго в пространстве `ΔB [нТл]` или `asinh_global`, НЕ пер-станционный z-score!) по расстоянию в геомагнитных координатах $d_{\text{mag}}(i,j) = \sqrt{(\Delta\text{MLat})^2 + (\lambda_a \cos\overline{\text{MLat}}\,\operatorname{wrap}_{180}(\Delta\text{MLon}))^2}$ (и сравнение с `haversine_km`): $w_{ji} \propto (d_{ji} + d_0)^{-p}$, $\hat{B}_j = \sum_{i \in O_K} \tilde{w}_{ji} B_i$. | Подбор $p \in \{1.5, 2.0, 3.0\}$, $K \in \{5, 10, \text{all}\}$, анизотропия $\lambda_a \in \{0.5, 1.0\}$ на `inner-val`. |
| **B** | **`Ordinary Kriging` (Zero-Shot)** | Пространственный гауссовский процесс с условием несмещённости $\sum_i w_i = 1$ (строго в `ΔB [нТл]` или `asinh_global`!) и анизотропным ядром Матерна-3/2 в `(MLat, MLon)` отдельно для `X, Y, Z`: $k(i,j) = \sigma_s^2 (1 + \sqrt{3} r_{ij})\exp(-\sqrt{3} r_{ij}) + \sigma_n^2 \delta_{ij}$, $r_{ij} = \sqrt{(\Delta\text{MLat}/L_\phi)^2 + (\operatorname{wrap}_{180}(\Delta\text{MLon})\cos\overline{\text{MLat}}/L_\lambda)^2}$. Параметры $(L_\phi, L_\lambda, \sigma_n/\sigma_s)$ подбираются по эмпирической вариограмме на `train`. | Работает на `M1–M4` и на полностью исключённых Zero-Shot станциях (`LVV, BEL, MEA`). |

### 6.2. Фаза 2 — Уровень 1 DL: Одноточечные Masked MLP

#### Общая схема входа/выхода (строго для обеих веток)
- Вход в минуту $t$: единый плоский вектор длины **`4N`** (`196` для `N = 49`):
  $$u_t = \bigl[\underbrace{(x_t \odot m_t^{\text{obs}})_{\text{flat}}}_{3N},\;\underbrace{m_t^{\text{obs}}}_{N}\bigr] \in \mathbb{R}^{4N}$$
  где $m_{t,i}^{\text{obs}} = 0$ на реальных пропусках **и** искусственно скрытых станциях, и в этих позициях значения поля строго `0.0`.
- Выход: единый вектор длины **`3N`** (`147` для `N = 49`) — предсказанные $(\Delta\hat{X}, \Delta\hat{Y}, \Delta\hat{Z})$ для всех `N` станций одновременно.

#### Ветка A — `Masked Conditional MLP-AE` / `β-VAE` (`mlp_ae.py`)
- **Энкодер:** `Linear(4N, 256) → LayerNorm → GELU → Dropout(0.1) → PreNormResMLP(256, drop=0.1) → Linear(256, 128) → LayerNorm → GELU`
  - Режим `AE`: `z = Linear(128, k)` (`k ∈ {3, 4, 5}`).
  - Режим `β-VAE`: `μ = Linear(128, k)`, `logvar = clamp(Linear(128, k), -6.0, 4.0)`; на `train` $z = \mu + e^{0.5\text{logvar}}\varepsilon$, на `eval` $z = \mu$.
- **Условный декодер:** вход $v_t = [z_t,\; c_{\text{time}}(t)] \in \mathbb{R}^{k + d_t}$, где $c_{\text{time}} = [\sin\text{UT}, \cos\text{UT}]$ (`d_t=2`; при наличии `>= 60` дней также `sin/cos(DOY)`, `d_t=4`; в абляции `no_UT`: `d_t=0`).
  - `Linear(k + d_t, 128) → LayerNorm → GELU → Linear(128, 256) → LayerNorm → GELU → PreNormResMLP(256, drop=0.1) → Linear(256, 3N)`.
- **Число параметров:** `~422K` (при `N=49, k=5, d_t=2`).
- **Функция потерь:**
  $$\mathcal{L}_A = \frac{\sum_{t,i,c} w_{t,i}\,\ell_{\text{Huber},\delta}(\hat{y}_{t,i,c} - y_{t,i,c})}{\sum_{t,i,c} w_{t,i}} + \beta(e)\,\mathcal{L}_{\text{KL}} + \lambda_{\text{cov}}\,\mathcal{L}_{\text{decorr}}(\mu) + \lambda_{\text{cons}}\,\mathcal{L}_{\text{cons}}$$
  где (все регуляризаторы считаются в `float32` вне AMP):
  - $w_{t,i} = M_{t,i}^{\text{qc}} \cdot \mathbb{I}[\text{in\_train}_i] \cdot (w_{\text{tgt}} M_{t,i}^{\text{tgt}} + w_{\text{obs}} M_{t,i}^{\text{obs}})$ с $w_{\text{tgt}} = 2.0$, $w_{\text{obs}} = 1.0$, $\delta = 1.0$;
  - $\mathcal{L}_{\text{KL}} = -\frac{1}{2Bk}\sum_{b,d} (1 + \log\sigma^2_{b,d} - \mu_{b,d}^2 - \sigma^2_{b,d})$ (среднее по батчу и $k$, линейный разогрев $\beta(e)$ за 15 эпох до $\beta_{\max} \in \{10^{-3}, 10^{-2}, 5\cdot 10^{-2}\}$);
  - $\mathcal{L}_{\text{decorr}}(\mu) = \frac{1}{k^2}\|\operatorname{Cov}(\mu) - I_k\|_F^2$ (несмещённая батч-ковариация по центрированному $\mu$ при $B \ge 8$, строго **`0.0` при $B < 8$** — защищает хвостовые мини-батчи Фазы 4 вроде `5634 % 64 = 2`!; $\lambda_{\text{cov}} \in \{0.0, 0.05, 0.1\}$ — одновременно ортогонализирует оси $\operatorname{Cov}_{a\neq b} \to 0$ и фиксирует единичную дисперсию $\operatorname{Var}(\mu_a) \to 1$, не допуская ни коллапса, ни расхождения масштаба!);
  - $\mathcal{L}_{\text{cons}} = \frac{1}{Bk}\sum_b \|\mu(u_b^{(1)}) - \mu(u_b^{(2)})\|_2^2$ строго по детерминированному $\mu$ (= $z$ в режиме `AE`, не конфликтует со стохастическим $\varepsilon$ в `β-VAE`!) на двух независимых масках момента $t$ ($\lambda_{\text{cons}} \in \{0.0, 0.1\}$; автоматически `0.0` в абляции `train_mask_mode="none"`).
- **Обязательная сетка абляций Ветки A:** (1) `k ∈ {3, 4, 5}`, (2) `с UT` vs `без UT` в декодере, (3) `zscore_per_station` vs `asinh_global`, (4) `plain AE` vs `AE + Cov(z)≈I` vs `β-VAE`, (5) обучение с масками (`M1+M2`) vs `без искусственных масок`.

#### Ветка B — `Masked Wide MLP` (`mlp_vm.py`)
- **Единая модель на всю сеть** (строго НЕ 49 раздельных моделей!):
  - `Linear(4N, 512) → LayerNorm → GELU → 3 × PreNormResMLP(512, hidden=512, drop=0.15) → LayerNorm → Linear(512, 3N)`.
  - `PreNormResMLP(d)`: `x + Dropout(Linear(d, d)(GELU(Linear(d, d)(LayerNorm(x)))))`.
- **Число параметров:** `~1.75M`.
- **Функция потерь:** взвешенный `HuberLoss(δ=1.0)` по `M_qc_clean` с повышенным весом на скрытые станции: `w_tgt = 4.0`, `w_obs = 0.25` (и абляция `w_obs = 0.0`).
- **Стратегия маскирования на `train`:** смесь `35% M1 + 35% M2(p∈[0.20, 0.35]) + 30% M3(R∈[500, 1000] км)`.
- **Оптимизатор и бюджет (Фаза 2, обе ветки):** `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR` + 5 эпох linear warmup, `batch_size=128`, `max_epochs=100`, `early_stopping(patience=15)` по `val_masked_rmse_nT`, `grad_clip=1.0`.
- **Ориентировочное время:** локально на CPU (Ryzen 6c/12t) `~30–60 с` на фолд (Ветка A) и `~60–110 с` на фолд (Ветка B); на Colab T4 `< 15 с` на фолд.

### 6.3. Фаза 3 — Уровень 2 DL: Одноточечные Spatial Set Transformers

#### Общий пространственный энкодер (`spatial_encoder.py`)
- **Вход:** все наблюдаемые и валидные станции $S_{\text{obs}}(t)$ как неупорядоченное множество токенов $(N_{\text{obs}}, d_{\text{in}})$ (в батче дополняются до $N$ с `obs_mask: bool[B, N]`):
  $$e_i(t) = \bigl[\Delta\tilde{X}_i, \Delta\tilde{Y}_i, \Delta\tilde{Z}_i,\; \tfrac{\text{MLat}_i}{90},\; \sin\text{MLT}_i,\; \cos\text{MLT}_i,\; \tfrac{\text{lat}_i}{90},\; \sin\text{lon}_i,\; \cos\text{lon}_i\bigr] \in \mathbb{R}^9$$
  - Масштабирование по умолчанию: **`asinh_global`** (станционно-независимое, обеспечивает Zero-Shot!).
  - Опциональная абляция `station_emb`: `Embedding(N + 1, d_emb=8)` (индекс `N` = `<UNK>`, **никогда `-1`** для совместимости с CUDA/ONNX, dropout ID `p=0.3` на обучении) — базовый вариант для Zero-Shot строго **`use_station_emb=False`**.
- **Относительный пространственный bias (`SpatialBiasMLP`):**
  - Для каждой пары $(i, j)$ формируется вектор относительной геометрии:
    $$p_{ij} = \Bigl[\frac{d_{\text{km}}(i,j)}{2000},\; \ln\!\bigl(1 + \frac{d_{\text{km}}(i,j)}{500}\bigr),\; \frac{\text{MLat}_i - \text{MLat}_j}{30},\; \sin\Delta\text{MLon}_{ij},\; \cos\Delta\text{MLon}_{ij}\Bigr] \in \mathbb{R}^5$$
    (Замечание: $\Delta\text{MLT}_{ij} \equiv \Delta\text{MLon}_{ij} / 15\text{ ч}$ инвариантно ко времени, поэтому матрица $p_{ij}$ вычисляется ровно один раз и не пересчитывается на каждом шаге!)
  - `Linear(5, 16) → GELU → Linear(16, n_heads)` выдаёт поправку $b_{ij}^{(h)}$, прибавляемую к логитам внимания $Q K^\top / \sqrt{d_k} + B_{\text{spatial}}$ перед `softmax`.
- **Ствол энкодера:** конфигурируемый вход `Linear(d_in, 128) → LayerNorm` (`d_in = 9` в Фазе 3, `d_in = 105` в Фазе 4, `d_in = 18` в `T5.2`) → `3 × PreNormSelfAttnBlock(d_model=128, n_heads=4, d_ff=256, drop=0.1, spatial_bias=True)`.
- **Гарантия отсутствия утечки и FP16/AMP-безопасность на Tesla T4:**
  1. `obs_tokens = torch.where(obs_mask[..., None], torch.nan_to_num(obs_tokens, nan=0.0), 0.0)` до первого слоя *(строго `torch.where`/`nan_to_num`, НЕ наивное умножение `* obs_mask`, т.к. в IEEE-754 `0 * NaN == NaN`!)*;
  2. Логиты внимания и `softmax` вычисляются в **`float32`** (либо с `mask_val = torch.finfo(attn_logits.dtype).min`, **ЗАПРЕЩЕНО** использовать `-1e9` в `fp16` — на Tesla T4 без `bf16` это вызывает `RuntimeError: value cannot be converted to type at::Half without overflow`!): `attn_logits = attn_logits.float().masked_fill(key_forbidden, -1e9)`, `attn = softmax(attn_logits, dim=-1)`, и защита от полностью замаскированной строки `attn = torch.where(all_forbidden, 0.0, attn)`;
  3. После энкодера `H_obs = H_obs * obs_mask[..., None]`.

#### Ветка A — `Perceiver-Bottleneck AE` (`perceiver_ae.py`)
- **Bottleneck-агрегатор:** $K_q = 4$ обучаемых латентных query-векторов $Q_{\text{lat}} \in \mathbb{R}^{4 \times 128}$ → 1 слой `PreNormCrossAttn(Q=Q_lat, K=V=H_obs, key_mask=~obs_mask)` → 1 слой `PreNormSelfAttn(Q_lat)` → flatten `(4*128=512)` → `Linear(512, 128) → GELU → Linear(128, k)` (`k ∈ {3, 4, 5}`, опц. `β-VAE` или `Cov(μ)≈I`).
- **Координатный Query-декодер:** для произвольной целевой станции $j$ с вектором координат $c_j(t) = [\frac{\text{MLat}_j}{90}, \sin\text{MLT}_j, \cos\text{MLT}_j, \frac{\text{lat}_j}{90}, \sin\text{lon}_j, \cos\text{lon}_j, c_{\text{time}}(t)] \in \mathbb{R}^{d_q}$ (`d_q = 6 + d_t`):
  - `q_j = Linear(d_q, 128) → LayerNorm → GELU`
  - **FiLM-модуляция от нейро-индекса $z$:** `γ, β = chunk(Linear(k, 256)(z), 2)`, `h_j = (1 + γ) * q_j + β`
  - `Linear(128 + k, 128)([h_j, z]) → LayerNorm → GELU → PreNormResMLP(128) → Linear(128, d_out)` → $\Delta\hat{B}_{\text{global}}(t, j)$ (конфигурируемый **`d_out: int = 3`** в Фазах 3–4 и **`d_out = 8`** в вейвлет-эксперименте `T5.2`!).
- **Число параметров:** `~785K`.
- **Лосс и маски:** то же, что в Фазе 2 (A), с обязательным включением `M2` (`p ∈ [0.15, 0.40]`) и `λ_cons = 0.1` для устойчивости $z(t)$ при выпадении 20–40% станций.

#### Ветка B — `Cross-Attention Virtual Magnetometer` (`cross_attn_vm.py`)
- **Без узкого горлышка!**
- **Целевой запрос:** для любой станции $j$ (скрытой или **Zero-Shot**): `q_j(t) = Linear(d_q, 128)(c_j(t)) → LayerNorm → GELU` (`d_q = 6 + d_t`).
- **2 слоя Spatial Cross-Attention:**
  - `CrossAttn_l(Q = q_j^{(l-1)}, K = H_obs, V = H_obs, attn_bias = SpatialBiasMLP(p_{j, \text{obs}}), mask = attn_forbidden_{j,i})` + `PreNorm FFN(128 → 256 → 128)`,
  - где `attn_forbidden[b, q, k] = (~obs_mask[b, k]) | (q_station_id[q] == k_station_id[k])` служит дополнительным защитным барьером (а при Zero-Shot запросах `q_station_id = -1` сводится к `~obs_mask[b, k]`).
  - **КРИТИЧЕСКОЕ ПРАВИЛО ЗАЩИТЫ ОТ MULTI-HOP УТЕЧКИ:** если станция `j` попала в `SpatialSetEncoder` (`obs_mask[b, j] == True`), её значение за 3 слоя Self-Attention переносится в соседние токены `H_obs[b, k]` (`k ≠ j`), и маскирование `k == j` в Cross-Attention **не** предотвращает утечку! Поэтому целевые станции `M_target` **ОБЯЗАНЫ иметь `obs_mask[b, j] == False` ДО входа в `SpatialSetEncoder`**, а лосс считается **СТРОГО по `M_target`** (`w_obs = 0.0`, `M_obs ∩ M_target = ∅`)!
- **Голова регрессии:** `LayerNorm(128) → Linear(128, 64) → GELU → Linear(64, 3)` → $\Delta\hat{B}_{\text{local}}(t, j)$.
- **Число параметров:** `~675K`.
- **Лосс и маски:** взвешенный `HuberLoss(δ=1.0)` строго по `M_target & M_qc_clean` (`w_tgt = 1.0`, `w_obs = 0.0`) на смеси `M1 (30%) + M2 (35%) + M3 (35%)` + **station-dropout augmentation**: в каждом батче случайно исключать 2–3 обучающие станции целиком (`obs_mask = False`, `M_target = True`) как «псевдо-Zero-Shot»!
- **Оптимизатор и бюджет (Фаза 3):** `AdamW(lr=5e-4, weight_decay=1e-4)`, warmup 5 эпох + Cosine decay, `batch_size=64` (или `128` на T4), `max_epochs=100`, `patience=15`, `grad_clip=1.0`.
- **Ориентировочное время:** локальный CPU smoke-test (5 эпох, 1 фолд) `~2–4 мин`; полный прогон 4 фолдов на **Colab/Kaggle T4**: `~6–12 мин` на конфигурацию.

### 6.4. Фаза 4 — Уровень 3 DL: Временной контекст `Shared 1D-CNN → Spatial Transformer`

#### Общий фронтенд `Shared Masked Dilated 1D-CNN` (`temporal_cnn.py`)
- **Вход:** для каждой станции $i \in \{0..N-1\}$ окно `X_win[:, i]` формы `(4, W)` = `[ΔX̃, ΔỸ, ΔZ̃, m_step]`, `W ∈ {32, 64}` мин (по умолчанию `W = 64` мин — вмещает `6–25` периодов `Pc5` и `2–6` периодов `Pc6`).
- **Корректная обработка пропусков (`MaskedConv1d`, ONNX opset-17 native без data-dependent `if`):**
  Для ядра размера `k=3`, дилатации `d` и свёртки **`bias=False`** (на входе первого слоя безопасная очистка `h_clean = torch.where(m > 0.5, torch.nan_to_num(h, nan=0.0), 0.0)`, защищающая от IEEE-754 `0 * NaN == NaN`!):
  $$\text{norm\_sum} = \operatorname{Conv1d}(m,\, \mathbf{1}_k,\, \text{dilation}=d), \qquad m_{\text{out}} = \mathbb{I}[\text{norm\_sum} > 0.5]$$
  $$h_{\text{out}} = \left(\frac{k \cdot \operatorname{Conv1d}(h_{\text{clean}} \odot m,\, W_f,\, d)}{\operatorname{clamp}(\text{norm\_sum},\, \min=1.0)} + b\right) \odot m_{\text{out}}$$
  *(Критично: порог строго `norm_sum > 0.5`, а НЕ `> 0`, т.к. Winograd/cuDNN на нулях может давать `1e-7`, что при делении на `clamp(1e-7, 1e-4)` взрывает шум в `30 000×`; поскольку $m \in \{0, 1\}$, `norm_sum > 0.5` гарантирует `norm_sum >= 1.0`!)*
  - **Режим `causal` (`[t-W+1 .. t]`):** строго левый паддинг `F.pad(..., (d*(k-1), 0))` — нулевая утечка будущего!
  - **Режим `centered` (`[t-W/2 .. t+W/2-1]`):** симметричный паддинг `F.pad(..., (d*(k-1)//2, d*(k-1)//2))`.
- **Стек слоёв (веса общие для всех `N` станций; строго покадровая нормализация!):**
  - **КРИТИЧНО ДЛЯ КАУЗАЛЬНОСТИ:** используется **покадровый `ChannelLayerNorm(64)`** (`transpose(1,2) → LayerNorm(64) → transpose(1,2)` с последующим умножением `* m_out`), а **НЕ `GroupNorm(8, 64)`**! Обычный `GroupNorm` по 3D-тензору `(B*N, 64, W)` усредняет статистику по всей оси времени `W`, нарушая строгую каузальность (протечка будущего `t' > t` в `t`!) и примешивая занулённые пропуски в среднее/дисперсию валидных шагов!
  - Входная проекция `MaskedConv1d(3 → 64, k=3, d=1)` → `ChannelLayerNorm(64) * m → GELU` (RF = `3` мин)
  - Стек остаточных блоков `MaskedDilatedResBlock(64, k=3, d)` (в каждом блоке — **2** слоя `MaskedConv1d(64→64, k=3, d)` + `ChannelLayerNorm(64) * m → GELU → Dropout(0.1)` + skip `(x_skip + conv_out) * m_out`):
    - для `W = 32`: **3 блока** с `d = (1, 2, 4)` (точное рецептивное поле `RF = 3 + 4·(1+2+4) = 31` мин);
    - для `W = 64`: **4 блока** с `d = (1, 2, 4, 8)` (точное рецептивное поле `RF = 3 + 4·(1+2+4+8) = 63` мин).
- **Сборка временного токена $h_i(t)$:**
  - Извлекаются (с маскированным средним `sum(h * m) / clamp(sum(m), min=1.0)`):
    1. состояние в целевой точке `h_target`: индекс `W - 1` для `causal`, индекс `W // 2` для `centered`;
    2. маскированное среднее за 15 мин `h_short`: срез `[W-15 : W]` для `causal`, срез `[W//2 - 7 : W//2 + 8]` для `centered`;
    3. маскированное среднее по всему окну `h_full`.
  - `[h_target, h_short, h_full] ∈ R^{192} → Linear(192, 96) → LayerNorm → GELU → * has_any` (где внутренний гейт `has_any = (m_step.sum(dim=-1, keepdim=True) > 0.5).to(h.dtype)` гарантирует строгий `0.0` для полностью пустой станции с `all m_step == 0`, не пропуская bias `Linear(192, 96)`!) `→ * obs_mask[..., None]`, затем конкатенируется с **мгновенным полем в якоре `ΔB̃_i(t) ∈ R^3`** (прямой residual-скип, гарантирующий, что Фаза 4 является строгим надмножеством Фазы 3!) и координатами `c_i(t) ∈ R^6` → тензор `d_in = 105` подаётся в **тот же самый `SpatialSetEncoder(d_in=105)` из Фазы 3**!
- **Ветка B (приоритет):** `TemporalCrossAttnVM` — сравнение `causal (W=64)` vs `centered (W=64)` (и абляция `W=32`), акцент на авроральную зону и сохранение пульсаций `Pc5/Pc6`. Базовый лосс — `HuberLoss(δ=1.0)` строго по `M_target & M_qc_clean`; опциональная абляция разностного члена (`λ_diff = 0.2 * Huber(Δ_1 y_hat - Δ_1 y)`) запускается только при `lambda_diff > 0` через сэмплер пар последовательных якорей `(t-1, t)` с общей маской (по умолчанию `lambda_diff = 0.0`, чтобы не усложнять без доказанной пользы!).
- **Ветка A (проверка гипотезы с явным критерием Go/No-Go):** `TemporalPerceiverAE` строго в режиме `causal` (`[t-W+1 .. t]`).
  - **Критерий Go/No-Go:** модель Фазы 4 (A) принимается как основная для Ветки A **только если** на 4-fold LOOE-CV выполняется хотя бы одно из условий:
    1. Снижение `RMSE` на маске `M2` составляет `≥ 3.0%` глобально (или `≥ 5.0%` в авроральном поясе) при статистической значимости `p < 0.05` (блочный бутстреп по 6-часовым блокам);
    2. Рост линейного $R^2(z(t) \to \text{OMNI AE/AL})$ составляет `≥ +0.03` **без** падения устойчивости `cos_sim(z_full, z_M2)` более чем на `0.01`.
  - В противном случае фиксируется **No-Go** (научно значимый вывод для диссертации!) и в Фазу 5 по Ветке A идёт одноточечный `Perceiver-Bottleneck AE` (Фаза 3).
- **Число параметров:** `~806K` (`TemporalCrossAttnVM`, Ветка B) и `~917K` (`TemporalPerceiverAE`, Ветка A).
- **Бюджет и время:** `batch_size=32..64`, `lr=3e-4`, `max_epochs=60`, `patience=12`. Локальный CPU smoke-test (2 эпохи, 1 фолд, 10% шагов) `~2–4 мин`; полный 4-fold прогон на **Colab/Kaggle T4**: `~15–25 мин`.

---

## 7. Пошаговый план атомарных задач (`T0.1` – `T6.3`)

Каждый последующий ИИ-агент берёт **одну конкретную задачу** (проверив выполнение её зависимостей), реализует строго указанные файлы, прогоняет обязательные тесты `pytest` и закрывает **Exit Gate**.

---

### ФАЗА 0. Среда и единый ML-датасет (общая)

#### `T0.1` — Изолированное окружение `.venv_ml` и каркас пакета `ml_research/`
- **Зависимости:** нет.
- **Создаваемые файлы:**
  - `ml_research/.gitignore`, `ml_research/pyproject.toml`, `ml_research/pytest.ini`, `ml_research/README.md`
  - `ml_research/requirements_local_cpu.txt`, `ml_research/requirements_colab_gpu.txt`
  - `ml_research/geomag_ml/__init__.py`, `ml_research/geomag_ml/core_bridge.py`
  - `ml_research/geomag_ml/utils/{__init__.py, seed.py, io.py, device.py}`
  - `ml_research/tests/conftest.py`
- **Требования к реализации:**
  1. Создать `.venv_ml` на базе Python 3.12 (`py -3.12 -m venv .venv_ml`).
  2. Установить `torch` (CPU wheel), `numpy`, `scipy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `pyyaml`, `pytest`, `onnx`, `onnxruntime` (опционально `hmmlearn` для `T5.1`, с обязательным фолбэком на `scikit-learn` `GaussianMixture` + марковское сглаживание); установить `pip install -e ml_research`.
  3. `core_bridge.py` подключает `<repo_root>/backend` в `sys.path` только на чтение и предоставляет `import_mlt()`, `import_conventions()`, `import_spectral()`, `import_polarization()`, `import_analytic()` (для `find_valid_subsegments` и `run_variant_b_pipeline` в `T5.2`/`T5.3`), `import_cross_wavelet()`, `import_etl_parsers()` и `import_ballistic_shift()` (для `geomag_api.data_store.compute_ballistic_shift` и `_resample_monotonic_series` в `T0.2`; `geomag_api.__init__` ленивый и не требует `fastapi`) с понятной ошибкой, если `backend/` недоступен (например, в облаке).
  4. `device.py` при `cpu` выставляет `torch.set_num_threads(min(6, os.cpu_count() // 2))` для избежания деградации на 12-поточном Ryzen.
  5. Не трогать `.venv/`, `backend/`, `frontend/`, `tests/`, корневой `pytest.ini`.
- **Обязательные автотесты (`pytest`):** проверка импорта `geomag_ml`, воспроизводимости `set_seed(42)`, чтения `geomag_core.mlt.geo_to_magnetic_dipole` через `core_bridge`, проверки что `torch.cuda.is_available()` корректно обрабатывается с фолбэком на `cpu`.
- **Exit Gate:**
  - Команда: `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests -k "seed or bridge"` → `0 failed`.
  - Команда проверки изоляции: `git status --short` не показывает изменений в `backend/`, `frontend/`, `tests/`, `data/`.

#### `T0.2` — Сборщик единого ML-датасета (`clean` версия октября 2026)
- **Зависимости:** `T0.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/data/{__init__.py, schema.py, build_dataset.py}`
  - `ml_research/configs/dataset_oct2026.yaml`
  - `ml_research/scripts/build_dataset.py`
  - `ml_research/tests/test_dataset_contract.py`
- **Требования к реализации:**
  1. Читать read-only `data/processed/catalog.sqlite` и все 7 `.parquet` файлов (`geomag/`, `solar_wind/`, `cosmic_rays/`).
  2. Собрать плотные массивы строго по контракту §5.1 (`T=14400`, `N=49` магнитометров в алфавитном порядке `ABK..YKC`), вычислить `MLat/MLon/MLT` строго через `core_bridge` (`geomag_core.mlt`), расстояния `station_dist_km` через `haversine_km`, и баллистически сдвинутый `sw_solar1_bsn` (1.4 млн км, окно 31 мин, строгая монотонность).
  3. Записать `M_valid`, `M_raw_valid`, `M_interpolated`, `M_despiked` и `M_qc_clean` (исключая из `M_qc_clean` подтверждённые многоминутные приборные аномалии `VAL 2026-10-09 09:19..10:42`, `OTT 2026-10-22 13:45..14:15`, `MEA 2026-10-23 17:54..18:43`, `IZN 2026-10-23 07:10..07:13`, `IZN 2026-10-30 07:16..07:19 UTC` и при `variation_step_policy="flag_qc_exclude"` эпизоды скачка базы `LON@storm_20261030_31`, `SFS@storm_20261009_11`, заданные декларативно в `dataset_oct2026.yaml`).
  4. Сохранить `ml_research/data_cache/geomag_ml_oct2026_clean_v1.npz` и `.meta.json` с SHA-256.
- **Обязательные автотесты (`pytest`):**
  - Точные формы `(14400, 49, 3)`, `(14400, 49)`, `(49, 49)`, dtypes, `X_raw` конечен везде где `M_valid == True` и `NaN` везде где `~M_valid`.
  - `station_has_baseline` = `False` ровно для `GCK` и `NCK` (`sum == 47`).
  - Пояса: `EU_mid=31, EU_aur=6, CA_mid=5, CA_aur=7`.
  - `mlt_hours ∈ [0, 24)`, `sin² + cos² = 1 ± 1e-5` для `UT`, `DOY`, `MLT`.
  - `station_dist_km` симметрична, диагональ `0`, `dist(ABK, KIR) ∈ [90, 93] км`.
  - `OMNI` и знак $E_y$: `ae_nT` валиден ровно в `7200` строках (эп. 0–1), `sym_h` — `0` строк (обрабатывается без падений!); положительная корреляция `corr(sw_solar1_bsn[:, ey_mVm], sw_omni[:, electric_field]) > 0.75` (факт на Oct-2026: `r = +0.870` по `3 847` общим минутам — защита от инверсии знака `vx_gse * bz_gsm`!).
- **Exit Gate:**
  - Команда: `.venv_ml\Scripts\python.exe ml_research/scripts/build_dataset.py --config ml_research/configs/dataset_oct2026.yaml`
  - Команда: `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_dataset_contract.py` → `100% pass`.
  - Артефакт: `ml_research/data_cache/geomag_ml_oct2026_clean_v1.npz` размером `< 25 МБ` + `.meta.json`.

#### `T0.3` — Сборщик недеспайкнутой тестовой версии (со спайками `VAL`, `OTT`, `MEA`, `IZN`)
- **Зависимости:** `T0.2`.
- **Обновляемые/создаваемые файлы:**
  - `ml_research/geomag_ml/data/build_dataset.py` (добавить `build_undespiked_dataset`)
  - `ml_research/scripts/build_dataset.py` (флаг `--also-undespiked`)
  - `ml_research/tests/test_dataset_contract.py` (расширить тестами)
- **Требования к реализации:**
  1. Прочитать сырые `.min` файлы из `GeoMag_Storms_Dataset_Oct-2026/GeoMag Storms/INTERMAGNET_1min/` **строго read-only** через `geomag_etl.intermagnet.parse_iaga2002_file` (и `interpolate_short_gaps(..., max_gap=3)` без `despike_station_network`), НЕ записывая ничего в `data/processed/`.
  2. Собрать `ml_research/data_cache/geomag_ml_oct2026_undespiked_v1.npz` на абсолютно той же сетке `(14400, 49, 3)`.
  3. Вычислить точный дифф `|X_undespiked - X_clean| > 1e-3` и подтвердить, что ровно 10 одноминутных точек (`VAL` 3, `OTT` 3, `MEA` 3, `IZN` 1) отличаются (с пиком `VAL 2026-10-09 10:39Z |ΔZ| > 1200 нТл` и `OTT 2026-10-22 13:48Z |ΔZ| > 350 нТл`), сохранить их объединение с 5 многоминутными приборными окнами (`VAL` 84м, `OTT` 31м, `MEA` 50м, `IZN` 4м+4м) в `M_known_spikes` (ровно **174** станция-минуты).
- **Обязательные автотесты (`pytest`):** равенство `X_undespiked` и `X_clean` везде вне `M_despiked`, ненулевые амплитуды на `VAL` и `OTT`, `int(M_known_spikes.sum()) == 174`.
- **Exit Gate:**
  - `.venv_ml\Scripts\python.exe ml_research/scripts/build_dataset.py --config ml_research/configs/dataset_oct2026.yaml --also-undespiked`
  - Артефакт: `ml_research/data_cache/geomag_ml_oct2026_undespiked_v1.npz` + `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_dataset_contract.py` → `100% pass`.

#### `T0.4` — Базовая линия $B_0$ и двойной скейлер (`z-score` / `asinh`) без утечек
- **Зависимости:** `T0.2`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/data/preprocessing.py`
  - `ml_research/tests/test_preprocessing_and_splits.py` (часть 1)
- **Требования к реализации:**
  1. `BaselineEstimator(mode, baseless_policy, variation_step_policy="flag_qc_exclude", step_warn_nT=30.0)` с методами `fit(dataset, train_idx)` и `transform(X_raw, M_valid, episode_idx) -> X_delta_nT`. Поддержка всех 3 режимов из §5.2 (`temp_train_quiet_proxy` с вычислением $\Delta_1$ строго внутри эпизодов, `target_kp_quiet_night`, `per_episode_median_ablation`), плюс корректное разделение статической обсерваторской базы $B_0$ для Zero-Shot (`LVV, BEL, MEA` по нетестовым эпизодам, `SUA` в Fold 3 по каузальным первым 6 ч).
  2. `FieldScaler(mode="asinh_global" | "zscore_per_station", min_std_nT=5.0, min_asinh_s_nT=10.0)` с `fit(X_delta, M_qc_clean, train_idx, station_belt_id, station_train_mask=None)`, `transform(X_delta)`, `inverse_transform(X_scaled)`, `to_dict()`, `from_dict()`.
  3. Явная обработка станций с `0` обучающих точек (`SUA` в Fold 3, Zero-Shot станции `LVV, BEL, MEA`): `in_train[i] == False`, фолбэк на поясной масштаб, никаких `NaN`/`0.0` в `std`!
- **Обязательные автотесты (`pytest`):**
  - Тест на **отсутствие утечки**: изменение всех значений в `test_idx` на `+5000 нТл` НЕ меняет параметры `B0`, `mean`, `std`, `scale_s` ни на `1e-7`!
  - Обратимость `inverse_transform(transform(X)) == X` с ошибкой `< 1e-4 нТл` для `zscore` и `asinh`.
  - Обработка станции с `0` валидных точек в `train` (`SUA` кейс) без `NaN`/`Inf`.
  - Сохранение/загрузка `scaler.json` даёт побитово идентичный `transform`.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_preprocessing_and_splits.py -k "baseline or scaler"` → `100% pass`.

#### `T0.5` — Адаптер расширяемости под будущую догрузку данных соруководителя
- **Зависимости:** `T0.2`, `T0.4`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/data/ingest_extended.py`
  - `ml_research/configs/dataset_extended_template.yaml`
  - `ml_research/configs/events_template.yaml`
  - Дополнение `ml_research/tests/test_dataset_contract.py`
- **Требования к реализации:**
  1. Парсинг внешних папок `ml_research/data_external/`: дополнительные `.min` файлы (новые даты, спокойные дни `Kp ≤ 2`, новые станции `N > 49`), обновлённый `OMNI_1min` (`.asc`/`.parquet` с `SYM-H`, `PC(N)`, `AE/AL/AU` — **соединять строго по `timestamp_utc`, игнорируя захардкоженный `episode_id` из `parse_omni_hro_file`!**), опциональный `Kp` CSV (3-часовой → ступенчатая привязка к 1-мин UTC), и `events.yaml` с диапазонами `quiet_kp_le2`, `storm`, `eclipse`, `spe` (для `eclipse` поддерживается как статический центр `(lat, lon, radius_km)`, так и движущаяся трасса `track: [{timestamp_utc, lat_deg, lon_deg, radius_km}]` с линейной интерполяцией во времени).
  2. Автоматическая группировка непрерывных суток в эпизоды (разрыв `> 60 мин` = новый эпизод), пересчёт `MLat/MLT/dist_km`, автоматическое выставление `recommended_use_doy = (span_days >= 60)` и `event_flags`.
  3. Слияние с октябрём 2026 в единый `geomag_ml_<ext_tag>.npz` без модификации `data/processed/`.
- **Обязательные автотесты (`pytest`):** синтетический тест догрузки 2 спокойных дней (`Kp=1`), 1 новой станции (`NEW`), заполненного `SYM-H` на новых датах (проверяет join по `timestamp_utc`!) и 1 интервала `eclipse` — проверка корректной сборки `.npz`, флагов `event_flags`, и работы `BaselineEstimator(mode="target_kp_quiet_night")`.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_dataset_contract.py -k "extended"` → `100% pass`.

---

### ФАЗА 1. Протокол валидации, генератор масок, метрики и линейные бейзлайны (общая)

#### `T1.1` — Строгий протокол сплитов без временных утечек и Window Boundary Guard
- **Зависимости:** `T0.2`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/data/splits.py`
  - Дополнение `ml_research/tests/test_preprocessing_and_splits.py`
- **Требования к реализации:**
  1. Режим **`looe_4fold`** (4-fold Leave-One-Episode-Out для октября 2026) с QC-верифицированным детерминированным `inner_val` эпизодом, гарантирующим **`OMNI > 0` в `train` каждого фолда** и **`0` необученных станций в `test` одновременно под `M_valid` и под `M_qc_clean`** (кроме структурно одноэпизодной `SUA` в Fold 3; исключается конфигурация `train=[ep1, ep3]`, где при `variation_step_policy="flag_qc_exclude"` станция `SFS@ep1` выбывает из `M_qc_clean` и оставалась бы с `0` обучающими строками):
     - Fold 0: `test=ep0`, `val=ep3`, `train=[ep1, ep2]` (`8 640 мин`, `OMNI=4 320 мин`, `0` пропавших из `train` в `test` под `M_valid` и `M_qc_clean`: у `SFS` в `ep2` `885` чистых / `224` ночных MLT / `68` тихих ночных мин; `SUA` в `val=ep3` отсекается флагом `in_train`)
     - Fold 1: `test=ep1`, `val=ep3`, `train=[ep0, ep2]` (`7 200 мин`, `OMNI=2 880 мин`, `0` пропавших из `train` в `test` под `M_valid` и `M_qc_clean`; `SUA` в `val=ep3` отсекается флагом `in_train`)
     - Fold 2: `test=ep2`, `val=ep1`, `train=[ep0, ep3]` (`5 760 мин`, `OMNI=2 880 мин`, `0` пропавших из `train` одновременно в `val` и `test` под `M_valid` и `M_qc_clean`: и `SFS`, и `LON` имеют по `2 880` чистых мин в `ep0`)
     - Fold 3: `test=ep3`, `val=ep0` (47 ст.), `train=[ep1, ep2]` (`8 640 мин`, `OMNI=4 320 мин`, `0` пропавших в `val` под `M_valid` и `M_qc_clean`, только структурно одноэпизодная `SUA` с `in_train=False` в `test`)
  2. Режим **`block_temporal`** (для расширенных данных): блочное разбиение `train/val/test` с обязательным буферным зазором `purge_gap_min >= 1440` (`≥ 1 сут.`) между любыми смежными сплитами; все минуты с флагами `eclipse` или `spe` жёстко прибиваются в `test` (с `AssertionError` при попытке отправить их в `train`).
  3. **Запрет случайного сплита:** явная проверка минимальной дистанции `min |t_train - t_test| >= 1440 мин` (для текущих эпизодов разрывы между бурями ≥ 3 сут. — выполняется автоматически!).
  4. Поддержка списка **`zero_shot_stations`** (по умолчанию `["LVV", "BEL", "MEA"]`): возвращает `station_train_mask: bool[N]`.
  5. Функция **`build_valid_window_anchors(timestamps_unix_s, episode_idx, split_indices, W, mode="causal"|"centered")`**: возвращает только те якорные индексы $t$, чьи окна `[t-W+1..t]` или `[t-W//2..t+W//2-1]` на 100% лежат внутри **одного эпизода**, **одного сплита** и имеют непрерывный шаг `60 с`.
- **Обязательные автотесты (`pytest`):**
  - `min_gap_sec >= 86400` между `train` и `test` во всех фолдах и в `block_temporal`.
  - В `looe_4fold` во всех 4 фолдах `OMNI_train_valid > 0` и единственная тестовая станция с `0` обучающих строк — `SUA` в Fold 3, причём проверка выполняется **одновременно по `M_valid` и по `M_qc_clean`** (для всех остальных тестовых станций, включая `SFS` и `LON`, число `M_qc_clean`-строк в `train` строго `> 0`, а ночных MLT-минут `>= 60`).
  - Все `eclipse`/`spe` точки строго в `test` (`0` в `train`/`val`).
  - Ни одно окно из `build_valid_window_anchors` не пересекает границу эпизода или границу `train/val/test`!
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_preprocessing_and_splits.py` → `100% pass`.

#### `T1.2` — Единый генератор масок (`M1`, `M2`, `M3`, `M4`)
- **Зависимости:** `T0.2`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/data/masks.py`
  - `ml_research/tests/test_masks_and_metrics.py` (часть 1)
- **Требования к реализации:**
  1. Класс `MaskGenerator(station_coords, station_dist_km, seed=42)` с детерминированными методами для оценки (`eval_masks`) и стохастическими для обучения (`sample_train_batch_masks`).
  2. Все 4 режима оценки:
     - `M1`: полный детерминированный перебор всех валидных станций по одной (`N` проходов или стратифицированный батч).
     - `M2(drop_frac=0.25, n_repeats=5)`: случайное скрытие 20–30% (и 40% для теста стабильности Ветки A) валидных станций при сохранении `min_obs >= 10`.
     - `M3(radius_km=500 | 1000)`: скрытие регионального кластера вокруг якорной станции $c$ в радиусе `R`. Для учёта разреженности Канады (`d_NN = 626..1330 км`): в `M3_500km` якоря выбираются из станций с `>= 2` валидными станциями в круге `500 км` (европейские кластеры), в `M3_1000km` — с `>= 2` станциями в `1000 км` (Европа + Канада); в `train`-смеси при `< 2` станций в круге включается фолбэк на `k=3` ближайших соседа (с верхним порогом `<= 35%` сети).
     - `M4`: скрытие всех станций с `MLat >= 60°` (13 авроральных), наблюдение только `MLat < 60°`.
  3. **Обязательный `in_train`-барьер (разделение стандартных `M1–M4` и `zero_shot`):**
     - В стандартных режимах оценки `M1, M2, M3_500km, M3_1000km, M4` целевая маска `M_target` (`eval_mask`) и маска наблюдений `M_obs` строго ограничиваются станциями `in_train`: `M_target <= (M_qc_clean & in_train)` и `M_obs <= (valid_mask & in_train)` (где `valid_mask = M_qc_clean` в чистом бенчмарке и `M_valid` в `T5.4`). Это гарантирует, что не-Zero-Shot модели (`PCA`, `Ridge`, `MLP`) никогда не оцениваются на необученных весах `SUA` (Fold 3 / `val=ep3` в Folds 0–1) и все модели Фаз 1–4 сравниваются на строго одном и том же множестве `in_train`-станций.
     - В режиме `zero_shot` (`unseen_stations`) для координатных моделей (`IDW`, `Ordinary Kriging`, Spatial/Temporal Transformers Фаз 3–4): целевые станции `~in_train` делятся на две раздельные подмаски — **`heldout_zero_shot = M_qc_clean & is_zero_shot`** (`LVV, BEL, MEA` в Zero-Shot прогонах с базой из `train`) и **`single_episode_unseen = M_qc_clean & (~in_train) & (~is_zero_shot)`** (одноэпизодная `SUA` в Fold 3 с каузальной базой первых 6 ч `ep3`), при этом станции с `~in_train` **никогда** не попадают в `obs_mask`: `M_obs = valid_mask & in_train`, `(M_obs & ~in_train).sum() == 0`.
  4. Инварианты (проверяются assert'ами): `M_obs <= (valid_mask & in_train)`, `M_target <= M_qc_clean`, `(M_obs & M_target).sum() == 0`, `M_obs.sum(axis=-1) >= 1`, а также `M_target <= in_train` для стандартных `M1–M4` и `M_target <= (~in_train)` для `zero_shot`.
- **Обязательные автотесты (`pytest`):** дизъюнктность `M_obs ∩ M_target = ∅`, отсутствие `M_obs` на пропущенных (`~valid_mask`) и не-обучаемых (`~in_train`) станциях, строгий `in_train`-фильтр для `M_target` в `M1–M4` и строгое `~in_train` (с разделением `heldout_zero_shot` и `single_episode_unseen`) для `M_target` в `zero_shot`, детерминизм по `seed`, корректное число скрытых станций в `M1` (=1), `M2` (20–30%), `M3` (кластер $\ge 2$ ст.), `M4` (все `MLat >= 60°` среди `in_train` в `M_target`, все `M_obs` имеют `MLat < 60°`).
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_masks_and_metrics.py -k "mask"` → `100% pass`.

#### `T1.3` — Единый модуль метрик и evaluator
- **Зависимости:** `T1.1`, `T1.2`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/eval/{__init__.py, metrics.py, evaluator.py}`
  - Дополнение `ml_research/tests/test_masks_and_metrics.py`
- **Требования к реализации:**
  1. `compute_reconstruction_metrics(y_true_nT, y_pred_nT, eval_mask, station_belt_id, station_regions)` → точный словарь по поясам (`global`, `europe_mid`, `auroral`, `europe_auroral`, `canada_auroral`, `canada`, `canada_mid`) и компонентам (`X, Y, Z, vector_3d`): `rmse_nT`, `mae_nT`, `r2`, `pearson_r`, `n_points`.
  2. `evaluator.py` жёстко применяет **`in_train`-барьер** из `T1.2`:
     - во всех стандартных режимах `M1, M2, M3_500km, M3_1000km, M4` в `compute_reconstruction_metrics` передаётся строго `eval_mask = M_target & M_qc_clean & in_train` (необученные станции вроде `SUA` в Fold 3 гарантированно не попадают в метрики `M1–M4` ни у `PCA`/`Ridge`/`MLP`, ни у трансформеров — все модели сопоставимы на одном наборе станций);
     - станции с `~in_train` оцениваются **исключительно** в корзине `metrics["eval_masks"]["zero_shot"]` (при гарантированном `(obs_mask & ~in_train).sum() == 0`) только для координатных моделей (`IDW`, `Ordinary Kriging`, Фазы 3–4) с **раздельными ключами** `heldout_zero_shot` (`is_zero_shot=True`: `LVV, BEL, MEA` — основная 4-фолдовая Zero-Shot метрика), `single_episode_unseen` (`~in_train & ~is_zero_shot`: `SUA` в Fold 3 с явной пометкой `baseline_note="causal_initial_6h"`) и `per_station`; для моделей с фиксированным множеством станций (`PCA`, `Ridge`, `MLP`) корзина `zero_shot` записывается как `{"status": "not_supported_fixed_stations"}`.
  3. `compute_latent_stability(z_full, z_masked_list)` → `cos_sim_mean`, `rel_l2_drift`, `axis_corr_mean`, `axis_decorr_frob`.
  4. `compute_space_weather_alignment(z_non_test, z_test, sw_omni, sw_solar1_bsn, oulu, non_test_idx, test_idx)` → корреляции Пирсона/Спирмена (NaN-safe, различая `{"status": "unavailable_all_nan"}` для полностью пустых `sym_h`/`pc_n`/`kp` и **`{"status": "unavailable_in_test_fold"}`** для `OMNI AE/AL/AU/Bz/Pdyn/Ey` в Folds 2–3, где `test_idx` не имеет покрытия `OMNI`!) + линейная/Ridge регрессия `z → target`, обучаемая на нетестовых индексах с покрытием (`train ∪ val`) и оцениваемая строго на `test_idx` без утечки!
- **Обязательные автотесты (`pytest`):** точные аналитические значения `RMSE=0, R²=1` на идеальном прогнозе, корректная агрегация по поясам, проверка `in_train`-барьера (станция с `in_train == False`, имитирующая `SUA`, не влияет на `M1–M4` метрики даже при прогнозе `1e6`/`NaN`, не смешивается с `heldout_zero_shot` `LVV/BEL/MEA` и не попадает в `obs_mask`), NaN-safe обработка как полностью пустых столбцов `OMNI` (`unavailable_all_nan`), так и фолдов без `OMNI` в `test_idx` (`unavailable_in_test_fold`), инвариантность `cos_sim = 1.0` при `z_masked == z_full`.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_masks_and_metrics.py` → `100% pass`.

#### `T1.4` — Линейные бейзлайны Фазы 1 (`PCA ± Sq`, `Ridge-Schur`, `IDW`, `Ordinary Kriging`)
- **Зависимости:** `T0.4`, `T1.1`, `T1.2`, `T1.3`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/{__init__.py, base.py, baselines.py}`
  - `ml_research/configs/phase1_baselines.yaml`
  - `ml_research/scripts/run_baselines.py`
  - `ml_research/tests/test_baselines.py`
- **Требования к реализации:**
  1. Ветка A: `EMPCABaseline(k, use_sq_detrend, scaling)` (`k ∈ {3, 4, 5}`, с/без $S_q(\text{UT})$ с масштабированной регуляризацией $\lambda = 0.01 \cdot n_{\text{obs}}$ на гармоники $m \ge 1$ и адаптивным понижением $M_h = 1$ при охвате `< 20/24` часовых бинов UT — защита `SFS` в Fold 0!, с обязательным центрированием по `μ_train` на `M_qc_clean & in_train`) с маскированной проекцией МНК на `M1–M4` и оценкой стабильности `z(t)` под `M2` + корреляций с `OMNI`/`SOLAR-1`/`OULU`.
  2. Ветка B: `RidgeSchurBaseline(alpha_grid)` с **EM-Gaussian ковариацией** (устойчивой к парам станций с `0` совместно валидных минут!) для `M1–M4`; `IDWBaseline` и `OrdinaryKrigingBaseline` (с `wrap_180(ΔMLon)` в `nT`/`asinh_global` и минимальным nugget-полом `σ_n / σ_s >= 1e-2`) для `M1–M4` **и** Zero-Shot (`LVV`, `BEL`, `MEA`).
  3. Полный 4-fold LOOE-CV прогон и сохранение артефактов в `ml_research/artifacts/p1_*` по контракту §5.4 + сводная таблица `ml_research/artifacts/phase1_summary.csv`.
- **Обязательные автотесты (`pytest`):**
  - `EMPCA` монотонно снижает `RMSE` при росте `k: 3 → 4 → 5`, корректно центрирует ненулевое среднее `μ ≠ 0`; маскированный вход `1e6` на скрытой станции не влияет на `z_hat` (`diff == 0`); `Sq-detrend` на станции с неполным охватом `15/24` ч UT (имитация `SFS@ep2` в Fold 0) не экстраполирует взрывно (`max|Sq(24h)| < 2 × max|y_train|`).
  - `Ridge-Schur` не имеет `NaN/Inf` **даже при наличии пар станций с `0` совместных наблюдений в `train`** (имитация `DOU↔DUR` в `train=[ep1, ep2]` или `CLF↔NCK` в `train=[ep0, ep3]`) и превосходит нулевой прогноз на синтетических коррелированных полях.
  - `IDW` и `Kriging` работают на станциях с `in_train == False` (Zero-Shot), корректно обрабатывают переход долготы `±180°` и дают точный `0` ошибки на постоянном поле.
- **Exit Gate:**
  - `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_baselines.py` → `100% pass`.
  - `.venv_ml\Scripts\python.exe ml_research/scripts/run_baselines.py --config ml_research/configs/phase1_baselines.yaml` → генерирует `phase1_summary.csv` и все `artifacts/p1_*/metrics.json`; все 4 фолда имеют конечные `RMSE [нТл]` и `R² > 0` на `M1` и `M2` по `europe_mid` и `auroral`.

---

### ФАЗА 2. Уровень 1 DL — Одноточечные Masked MLP (обе ветки)

#### `T2.1` — Общий `Dataset`, лоссы, единый `Trainer` и менеджер артефактов
- **Зависимости:** `T0.4`, `T1.1`, `T1.2`, `T1.3`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/data/datasets.py` (`SingleStepDataset`)
  - `ml_research/geomag_ml/train/{__init__.py, losses.py, trainer.py, artifacts.py}`
  - `ml_research/scripts/train.py`, `ml_research/scripts/evaluate.py`
- **Требования к реализации:**
  1. `losses.py`: `masked_weighted_huber_loss`, `beta_vae_kl_loss`, `covariance_decorr_loss` (строго возвращает `0.0` при `B < 8` без деления на 0 и предупреждений `DoF <= 0`!), `latent_consistency_loss`. Все потери защищены от пустой маски (`denom.clamp_min(1.0)`) и проверяются на `torch.isfinite`.
  2. `trainer.py`: детерминированный цикл `train/val`, `AdamW`, `CosineAnnealingLR` с warmup, `clip_grad_norm_(1.0)`, опциональный `torch.amp.autocast("cuda")` (в `float32` на CPU и внутри критичных лоссов), ранний останов по `val_masked_rmse_nT`, запись `history.csv`, `model.pt`, `config.json`, `scaler.json`, `stations.json`, `metrics.json`, `onnx_export_spec.json`.
  3. CLI `train.py` поддерживает `--config`, `--dataset` (опционально переопределяет путь в YAML), `--fold (0|1|2|3|all)`, `--ablation (base|all|<name>)`, `--seed`, `--device (auto|cpu|cuda)`, `--out-dir` (по умолчанию `ml_research/artifacts`), `--smoke` (быстрый 3-эпохный прогон для локальной проверки).
- **Обязательные автотесты (`pytest`):** переобучение мини-модели до `loss < 0.05` на 32 синтетических точках, отсутствие `NaN/Inf` в лоссе и градиентах (вкл. `covariance_decorr_loss` при `B = 2 < 8`), воспроизводимость двух запусков с одинаковым `seed=42` (`max|w1 - w2| < 1e-6`).
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase2_mlps.py -k "trainer or loss or seed"` → `100% pass`.

#### `T2.2` — Ветка A: `Masked Conditional MLP-AE` / `β-VAE` и обязательные абляции
- **Зависимости:** `T2.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/mlp_ae.py`
  - `ml_research/configs/phase2_branch_a_mlp.yaml`
  - `ml_research/tests/test_phase2_mlps.py` (часть Ветки A)
- **Требования к реализации:**
  1. Реализовать `MaskedConditionalMLPAE` строго по §6.2: вход `(B, 4*N)` + `c_time: Tensor | None = None` (`(B, d_t)` при `d_t > 0`, и строго `None` / без нулевого тензора при `d_t == 0` для чистого экспорта ONNX!), узкое горлышко `z ∈ R^k (k ∈ {3, 4, 5})`, выход `(B, 3*N)`.
  2. Поддержать флагами в YAML всю матрицу абляций: `k ∈ {3, 4, 5}`, `use_ut_cond ∈ {True, False}`, `scaler_mode ∈ {zscore_per_station, asinh_global}`, `bottleneck_mode ∈ {ae, ae_decorr, beta_vae}`, `train_mask_mode ∈ {m1_m2, none}`.
  3. Провести полный 4-fold LOOE-CV локально на CPU для лучшего базового конфига и ключевых абляций.
- **Обязательные автотесты (`pytest`):**
  - Формы тензоров `(B, 4*N) → (B, 3*N)` и `z: (B, k)`.
  - **Тест отсутствия утечки маски:** при обнулении входа через `m_obs` изменение скрытой станции в исходном массиве на `+1e6` даёт `diff == 0.0` в `z` и `y_hat`.
  - Отсутствие `NaN/Inf` в forward/backward для всех 3 режимов (`ae`, `ae_decorr`, `beta_vae`).
  - Корректный `eval()` детерминизм для `β-VAE` (`z = μ`, два прогона дают одинаковый результат).
- **Exit Gate:**
  - `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase2_mlps.py -k "branch_a"` → `100% pass`.
  - Локальный прогон 4-fold LOOE-CV (`.venv_ml\Scripts\python.exe ml_research/scripts/train.py --config ml_research/configs/phase2_branch_a_mlp.yaml --fold all`) завершается, все абляции зафиксированы в `artifacts/p2_A_*`, лучший `MLP-AE` бьёт или сопоставим с `PCA` того же `k` на `M2` по `RMSE [нТл]` и стабильности `z(t)`.

#### `T2.3` — Ветка B: `Masked Wide MLP` (единая сеть на всю сеть) и оценка «цены сжатия»
- **Зависимости:** `T2.1`, `T2.2`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/mlp_vm.py`
  - `ml_research/configs/phase2_branch_b_mlp.yaml`
  - `ml_research/scripts/check_phase2_gate.py`
  - Дополнение `ml_research/tests/test_phase2_mlps.py` (часть Ветки B)
- **Требования к реализации:**
  1. Реализовать `MaskedWideMLP` строго по §6.2: единая сеть `(B, 4*N) → 512 × 3 ResBlocks → (B, 3*N)`, `HuberLoss` с `w_tgt = 4.0, w_obs = 0.25`, смесь масок `M1+M2+M3`.
  2. Скрипт `ml_research/scripts/check_phase2_gate.py` автоматически сравнивает артефакты Фаз 1–2 и сохраняет в `artifacts/phase2_gate_report.json`:
     - `Wide MLP (B)` vs `Ridge-Schur (B)` на `M1, M2, M3, M4` по поясам;
     - `MLP-AE (A)` vs `PCA (A)` на `k=3, 4, 5`;
     - точную **«цену сжатия»** `ΔRMSE = RMSE(MLP-AE_k) - RMSE(Wide MLP)` по поясам и маскам.
- **Обязательные автотесты (`pytest`):** формы, zero-leakage маски, finite grad/loss, seed parity, наличие ровно одной модели (не 49 экземпляров).
- **Exit Gate:**
  - `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase2_mlps.py` → `100% pass`.
  - Локальный 4-fold прогон `phase2_branch_b_mlp.yaml` + `.venv_ml\Scripts\python.exe ml_research/scripts/check_phase2_gate.py` завершены; `Wide MLP (B)` статистически превосходит или сопоставима с `Ridge-Schur` на нелинейных масках `M2/M3/M4` (особенно в `auroral`), и сформирован `artifacts/phase2_gate_report.json`.

---

### ФАЗА 3. Уровень 2 DL — Одноточечные Spatial Set Transformers (обе ветки)

#### `T3.1` — Общий пространственный энкодер `SpatialSetEncoder` + `SpatialBiasMLP`
- **Зависимости:** `T2.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/spatial_encoder.py`
  - `ml_research/tests/test_phase3_transformers.py` (часть энкодера)
- **Требования к реализации:**
  1. Реализовать `compute_pair_geometry(station_coords_q: Tensor[N_q, 4], station_coords_k: Tensor[N_k, 4]) -> Tensor[N_q, N_k, 5]` (вход строго `[lat_deg, lon_deg, mlat_deg, mlon_deg]`, выход строго по §6.3: `[d_km/2000, log1p(d_km/500), (MLat_q - MLat_k)/30, sin(ΔMLon), cos(ΔMLon)]`), работающую для произвольных `N_q, N_k` (включая Zero-Shot!).
  2. Реализовать `SpatialBiasMLP(in_dim=5, hidden=16, n_heads=4)` и `SpatialSetEncoder(d_in=9, d_model=128, n_layers=3, n_heads=4, d_ff=256, drop=0.1, use_spatial_bias=True, use_station_emb=False)` с конфигурируемым `d_in` (для переиспользования в Фазе 4 и `T5.2`).
  3. Строгое маскирование и FP16/AMP-безопасность: безопасное обнуление `obs_tokens = torch.where(obs_mask[..., None], torch.nan_to_num(obs_tokens, nan=0.0), 0.0)` до проекции (защита от IEEE-754 `0 * NaN == NaN`) + маскированный `softmax` в `float32` (без `-1e9` в `fp16`!) с защитой `torch.where(all_forbidden, 0.0, attn)` + обнуление выходного `H_obs` по `obs_mask`.
- **Обязательные автотесты (`pytest`):**
  - **Перестановочная эквивариантность (при `use_station_emb=False`):** перестановка порядка станций в батче переставляет `H_obs` ровно тем же образом (`max_diff < 1e-5`)!
  - **Adversarial leakage & NaN-at-masked test:** замена значений на маскированных станциях (`~obs_mask`) на `+1e6` **или `NaN`** меняет `H_obs` на наблюдаемых станциях строго на `0.0` (`< 1e-6`) без единого `NaN` в выходе и градиентах!
  - **FP16/all-masked safety test:** прогон в `float16`/`autocast` и при пустой строке маски не даёт `RuntimeError` и `NaN`.
  - Переменное число станций: энкодер без переинициализации принимает `N=49`, `N=35` и `N=60` (будущие станции!).
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase3_transformers.py -k "encoder"` → `100% pass`.

#### `T3.2` — Ветка A: `Perceiver-Bottleneck AE` + Coordinate Query Decoder
- **Зависимости:** `T3.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/perceiver_ae.py`
  - `ml_research/configs/phase3_branch_a_perceiver.yaml`
  - Дополнение `ml_research/tests/test_phase3_transformers.py`
- **Требования к реализации:**
  1. Реализовать `PerceiverBottleneckAE(d_in=9, d_out=3, ...)` строго по §6.3 с **конфигурируемыми `d_in` и `d_out`** (`d_in=9, d_out=3` в Фазе 3; `d_in=105, d_out=3` в Фазе 4; `d_in=18, d_out=8` в `T5.2`!): `SpatialSetEncoder` → `K_q=4` Perceiver cross-attention queries → `z ∈ R^k (k ∈ {3, 4, 5})` (поддержка `ae_decorr` и `beta_vae` + `mask_consistency` loss по `μ`) → `FiLM Coordinate QueryDecoder(z, coords_j) → (B, N_q, d_out)`.
  2. Обязательная оценка против Фазы 2 (`MLP-AE`): устойчивость $z(t)$ при выпадении **20%, 30% и 40%** станций (`M2_20`, `M2_30`, `M2_40`), плюс кросс-буревое обобщение на 4 фолдах.
- **Обязательные автотесты (`pytest`):** adversarial leakage (`+1e6` → `0` изменение `z`), переменное `N_q` в декодере (вкл. произвольные новые координаты) и переменное `(d_in=18, d_out=8)` для `T5.2`, отсутствие `NaN/Inf`, smoke-обучение 2 эпохи на CPU.
- **Exit Gate:**
  - Локально: `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase3_transformers.py -k "perceiver"` → `100% pass` + локальный `--smoke` прогон (2 эпохи) успешен.
  - В Colab/Kaggle (T4): полный 4-fold LOOE-CV (`phase3_branch_a_perceiver.yaml`), артефакты возвращены в `ml_research/artifacts/p3_A_*`; `cos_sim(z_full, z_M2_40)` выше, чем у `MLP-AE` (Фаза 2).

#### `T3.3` — Ветка B: `Cross-Attention Virtual Magnetometer` (+ Zero-Shot тест `LVV`, `BEL`, `MEA`)
- **Зависимости:** `T3.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/cross_attn_vm.py`
  - `ml_research/configs/phase3_branch_b_cross_attn.yaml`
  - Дополнение `ml_research/tests/test_phase3_transformers.py`
- **Требования к реализации:**
  1. Реализовать `CrossAttentionVirtualMagnetometer` строго по §6.3: `SpatialSetEncoder` → 2 слоя `SpatialCrossAttention(Q=coords_j, K/V=H_obs, bias=SpatialBiasMLP, mask=attn_forbidden)` → `ΔB̂_local(t, j)`.
  2. Защита от multi-hop утечки: лосс считается **строго на целевых станциях `M_target`** (где `obs_mask[:, j] == False` до входа в `SpatialSetEncoder`, `w_obs = 0.0`) + `attn_forbidden` как дополнительный барьер + station-dropout аугментация на `train` + абляции: `with_spatial_bias` vs `without`, `use_station_emb=False` vs `True` (при `use_station_emb=True` неизвестные/Zero-Shot станции получают индекс `<UNK> = N`, а `q_station_id = -1` используется только вне `nn.Embedding` для построения `attn_forbidden`!).
  3. Два обязательных режима оценки на 4-fold LOOE-CV:
     - Стандартный (`M1, M2, M3, M4`) — прямое сравнение с `Wide MLP (B)` и `Ridge-Schur (B)`;
     - **True Zero-Shot (`LVV`, `BEL`, `MEA` полностью исключены из `train`, плюс `SUA` в Fold 3)** — прямое сравнение с `IDW` и `Ordinary Kriging` (Фаза 1)!
- **Обязательные автотесты (`pytest`):**
  - **Adversarial leakage test (включая multi-hop):** при `obs_mask[:, j] == False` (целевая станция `j ∈ M_target`) изменение её поля в `obs_tokens[:, j]` на `+1e6` меняет `H_obs` и прогноз `y_hat[:, j]` строго на `0.0` (`< 1e-6`); плюс при `q_station_id == k_station_id` `attn_forbidden` обнуляет вес внимания на `(j, j)`.
  - **Zero-Shot invariance test:** модель, обученная на 10 станциях, без правки весов предсказывает 3 новые станции с произвольными координатами (`q_station_id = -1`).
  - Smoke-прогон 2 эпохи на CPU без `NaN/Inf`.
- **Exit Gate:**
  - Локально: `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase3_transformers.py` → `100% pass` + `--smoke` прогон успешен.
  - В Colab/Kaggle (T4): полные 4-fold прогоны (стандартный + Zero-Shot) возвращены в `ml_research/artifacts/p3_B_*`; `Cross-Attn VM` превосходит `IDW`/`Kriging` на Zero-Shot станциях (`LVV`, `BEL`, `MEA`) и сопоставим/лучше `Wide MLP` на `M3`/`M4`.

#### `T3.4` — Инфраструктура облачного переноса (Colab / Kaggle упаковщик и тонкие ноутбуки)
- **Зависимости:** `T2.1`, `T3.2`, `T3.3`.
- **Создаваемые файлы:**
  - `ml_research/scripts/pack_for_cloud.py`
  - `ml_research/notebooks/colab_runner.ipynb`
  - `ml_research/notebooks/kaggle_runner.ipynb`
- **Требования к реализации:**
  1. `pack_for_cloud.py` собирает один компактный `ml_research_cloud_bundle.zip` (`geomag_ml/`, `configs/`, `scripts/`, `pyproject.toml`, `requirements_colab_gpu.txt`, `pytest.ini`, `tests/` + выбранный `.npz` и `.meta.json`, строго без `.venv_ml/` и `artifacts/`).
  2. Ноутбуки не содержат кода моделей — только ячейки: (1) проверка GPU `nvidia-smi`, (2) распаковка / `pip install -e . --no-deps` + недостающие пакеты, (3) запуск `python scripts/train.py --config ... --device cuda`, (4) упаковка `artifacts/` в zip с контрольными суммами.
- **Exit Gate:** локальный сухой запуск `pack_for_cloud.py` создаёт валидный zip `< 25 МБ`, внутри которого отсутствует `.venv_ml` и успешно проходит `.venv_ml\Scripts\pytest.exe -c pytest.ini tests -k "not bridge and not dataset_contract"` во временной папке без `backend/`.

---

### ФАЗА 4. Уровень 3 DL — Временной контекст `Shared 1D-CNN → Spatial Transformer`

#### `T4.1` — `TemporalWindowDataset` и `Shared Masked Dilated 1D-CNN` фронтенд
- **Зависимости:** `T1.1`, `T3.1`.
- **Создаваемые файлы:**
  - Дополнение `ml_research/geomag_ml/data/datasets.py` (`TemporalWindowDataset`)
  - `ml_research/geomag_ml/models/temporal_cnn.py` (`MaskedConv1d`, `ChannelLayerNorm`, `MaskedDilatedResBlock`, `SharedTemporalCNN`)
  - `ml_research/tests/test_phase4_temporal_cnn.py` (часть фронтенда)
- **Требования к реализации:**
  1. `TemporalWindowDataset` использует исключительно `build_valid_window_anchors` (нулевое пересечение границ эпизодов и сплитов!), строит 4-й канал `m_step` строго по активной маске `valid_mask` (`M_qc_clean` на `train` и чистой оценке — исключая проникновение `-1036 нТл` сбоя `VAL` и `-320 нТл` сбоя `MEA` в соседние 64-мин окна! — и `M_valid` при сыром инференсе `T5.4`), и отдаёт `X_win: float32[N, 4, W]` со строгим занулением (через `where`/`nan_to_num`) пропущенных шагов и скрытых станций.
  2. `SharedTemporalCNN(W=64, causal=True|False, d_hidden=64, d_out=96, dilations=(1,2,4,8))` с нормализованной `MaskedConv1d` (`bias=False`, `torch.where`/`nan_to_num` на входе, `norm_sum > 0.5`, `clamp(norm_sum, min=1.0)`), покадровой `ChannelLayerNorm(64) * m_out` (без `GroupNorm`!), точными срезами `h_short` (`[W-15:W]` / `[W//2-7:W//2+8]`), общими весами по всем `N` станциям, отсутствием data-dependent Python `if` (для ONNX) и внутренним гейтом `has_any = (m_step.sum(-1, keepdim=True) > 0.5)` на выходе, гарантирующим строгий `0.0` для полностью пропущенной станции (`all m_step == 0`) даже без внешнего `obs_mask`.
- **Обязательные автотесты (`pytest`):**
  - **Causal leakage test:** при `causal=True` изменение любого элемента окна справа от `t_cur` (при инференсе всей последовательности `[0..W-1]`) НЕ меняет выход в `t_cur` (`diff == 0.0` — гарантированно проходит благодаря покадровому `ChannelLayerNorm`!).
  - **Gap & NaN robustness test:** вставка 10-мин и 30-мин пропуска (`m_step=0`, значения `0.0`, `+1e6` **или `NaN`** до зануления) не даёт `NaN/Inf`, а полностью пустая станция (`all m_step == 0`) выдаёт строгий нулевой вектор `0.0` без деления на 0 и утечки bias!
  - **Station weight sharing test:** идентичные временные окна на станции `i` и станции `j` дают побитово одинаковые `h_temp` (`diff == 0.0`).
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase4_temporal_cnn.py -k "cnn or window"` → `100% pass`.

#### `T4.2` — Ветка B (приоритет): `Temporal Cross-Attention VM` (`causal` vs `centered`)
- **Зависимости:** `T3.3`, `T4.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/models/temporal_models.py` (`TemporalCrossAttnVM`)
  - `ml_research/configs/phase4_branch_b_temporal.yaml`
  - Дополнение `ml_research/tests/test_phase4_temporal_cnn.py`
- **Требования к реализации:**
  1. Подключить `SharedTemporalCNN` с прямым скипом мгновенного поля `[h_temp (96), ΔB̃(t) (3), coords (6)]` (`d_in=105`) к `SpatialSetEncoder(d_in=105)` + `CrossAttentionVirtualMagnetometer` (Фаза 3).
  2. Провести два основных 4-fold прогона на Colab/Kaggle T4: `centered (W=64)` (для офлайн-заполнения пропусков и детекции аномалий) и `causal (W=64)` (для квазиреального времени), плюс абляция `W=32` и проверка Zero-Shot (`LVV`, `BEL`, `MEA`).
  3. Дополнительно вычислить спектральную точность в `Pc5`/`Pc6` (PSD-отношение и корреляция полосовых рядов в `auroral` и `europe_mid`).
- **Обязательные автотесты (`pytest`):** формы `(B, N, 4, W) → (B, N_q, 3)`, adversarial masking test (при `obs_mask[:, j] == False` возмущение `+1e6` во всём окне скрытой станции `j` → `0.0` изменение прогноза `y_hat[:, j]`), smoke-прогон 2 эпохи на CPU.
- **Exit Gate:**
  - Локально: `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase4_temporal_cnn.py -k "branch_b"` → `100% pass` + локальный `--smoke` успешен.
  - В Colab/Kaggle (T4): 4-fold прогоны `centered` и `causal` возвращены в `artifacts/p4_B_*`; зафиксирован прирост (или детальное сравнение) против Фазы 3 по `auroral` RMSE и `Pc5/Pc6` точности.

#### `T4.3` — Ветка A (гипотеза Go/No-Go): `Causal Temporal Perceiver-AE`
- **Зависимости:** `T3.2`, `T4.1`.
- **Создаваемые файлы:**
  - Дополнение `ml_research/geomag_ml/models/temporal_models.py` (`TemporalPerceiverAE`)
  - `ml_research/configs/phase4_branch_a_temporal.yaml`
  - `ml_research/scripts/check_phase4_branch_a_gate.py`
  - Дополнение `ml_research/tests/test_phase4_temporal_cnn.py`
- **Требования к реализации:**
  1. Строго **`causal=True`** (`[t-W+1 .. t]`, `W=64`) `SharedTemporalCNN` + скип `ΔB̃(t)` → `SpatialSetEncoder(d_in=105)` → `Perceiver-Bottleneck AE`.
  2. Автоматический скрипт `check_phase4_branch_a_gate.py` сравнивает 4-fold результаты `p4_A_*` с `p3_A_*` с блочным бутстрепом (блоки по 360 мин, 1000 реплик) и выносит формальный вердикт `GO` или `NO_GO` по критерию §6.4, сохраняя `artifacts/phase4_branch_a_decision.json`.
- **Обязательные автотесты (`pytest`):** строгая каузальность, zero-leakage маски, корректность логики принятия решения `GO`/`NO_GO` на синтетических метриках.
- **Exit Gate:**
  - Локально: `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase4_temporal_cnn.py` → `100% pass`.
  - После Colab/Kaggle 4-fold прогона: сформирован `artifacts/phase4_branch_a_decision.json`, официально фиксирующий лучшую модель Ветки A для Фазы 5.

---

### ФАЗА 5. Научные эксперименты под тему диплома (события + вейвлеты + синтез)

#### `T5.1` — Ветка A: Физическая валидация `z(t)` и распознавание фаз бури
- **Зависимости:** лучшая модель Ветки A (`T3.2` или `T4.3`).
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/experiments/{__init__.py, branch_a_physics.py}`
  - `ml_research/configs/phase5_experiments.yaml`
  - `ml_research/scripts/run_phase5.py`
  - `ml_research/tests/test_phase5_experiments.py` (часть 1)
- **Требования к реализации:**
  1. **Учёт произвольного поворота/знака латентных осей между фолдами (КРИТИЧНО!):** 4 независимо обученных автоэнкодера (или PCA) в `looe_4fold` живут в произвольно повёрнутых/отражённых базисах $\mathbb{R}^k$, поэтому наивная склейка сырых `z_test` из 4 фолдов разрушает поосевые корреляции, траектории и кластеризацию!
     - Для строгой количественной оценки (`R²` регрессии `z → target`, кросс-корреляции по лагам, фазовая кластеризация) расчёты выполняются **строго внутри каждого фолда `f`** (обучение регрессора и кластеризатора на `z^{(f)}(non_test_f)`, оценка на `z^{(f)}(test_f)` с агрегацией `mean ± std` по фолдам; `OMNI` агрегируется строго по покрытым тестовым фолдам — Folds 0–1 на октябре 2026).
     - Для сквозной 4-буревой визуализации `z_1..z_k(t)` и 2D/3D-траекторий (`fig2`) выполняется **ортогональное выравнивание Прокруста без утечки `test`**: для каждого фолда `f ∈ {1, 2, 3}` находится ортогональная матрица $R_f \in O(k)$ строго на общем нетестовом перекрытии $S_{\text{align}}(f) = \text{non\_test}_f \cap \text{non\_test}_0$ (гарантированно `≥ 7 200 мин` в каждом фолде: `[ep2, ep3]` для `f=1`, `[ep1, ep3]` для `f=2`, `[ep1, ep2]` для `f=3`; нулевое пересечение с `test_f` и `test_0`!), после чего оси канонически упорядочиваются по SVD на `z^{(0)}(train_0)` со знаком, фиксированным по отрицательной корреляции `z_1` с медианой $\Delta X$ в `EU_mid` (положительный `z_1` = буревая депрессия).
  2. Посчитать пер-фолдовые и выровненные корреляции (Пирсон, Спирмен), кросс-корреляцию по лагам `[-90, +90] мин` и out-of-fold линейную/$R^2$ регрессию `z(t) → {OMNI AE/AL/AU/SYM-H*/PC*/Kp*, SOLAR-1 BSN Bz/|B|/Vx/Pdyn/Ey, OULU ΔN/N0}`.
  3. Обучить кластеризацию / модель режимов в $z$-пространстве (`GaussianMixture` + марковское сглаживание Витерби / `KMeans` из `scikit-learn`, плюс опциональный `GaussianHMM` при наличии `hmmlearn`, для `K ∈ {3, 4, 5}`) **строго внутри каждого фолда** (fit на `non_test_f`, predict на `test_f`) для автоматического выделения фаз бури (*спокойная/предштормовая, начальная/SC, главная фаза/суббури, восстановление*) и оценить согласованность с квантилями `AE` / `Bz` / `SYM-H*` (ARI, NMI, Silhouette — инвариантны к перестановке номеров кластеров).
  4. Сохранить 2D/3D-траектории (с Прокруст-метрикой согласования `cos_sim` на $S_{\text{align}}$) и таблицы в `artifacts/p5_branch_a_physics/`.
- **Обязательные автотесты (`pytest`):** корректность вычисления лаг-профиля на сдвинутом синусе, точность ортогонального выравнивания Прокруста на случайно повёрнутом/отражённом синтетическом $z$ (без касания `test_idx`!), NaN-safe пропуск отсутствующих столбцов/фолдов OMNI, конечные ARI/NMI.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase5_experiments.py -k "branch_a"` → `100% pass` + сгенерированы `artifacts/p5_branch_a_physics/summary.json` и таблицы корреляций.

#### `T5.2` — Вейвлет-эксперимент Ветки A: «Нейро-ULF-индекс» (`Pc5`, `Pc6`, `30–120 мин`, `>2 ч`)
- **Зависимости:** `T3.2`, `T5.1`.
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/experiments/neuro_ulf_index.py`
  - Дополнение `ml_research/tests/test_phase5_experiments.py`
- **Требования к реализации:**
  1. Построить **единую глобальную сетку масштабов** однократным вызовом `geomag_core.cross_wavelet.build_morlet_scales(n_samples=max_ep_len, dt=60.0, period_min_sec=120.0, period_max_sec=14400.0, dj=1/12, omega0=6.0)` (`83` масштаба, `120 .. 13 684 с` — нельзя пересчитывать `build_morlet_scales` на каждом коротком подсегменте, иначе длины сеток разойдутся!). Для каждой станции и эпизода разбить ряд на чистые непрерывные подсегменты **строго по `M_qc_clean`** через `find_valid_subsegments(..., min_length=60)` (не по `M_valid`, чтобы приборные плато `VAL` `-1036 нТл` и `MEA` `-320 нТл` не давали гигантский звон `~10^6 нТл²`!) и рассчитать CWT через `morlet_cwt_single(seg, scales, dt=60.0, omega0=6.0)` *(при альтернативном использовании `compute_gap_coi_scales` помнить про умножение на `MORLET_FOURIER_FACTOR ≈ 1.033`, тогда как `morlet_cwt_single` возвращает `coi_periods` сразу в секундах периода!)*.
  2. Вычислить 4 полосовые **масштабно-усреднённые вейвлет-мощности по Torrence & Compo (1998, Eq. 24)** в физических единицах $[\text{нТл}^2]$ (устраняет `12×` перекос масштабов наивного среднего $|W|^2$!):
     $$P_{\text{band}}(t) = \frac{\delta j \cdot \delta t}{C_\delta}\sum_{j \in \text{band}} \frac{|W(s_j, t)|^2}{s_j}, \qquad C_\delta = 0.776 \text{ (Морле } \omega_0=6)$$
     и взять $\log_{10}(1 + P_{\text{band}})$ отдельно для горизонтального поля ($|W_X|^2 + |W_Y|^2$) и вертикального ($|W_Z|^2$) → тензор `W_pow: float32[T, N, 8]` и строгую пополосную COI-маску **`M_coi[t, i, b] = bool(coi_periods[t] >= periods_sec[band_b].max())`** (`bool[T, N, 4]`: гарантирует, что все масштабы полосы лежат внутри COI — на 2-суточных эпизодах COI обрезает `21.7%` отсчётов в полосе `>2ч`, а на 512-мин сегменте `LVV@ep3` — `100.0%` для `>2ч`!):
     - `Pc5`: `150–600 с` (`[150, 600)`)
     - `Pc6`: `600–1800 с` (`[600, 1800)`)
     - `Substorm / meso`: `1800–7200 с` (`[1800, 7200)`, `30–120 мин`)
     - `Storm / long`: `7200–14400 с` (`[7200, 14400]`, `> 2 ч`)
  3. Стандартизовать 8 каналов `W_pow` покомпонентным `z-score` строго по `train` с маской `M_coi_exp` (`bool[T, N, 8]`), и прогнать лучшую архитектуру Ветки A (`PerceiverBottleneckAE(d_in=18, d_out=8)` со входом `[W_pow_norm_masked (8), M_coi (4), coords (6)]`, `obs_mask = M_coi.any(-1)` и пополосным маскированием `HuberLoss` по `M_coi_exp`, плюс сравнение с `PCA(k=3..5)` по тем же вейвлет-признакам) на 4-fold LOOE-CV → получить 3–5-мерный **нейро-ULF-индекс $z_{\text{ULF}}(t)$** (с тем же пер-фолдовым расчётом $R^2$ и Прокруст-выравниванием из `T5.1` для сводных графиков `fig3`!).
  4. Проверить связь $z_{\text{ULF}}(t)$ со скоростью солнечного ветра `|Vx|`, `Pdyn`, `Bz_GSM` (BSN), `AE` и `OULU`.
- **Обязательные автотесты (`pytest`):** монохроматическая волна с амплитудой `A = 10 нТл` (дисперсия `50 нТл²`) и периодом `T = 300 с` даёт `P_Pc5 ∈ [45, 55] нТл²` (и `< 1` в других полосах), а волна `T = 1200 с` — `P_Pc6 ∈ [45, 55] нТл²` (подтверждает отсутствие масштабного перекоса!); отсутствие `NaN` внутри валидной COI-маски; корректный прогон `PerceiverBottleneckAE(d_in=18, d_out=8)` с частичной маской `M_coi`.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase5_experiments.py -k "ulf"` → `100% pass` + сгенерирован `artifacts/p5_neuro_ulf_index/summary.json` с out-of-fold $R^2$ и корреляциями `z_ULF(t) ↔ {Vx, Pdyn, AE}`.

#### `T5.3` — Ветка B: Заполнение 6–12 ч пропусков и проверка сохранности `Stokes` & `XWT/WTC`
- **Зависимости:** лучшие модели Ветки B всех уровней (`T1.4`, `T2.3`, `T3.3`, `T4.2`).
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/eval/stokes_wavelet_eval.py`
  - `ml_research/geomag_ml/experiments/gap_filling_stokes.py`
  - Дополнение `ml_research/tests/test_phase5_experiments.py`
- **Требования к реализации:**
  1. На 4 эталонных станциях (`LVV`, `BEL`, `ABK`, `OTT`) в активных интервалах out-of-fold эпизодов с непрерывным чистым покрытием `≥ 24 ч` *(для `LVV` использовать `ep0/ep1/ep2`, т.к. в `ep3` у `LVV` только 512 мин!)* вырезать искусственные непрерывные пропуски длительностью **6 ч (360 мин)** и **12 ч (720 мин)** (не менее 2 окон на станцию) с обязательным отступом **`≥ 3 ч` (`180 мин`)** от границ чистого подсегмента/эпизода (гарантирует 100% валидность окон Фазы 4 `W=64` и 100% попадание всего пропуска внутрь COI для `Pc5` и `Pc6`!).
  2. **Основной бенчмарк (режим штатного простоя прибора):** восстановить пропуски **стандартными** out-of-fold моделями Ветки B (где все 4 обсерватории `LVV, BEL, ABK, OTT` входят в `in_train`, что имитирует реальный временный простой прибора действующей станции и позволяет честно сравнить все уровни): `LinearInterp`, `Ridge/Kriging` (Ф1), `Wide MLP` (Ф2), `Cross-Attn VM` (Ф3), `Temporal 1D-CNN+Transformer centered` (Ф4), и **вшить восстановленный фрагмент обратно в полный эпизод** (устраняя искусственные краевые ступеньки для вейвлета!). Опционально (отдельной вторичной строкой сравнения в `comparison.csv`) прогнать Zero-Shot модели координатного класса (`IDW/Kriging`, `Cross-Attn VM`, `Temporal Cross-Attn VM`, где `LVV` и `BEL` полностью исключены из `train`) на тех же 6–12 ч окнах `LVV` и `BEL`.
  3. Прогнать **эталонный** и **восстановленные** ряды через `backend/geomag_core` (`core_bridge`):
     - `compute_welch_spectral_matrix` + `lookup_welch_effective_n(..., spectrum_type="red")` (плюс опциональная проверка `"flat"`) + `stokes_from_spectral_matrix_2d` (для `Pc5` `L=128, M=64` и `Pc6` `L=256, M=64`, `alpha=0.05`, по окну пропуска): измерить ошибку $\Delta \log_{10} I$ (и отношение мощности `I_recon / I_fact`), `MAE(p)`, точную циркулярную ошибку азимута **$\Delta\psi_{\text{circ}} = \min(|\Delta\psi| \bmod 180^\circ,\, 180^\circ - (|\Delta\psi| \bmod 180^\circ)) = |((\Delta\psi + 90^\circ) \bmod 180^\circ) - 90^\circ| \in [0^\circ, 90^\circ]$** (на значимых окнах `p_fact > p_crit`), `MAE(ε)` и долю совпадения знака вращения `CW/CCW` (`sign(ε)`), а также `polarization_3d` (`P`, `θ`, `b/a`).
     - **КРИТИЧЕСКИЙ ПОРОГ ДЛЯ `LinearInterp` / НУЛЕВОЙ МОЩНОСТИ:** `compute_welch_spectral_matrix` выполняет `detrend_linear_ols`, который полностью обнуляет линейную интерполяцию пропуска до машинного шума `tr(S) ~ 10^-22` (давая ложную поляризацию `p = 0.96` из пыли округления!) или строгого `0.0` (где `polarization_3d(S)` бросает `ValueError: Trace of spectral matrix must be positive`)! Поэтому в `stokes_wavelet_eval.py` обязателен порог **`I_min = tr_S_min = 1e-9 нТл²/Гц`**: при `I <= 1e-9` или `tr(S) <= 1e-9` выставляется `stokes_defined = False`, `p/ψ/ε/P_3D = NaN`, `power_ratio = 0.0`, `log10(max(I, 1e-12))`, без вызова `polarization_3d` (что физически и количественно доказывает полную потерю `Pc5/Pc6` волн при линейной интерполяции!).
     - `compute_xwt_wtc` (строго две раздельные проверки внутри `gap & inside_coi` для `Pc5` и `Pc6`): (а) **самокогерентность `fact_i ↔ recon_i`** (точность восстановления фазы и огибающей: целевые `WTC R² → 1`, `Δφ → 0°`, `n_surrogates=0`); (б) **сохранность межстанционного `XWT/WTC` (`recon_i ↔ ref_j` vs `fact_i ↔ ref_j`)** на парах `LVV↔BEL`, `ABK↔KIR`, `OTT↔STJ` (где опорная станция `ref_j` остаётся **наблюдаемой**, а не скрытой одновременно; для ключевых пар также `n_surrogates=300` для проверки совпадения 95% контуров значимости!).
- **Обязательные автотесты (`pytest`):** проверка обёртки `stokes_wavelet_eval.py` на `synthetic_ellipse_2d` (идеальная копия даёт `MAE(p)=0, Δψ=0°, MAE(ε)=0, WTC R² > 0.99`), корректный циркулярный `Δψ_circ = 2°` на границе `+89°` vs `-89°`, и **защитный тест `LinearInterp` (наклонная и постоянная прямая не роняют `polarization_3d` и возвращают `stokes_defined=False`, а НЕ фиктивное `p > 0.9`)**.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase5_experiments.py -k "stokes or gap"` → `100% pass` + таблица `artifacts/p5_gap_filling_stokes/comparison.csv`.

#### `T5.4` — Ветка B: Детектор аномалий `факт − прогноз` (реальные спайки + синтетика `5..50 нТл` + сценарий затмения/SPE)
- **Зависимости:** лучшая модель Ветки B (`T3.3` или `T4.2`), недеспайкнутый датасет (`T0.3`).
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/experiments/anomaly_detection.py`
  - `ml_research/geomag_ml/experiments/eclipse_spe_protocol.py`
  - Дополнение `ml_research/tests/test_phase5_experiments.py`
- **Требования к реализации:**
  1. Вычислить leave-one-station-out (`M1`) невязку $r_{t,i,c} = \Delta B_{\text{fact}}(t,i,c) - \Delta\hat{B}_{B,\text{M1}}(t,i,c)$ (станция `i` никогда не предсказывает сама себя!) и робастно нормированный скор $s(t, i) = \sqrt{\sum_{c} ((r_{t,i,c} - \tilde{\mu}_{i,c})/\max(\tilde{\sigma}_{i,c}^{\text{MAD}}, 1.0))^2}$. Параметры $(\tilde{\mu}_{i,c}, \tilde{\sigma}_{i,c}^{\text{MAD}})$ оцениваются строго по чистым `val`-невязкам (`M_qc_clean` в `val_idx`) при `n_val_clean(i) >= 60`, а для станций с `< 60` чистых минут в `val` *(в `looe_4fold` это `2–9` тестовых станций/фолд: `CLF/DUR/EBR/GUI/HRB/LON/RES/SFS/SPT` в Folds 0–1, включая бонусную `SFS@ep1`, `DOU/MAB/NCK/SFS` в Fold 2 и `NCK/SUA` в Fold 3!)* применяется **поясной фолбэк** $(\tilde{\mu}_{\text{belt}(i),c}, \tilde{\sigma}_{\text{belt}(i),c}^{\text{MAD}})$ — медиана по станциям того же пояса в `val` (нулевой `NaN`, нулевая утечка `test`!). При прогоне на недеспайкнутом наборе строго применяются замороженные `B0_nT` и `scaler.json` из clean-обученного чекпоинта (нулевой перефит, нулевое заражение скейлера спайками!).
  2. **Тест 1 (реальные приборные дефекты):** прогон на `geomag_ml_oct2026_undespiked_v1.npz` — измерить Precision/Recall/F1 и ранги обнаружения на всех **10 реальных 1-минутных спайках (`VAL`, `OTT`, `MEA`, `IZN`)** и на всех **5 многоминутных приборных окнах (`VAL` 84м, `OTT` 31м, `MEA` 50м, `IZN` 4м+4м — 174 станция-минут)** в сравнении с базовым 1-минутным деспайкером Хампеля, плюс бонусный тест обнаружения базовых ступенек на `LON@ep3` (`+72..76 нТл`) и `SFS@ep1` (`+61 нТл`)!
  3. **Тест 2 (синтетические локальные импульсы):** инжекция одностанционных импульсов амплитудой **`5, 10, 20, 50 нТл`** и длительностью `1, 5, 30 мин` в чистые тестовые интервалы → ROC-AUC, PR-AUC и Recall @ 1% FPR отдельно по поясам (`europe_mid`, `auroral`, `canada`).
  4. **Тест 3 (готовый сценарий для реальных дат затмений и SPE — `eclipse_spe_protocol.py`):**
     - Режим `eclipse`: по статическому центру `(lat, lon, R)` или движущейся трассе затмения `track: [{timestamp_utc, lat_deg, lon_deg, radius_km}]` (`R ∈ [500, 1000] км`) динамически в каждую минуту `t` маскирует **все** станции в зоне тени (маска типа `M3`), строит невозмущённый прогноз по внешним станциям и выделяет остаточный ионосферный сигнал `B_fact - B_pred` с доверительными интервалами по контрольным дням.
     - Режим `spe`: применяет маску `M4` (прогноз `MLat ≥ 60°` по `MLat < 60°`), выделяет высокоширотную аномалию `B_fact - B_pred` в сопоставлении с `OULU` и `SOLAR-1`.
     - Проверить готовность `eclipse_spe_protocol.py` сквозным прогоном на синтетической заглушке события (`dry-run`), чтобы при получении реальных дат от соруководителя достаточно было заполнить `events.yaml` и запустить 1 команду!
- **Обязательные автотесты (`pytest`):** `ROC-AUC > 0.95` на синтетическом импульсе `50 нТл` в `europe_mid`; **поясной фолбэк даёт конечный скор $s(t, i)$ без `NaN` на тестовой станции с `0` строками в `val`** (имитация `SFS@ep1`); `eclipse` маска гарантированно исключает все станции внутри радиуса тени (вкл. движущуюся трассу) из `M_obs`; `spe` маска исключает все станции `MLat >= 60°`.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase5_experiments.py -k "anomaly or eclipse"` → `100% pass` + `artifacts/p5_anomaly_detection/summary.json` (таблицы реальных спайков + ROC/PR для `5, 10, 20, 50 нТл`).

#### `T5.5` — Синтез A + B: трёхуровневое физическое разложение поля
- **Зависимости:** лучшие модели A (`T3.2`/`T4.3`) и B (`T3.3`/`T4.2`).
- **Создаваемые файлы:**
  - `ml_research/geomag_ml/experiments/synthesis_decomposition.py`
  - Дополнение `ml_research/tests/test_phase5_experiments.py`
- **Требования к реализации:**
  1. На всех out-of-fold тестовых фолдах вычислить точное разложение:
     $$\Delta B_{\text{fact}} = \underbrace{\Delta\hat{B}_A}_{\text{global}} + \underbrace{(\Delta\hat{B}_B - \Delta\hat{B}_A)}_{\text{meso}} + \underbrace{(\Delta B_{\text{fact}} - \Delta\hat{B}_B)}_{\text{local}}$$
     (проверять точное алгебраическое тождество `max|global + meso + local - fact| < 1e-5 нТл`!).
  2. Вычислить вклад каждого уровня в дисперсию и RMS [нТл] по поясам (`europe_mid`, `auroral`, `canada`), по компонентам `(X, Y, Z)` и по фазам бури (из `T5.1`).
  3. Проанализировать физическую природу `meso` (авроральные электроджеты, региональные FAC) и `local` (приборные сбои, локальные индуцированные токи / ионосферные эффекты).
- **Обязательные автотесты (`pytest`):** точное тождество суммы 3 компонент `diff < 1e-5`, корректный подсчёт долей дисперсии.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_phase5_experiments.py` → `100% pass` + `artifacts/p5_synthesis/decomposition_stats.csv`.

---

### ФАЗА 6. Сводные таблицы, воспроизводимость и подготовка артефактов

#### `T6.1` — Единый агрегатор сводных таблиц всех фаз и веток
- **Зависимости:** Фазы 1–5 завершены.
- **Создаваемые файлы:**
  - `ml_research/scripts/aggregate_benchmarks.py`
- **Требования к реализации:**
  1. Собрать все `artifacts/*/metrics.json` по единому реестру лучших запусков (4-fold `mean ± std`; для `OMNI AE/AL/AU` — среднее строго по покрытым тестовым фолдам с явной колонкой `n_folds_covered=2/4` без `NaN`; для `Zero-Shot` — отдельно `heldout_zero_shot` (`LVV, BEL, MEA` по 4 фолдам) и `single_episode_unseen` (`SUA` в Fold 3)), провалидировать совпадение `dataset_sha256` и протокола сплитов (при несовпадении бросить `ValueError`, исключая нечестные сравнения!).
  2. Сгенерировать единые сводные таблицы (`CSV`, `Markdown`, `LaTeX`) в `ml_research/artifacts/final_tables/`:
     - `table1_branch_a_comparison.{csv,md,tex}` (`PCA`, `PCA-Sq`, `MLP-AE/β-VAE`, `Perceiver-AE`, `Temporal Perceiver` — RMSE/R², стабильность `M2`, корреляции `OMNI/SOLAR-1/OULU`);
     - `table2_branch_b_comparison.{csv,md,tex}` (`Ridge`, `IDW`, `Kriging`, `Wide MLP`, `Cross-Attn VM`, `Temporal CNN+Attn causal/centered` — `M1..M4` + `Zero-Shot` по поясам и компонентам);
     - `table3_compression_price_and_decomposition.{csv,md,tex}`;
     - `table4_stokes_wtc_and_anomaly_roc.{csv,md,tex}`.
- **Exit Gate:** запуск `.venv_ml\Scripts\python.exe ml_research/scripts/aggregate_benchmarks.py` → без ошибок, все 4 таблицы непустые и без `NaN` в ключевых столбцах.

#### `T6.2` — Генератор публикационных графиков диссертации (`300 DPI PNG` + `SVG`)
- **Зависимости:** `T6.1`, Фаза 5.
- **Создаваемые файлы:**
  - `ml_research/scripts/make_thesis_figures.py`
- **Требования к реализации:**
  1. Светлая печатная палитра (согласована с `Print Light` темой `GeoMag Analyzer`), шрифт ≥ 10 pt, экспорт каждой иллюстрации одновременно в `PNG 300 DPI` и `SVG` в `ml_research/artifacts/figures/`:
     - `fig1_network_belts_and_baseline`: карта поясов и иллюстрация $B_0$ vs буревой сдвиг;
     - `fig2_branch_a_neuro_index_vs_omni`: временной ход `z_1..z_k(t)` рядом с `AE/AL`, `SOLAR-1 Bz/Pdyn`, `OULU` и 2D/3D-траектория фаз бури;
     - `fig3_neuro_ulf_index`: многомасштабный вейвлет-индекс `z_ULF(t)` vs `|Vx|` и `AE`;
     - `fig4_branch_b_gap_filling_stokes`: пример заполнения 12-ч пропуска (`LVV` и `ABK`) + сравнение `Stokes (p, ψ, ε)` и `WTC R²`;
     - `fig5_anomaly_detection_val_ott_roc`: реальные аномалии `VAL` (84 мин + спайк) и `OTT` (`fact vs pred vs residual`) + кривые ROC/PR для `5, 10, 20, 50 нТл` по поясам;
     - `fig6_three_level_synthesis`: трёхуровневое разложение `global + meso + local` в ходе бури 9–11 окт. 2026.
- **Exit Gate:** запуск `.venv_ml\Scripts\python.exe ml_research/scripts/make_thesis_figures.py` → все 6 пар файлов `.png` (проверено 300 DPI) и `.svg` созданы, размер `> 0`.

#### `T6.3` — Проверка и фиксация ONNX-экспорта лучших моделей (A и B)
- **Зависимости:** лучшие чекпоинты Ветки A и Ветки B.
- **Создаваемые файлы:**
  - `ml_research/scripts/export_onnx.py`
  - `ml_research/tests/test_onnx_export.py`
- **Требования к реализации:**
  1. Экспортировать лучшие модели Ветки A и Ветки B в `model.onnx` (`opset_version=17`, динамическая ось батча `B`, а для трансформеров Фаз 3–4 — также динамические оси `N_obs` и `N_q`, чтобы принимать расширенную сеть `N > 49` без переэкспорта; для `MLP-AE` при `d_t == 0` не передавать тензор нулевой ширины!) с сохранением в `artifacts/<run_id>/model.onnx`.
  2. Прогнать `onnx.checker.check_model` и сравнить выход `PyTorch (CPU eval)` и `onnxruntime.InferenceSession (CPUExecutionProvider)` на случайных и реальных батчах (включая переменное `N_obs` и `N_q`): обязательный допуск `max|y_torch - y_onnx| < 1e-4`.
  3. Зафиксировать обновлённый `onnx_export_spec.json` и контрольные суммы (готово к будущей интеграции в `GeoMag Analyzer` без зависимости от PyTorch в дистрибутиве Windows!).
- **Обязательные автотесты (`pytest`):** экспорт и паритет `< 1e-4` для `MaskedConditionalMLPAE`, `MaskedWideMLP`, `PerceiverBottleneckAE`, `CrossAttentionVirtualMagnetometer`, `TemporalCrossAttnVM`.
- **Exit Gate:** `.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests\test_onnx_export.py` → `100% pass` + `export_onnx.py` создаёт валидные `.onnx` для финалистов A и B.

---

## 8. Схема запуска: локально (CPU) vs Google Colab / Kaggle (GPU T4)

### 8.1. Распределение вычислений

| Этап | Где запускается | Причина и ориентировочный бюджет |
|---|---|---|
| **Фаза 0** (`.venv_ml`, сборка `clean` + `undespiked` `.npz`, скейлеры) | **Локально (CPU)** | Чтение локальных `data/processed/` и `GeoMag_Storms_Dataset_Oct-2026/`; `< 20 с`. |
| **Фаза 1** (`PCA ± Sq`, `Ridge-Schur`, `IDW`, `Kriging`, 4 фолда) | **Локально (CPU)** | Чистая линейная алгебра NumPy/SciPy; `< 2 мин` на всё. |
| **Фаза 2** (`Masked MLP-AE/β-VAE` + абляции, `Wide MLP`, 4 фолда) | **Локально (CPU)** *(или Colab при расширенном многомесячном датасете)* | На 10 сутках (`5 760–8 640` train-точек/фолд при выделенном `inner_val`) 1 фолд занимает `~1–2 мин` на Ryzen 6c/12t; вся Фаза 2 с абляциями ~20–30 мин. |
| **Все автотесты `pytest` и smoke-тесты Фаз 3–4** | **Локально (CPU)** | Мгновенная обратная связь при разработке (`--smoke`: 2–3 эпохи, 1 фолд, `< 3 мин`). |
| **Фаза 3** (Spatial Set Transformers A и B, 4 фолда + Zero-Shot + абляции) | **Colab / Kaggle (GPU T4)** | Self/Cross-Attention $O(N^2)$ на `5 760–8 640` train-точках × 100 эпох × 4 фолда × абляции: на CPU ~2–4 ч, на **T4 ~6–12 мин**! |
| **Фаза 4** (`Shared Masked 1D-CNN → Spatial Transformer`, `W=64`, A и B) | **Colab / Kaggle (GPU T4)** | Тензор `(B, 49, 4, 64)` с 4 блоками Dilated MaskedConv1d + Transformer: на CPU ~8+ ч, на **T4 ~15–25 мин** на конфигурацию! |
| **Полный пересчёт после догрузки многомесячных данных** | **Colab / Kaggle (GPU T4)** | Сборка нового `.npz` локально → загрузка 1 файла в облако → перезапуск всех конфигов Фаз 2–4 на T4. |
| **Фазы 5 и 6** (физика `z(t)`, CWT-вейвлеты, `Stokes`/`XWT/WTC` через `geomag_core`, аномалии, таблицы, графики, ONNX) | **Локально (CPU)** | Использует готовые веса из `artifacts/` и прямой `core_bridge` к `backend/geomag_core/`; инференс на CPU занимает секунды. |

### 8.2. Пошаговый протокол переноса в Colab / Kaggle и возврата артефактов

1. **Локальная упаковка (1 команда):**
   ```powershell
   .venv_ml\Scripts\python.exe ml_research/scripts/pack_for_cloud.py `
       --dataset ml_research/data_cache/geomag_ml_oct2026_clean_v1.npz `
       --out ml_research/data_cache/ml_research_cloud_bundle.zip
   ```
   Архив (~15–20 МБ) содержит: `geomag_ml/`, `configs/`, `scripts/`, `pyproject.toml`, `requirements_colab_gpu.txt`, `.npz` и `.meta.json`.
2. **Запуск в Google Colab или Kaggle (GPU T4):**
   - Загрузить `ml_research_cloud_bundle.zip` на Google Drive (`/MyDrive/ONU_Thesis_ML/`) или как приватный Kaggle Dataset.
   - Открыть `ml_research/notebooks/colab_runner.ipynb` (или `kaggle_runner.ipynb`), выбрать runtime `T4 GPU`.
   - Выполнить 3 стандартные ячейки:
     ```bash
     unzip -q /content/drive/MyDrive/ONU_Thesis_ML/ml_research_cloud_bundle.zip -d /content/ml_work
     cd /content/ml_work && pip install -q -e . --no-deps && pip install -q pyyaml pyarrow
     python scripts/train.py \
         --config configs/phase3_branch_b_cross_attn.yaml \
         --dataset data_cache/geomag_ml_oct2026_clean_v1.npz \
         --fold all --device cuda --out-dir /content/artifacts_out
     ```
   - По завершении упаковать результаты:
     ```bash
     cd /content/artifacts_out && zip -qr /content/drive/MyDrive/ONU_Thesis_ML/artifacts_p3_B.zip .
     ```
3. **Возврат и верификация локально:**
   - Скачать `artifacts_p3_B.zip` и распаковать в `c:\Programming\Projects\ONU_Thesis\ml_research\artifacts\`.
   - Проверить целостность и воспроизводимость инференса локально на CPU:
     ```powershell
     .venv_ml\Scripts\python.exe ml_research/scripts/evaluate.py `
         --run-dir ml_research/artifacts/<run_id> `
         --dataset ml_research/data_cache/geomag_ml_oct2026_clean_v1.npz `
         --device cpu
     ```
     Скрипт сверяет `dataset_sha256` из `config.json` и подтверждает совпадение пересчитанных метрик на CPU с облачным `metrics.json` (допуск `< 1e-3 нТл`).

---

## 9. Матрица рисков и инженерные меры защиты в коде

| № | Риск (источник и проявление) | Тяжесть | Конкретная инженерная мера защиты в коде | Где проверяется (`pytest` / Gate) |
|---|---|---|---|---|
| **R1** | **Переобучение на 10 сутках (4 бури)** до прихода расширенных данных: модели запоминают конкретные бури или календарный день (`DOY`). | Высокая | (1) Строгий 4-fold LOOE с отдельным `inner_val` эпизодом; (2) компактные сети (`0.4–1.8M` парам.), `weight_decay=1e-4`, `Dropout(0.1..0.15)`, early stopping; (3) `use_doy=False` при охвате `< 60` дней; (4) аугментация смесями масок + station-dropout; (5) отчёт строго `mean ± std` по 4 фолдам. | `T1.1`, `T2.2`, `T3.2`, `T6.1` |
| **R2** | **Утечка базовой линии $B_0$ и скейлера / срезание `Dst/SYM-H`:** подсчёт $B_0$ по всему датасету или вычитание по-эпизодной медианы уничтожает буревой сдвиг и завышает метрики. | Критическая | (1) `BaselineEstimator.fit` и `FieldScaler.fit` принимают строго `train_idx`; (2) два явных режима (`temp_train_quiet_proxy` с внутриэпизодной $\Delta_1$ и `target_kp_quiet_night`), по-эпизодная медиана запрещена как основной режим; (3) `fit_split_hash` сохраняется в `scaler.json`. | `T0.4`: тест возмущения `test` на `+5000 нТл` → `0` сдвиг `B0`/`scaler` |
| **R3** | **Временная утечка через сплит или окна (автокорреляция `r(1 мин) = 0.9975`):** случайный сплит или окно `W=64`, пересекающее границу train/test, даёт фиктивный `R² ≈ 0.99`. | Критическая | (1) Случайный минутный сплит запрещён архитектурно; (2) `purge_gap_min >= 1440` в `block_temporal`; (3) `eclipse`/`spe` жёстко в `test`; (4) `build_valid_window_anchors` отбрасывает любое окно, пересекающее границу эпизода/сплита или имеющее шаг `≠ 60 с`. | `T1.1`, `T4.1`: жёсткие unit-тесты на зазор и границы окон |
| **R4** | **Доминирование авроральных амплитуд (`σ ≈ 105 нТл` vs `22 нТл` в EU_mid → `23×` перекос в MSE):** модель игнорирует среднеширотные вариации и `Pc5/Pc6`. | Высокая | (1) Основной режим масштабирования **`asinh_global`** (линейный вблизи 0, логарифмический на пиках `500–1500 нТл`); (2) `HuberLoss(δ=1.0)`; (3) обязательная раздельная отчётность по поясам (`europe_mid`, `auroral`, `canada`) и компонентам `(X, Y, Z)`. | `T0.4`, `T1.3`, `T2.2` (абляция `z-score` vs `asinh`) |
| **R5** | **Безбазисные `GCK`/`NCK` и скачки базы `Variation` на `LON@ep3` (`+72..76 нТл`) / `SFS@ep1` (`+61 нТл`):** смещённый DC-уровень вносит `45–75 нТл` фиктивного RMSE в `EU_mid`. | Высокая | (1) Флаг `station_has_baseline` из `catalog.sqlite`; (2) `baseless_policy` + абляция `with/without GCK,NCK`; (3) `variation_step_policy="flag_qc_exclude"` исключает `LON@ep3`/`SFS@ep1` из `M_qc_clean` (оставляя их как тест ступенек в `T5.4`) + автодетектор скачка `> 30 нТл` в `.meta.json`. | `T0.2`, `T0.4`, `T2.2`, `T5.4` |
| **R6** | **Неполные станции (`SUA` только в `ep3`, `DUR/EBR/HRB` только в `ep0–1`, `NCK` только в `ep2–3`) и QC-исключение `SFS@ep1`:** наивные сплиты дают `0` train-строк, `NaN` в скейлере/детекторе аномалий, взрыв $S_q(\text{UT})$ на `15/24` ч UT (`+261 нТл`) и падение `Ridge-Schur` на `0`-overlap парах! | Высокая | (1) QC-верифицированный `looe_4fold` (`T1.1`) даёт `0` необученных тестовых станций под `M_valid` и `M_qc_clean` кроме одноэпизодной `SUA` (Fold 3) и `OMNI > 0` в каждом фолде; (2) EM-Gaussian ковариация в `Ridge-Schur` и адаптивное $S_q(\text{UT})$ (`λ=0.01·n_obs`, `M_h=1` при `<20` ч UT); (3) `in_train`-барьер (`M1–M4` vs раздельные `heldout_zero_shot` и `single_episode_unseen`), поясной фолбэк скейлера и `val`-калибратора `T5.4`. | `T0.4`, `T1.1`, `T1.2`, `T1.3`, `T1.4`, `T3.3`, `T5.4` |
| **R7** | **Многоминутные приборные аномалии в `data/processed/` (`VAL` 84м, `OTT` 31м, Zero-Shot `MEA` 50м до `-320 нТл`, `IZN` 4м+4м):** искажают `train`-лосс, CWT-мощность и Zero-Shot метрику на `MEA`! | Высокая | (1) Декларативная маска `M_qc_clean` исключает все 5 подтверждённых окон из `train`-лосса, чистой оценки, Фазы 4 `m_step` и CWT-подсегментов `T5.2` (не меняя `data/processed/`); (2) `HuberLoss` защищает от мелких неучтённых выбросов; (3) в `T5.4` эти 174 станция-минуты служат реальным бенчмарком детектора Ветки B! | `T0.2`, `T0.3`, `T4.1`, `T5.2`, `T5.4` |
| **R8** | **Утечка маскированных значений (вкл. multi-hop через Self-Attention), IEEE-754 `0 * NaN == NaN` и FP16-краш `-1e9` на Tesla T4:** модель копирует вход в выход или даёт `NaN`/краш в AMP. | Критическая | (1) Безопасное зануление `torch.where(mask, nan_to_num(x, 0.0), 0.0)` до первого слоя + в Ветке B лосс строго по `M_target` (`obs_mask == False` ДО `SpatialSetEncoder`, `w_obs=0.0`); (2) маскированный `softmax` строго в `float32` (без `-1e9` в `fp16`!) с защитой `torch.where(all_forbidden, 0.0, attn)`; (3) `attn_forbidden` + `M_obs ∩ M_target == ∅`. | `T1.2`, `T2.2`, `T3.1` (`NaN`/fp16 тест), `T3.3` (`+1e6` → `diff < 1e-6`) |
| **R9** | **Провал `1D-CNN` на пропусках и утечка будущего через `GroupNorm` (Фаза 4):** `GroupNorm` смешивает всю ось `W`, `norm_sum > 0` на Winograd-шуме `1e-7` взрывает выход, а `Linear` bias протекает на пустой станции. | Критическая | (1) `MaskedConv1d(bias=False)` с `torch.where`/`nan_to_num`, порогом `norm_sum > 0.5` и `clamp(norm_sum, min=1.0)`; (2) покадровый `ChannelLayerNorm(64) * m_out` вместо `GroupNorm`; (3) прямой residual-скип `ΔB̃(t)` в `SpatialSetEncoder(d_in=105)`; (4) внутренний гейт `has_any = (m_step.sum(-1, keepdim=True) > 0.5)` гарантирует строгий `0.0` для пустой станции. | `T4.1`: causal leakage (`diff==0`), 10/30-мин пропуски, `NaN` и пустая станция |
| **R10** | **Разреженность канадских станций ломает наивную маску `M3 (500 км)`:** у всех 12 станций Канады `0` соседей в 500 км (`d_NN = 626..1330 км`). | Средняя | (1) Раздельные отчёты `M3_500km` (европейские кластеры `≥ 2` ст.) и `M3_1000km` (Европа + Канада); (2) в `train`-маскировании `M3` при `< 2` станциях в круге — фолбэк на `k=3` ближайших соседа. | `T1.2`: проверка покрытия якорей и размера кластеров `M3` |
| **R11** | **Неполный `OMNI` (пустые `SYM-H/PC/Kp`, нет эп. 3–4), инверсия знака $E_y$ и сдвиг `SOLAR-1 (L1 → BSN)`:** падение метрик или перевёрнутые корреляции. | Средняя | (1) NaN-safe корреляции со статусами `unavailable_all_nan` и `unavailable_in_test_fold` + агрегация `OMNI` по покрытым фолдам (`2/4`) в `T6.1`; (2) точная формула **`ey = +vx_gse * bz_gsm * 1e-3`** с проверкой `corr > 0.75` против `OMNI electric_field_mVm` в `T0.2` (факт: `r = +0.870`); (3) баллистический сдвиг `sw_solar1_bsn` + лаг-скан `±90 мин` в `T5.1`; (4) при догрузке OMNI в `T0.5` слияние строго по `timestamp_utc`. | `T0.2`, `T0.5`, `T1.3`, `T5.1`, `T6.1` |
| **R12** | **Несовместимость обученных моделей с будущим экспортом в `ONNX`:** неподдерживаемые операции, data-dependent `if`, 0-width тензоры или фиксированное `N_obs`. | Средняя | (1) Строгие тензорные `forward`-сигнатуры без `dict` и data-dependent `if`; (2) динамические оси `B`, `N_obs`, `N_q` в `onnx_export_spec.json`; (3) тест экспорта opset 17 и паритета с `onnxruntime` (`< 1e-4`) в `T6.3`. | `T6.3`: `test_onnx_export.py` |

---

### Краткая карта порядка исполнения для последующих агентов

1. **Локально (CPU):** `T0.1 → T0.2 → T0.3 → T0.4 → T0.5` *(Фундамент данных готов)*
2. **Локально (CPU):** `T1.1 → T1.2 → T1.3 → T1.4` *(Бейзлайны и протокол зафиксированы)*
3. **Локально (CPU):** `T2.1 → T2.2 → T2.3` *(Уровень 1 DL и «цена сжатия» измерены)*
4. **Код + smoke локально → полный прогон в Colab/Kaggle (T4):** `T3.1 → T3.2 → T3.3 → T3.4` *(Spatial Set Transformers + Zero-Shot)*
5. **Код + smoke локально → полный прогон в Colab/Kaggle (T4):** `T4.1 → T4.2 → T4.3` *(Temporal 1D-CNN + Go/No-Go для Ветки A)*
6. **Локально (CPU):** `T5.1 → T5.2 → T5.3 → T5.4 → T5.5` *(Научные эксперименты: физика `z(t)`, нейро-ULF-индекс, `Stokes`/`WTC`, детектор аномалий, синтез A+B)*
7. **Локально (CPU):** `T6.1 → T6.2 → T6.3` *(Сводные таблицы, графики 300 DPI/SVG, ONNX-валидация)*
8. **После получения расширенных данных от соруководителя:** заполнить `dataset_extended_template.yaml` + `events_template.yaml` → запустить `T0.5` (`target_kp_quiet_night` + `block_temporal` + `use_doy=True`) → повторить обучение Фаз 1–4 на GPU T4 и перезапустить Фазы 5–6 без переписывания кода.
