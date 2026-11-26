# Руководство по сборке, установке и развёртыванию GeoMag Analyzer

Система поддерживает два режима эксплуатации на базе единой кодовой базы (`FastAPI + geomag_core + React/TypeScript`):
1. **Автономное настольное приложение Windows x64** (`PyInstaller --onedir` + `pywebview Edge WebView2` + автоматический фолбэк `--browser`), не требующее установленного Python, прав администратора и доступа к интернету.
2. **Веб-сервис (`Headless Server Mode`)** под `uvicorn --workers 1` за обратным прокси `Nginx` или `Caddy`.

---

## 1. Режим настольного приложения Windows (для конечных пользователей)

### 1.1. Портативная версия (`GeoMagAnalyzer_Portable_Win64.zip`)
1. Распакуйте `GeoMagAnalyzer_Portable_Win64.zip` в любую пользовательскую папку (включая путь с пробелами или USB-накопитель; права администратора не требуются).
2. Запустите `GeoMagAnalyzer.exe`:
   - при наличии системного компонента **Microsoft Edge WebView2 Runtime** открывается нативное окно приложения;
   - при отсутствии WebView2 (или при передаче ключа `--browser` / запуске `GeoMagAnalyzer_Browser.bat`) автоматически запускается системный браузер по умолчанию с адресом `http://127.0.0.1:<port>/?token=<bootstrap_token>` и иконкой управления в системном трее Windows (`Open GeoMag Analyzer`, `Stop Server / Exit`).
3. **Разделение статики и пользовательских данных:**
   - папка `_internal/data/processed/` (`catalog.sqlite`, `.parquet`) и `_internal/frontend/dist/` (включая локальные шрифты и `world-110m.topojson`) используются строго **read-only**;
   - рядом с `GeoMagAnalyzer.exe` в портативной сборке лежит маркер `portable.mode`, поэтому изменяемые `runtime.sqlite`, `user_cache/` и `logs/` записываются в `./user_data/` (при отсутствии прав записи выполняется автоматический фолбэк в `%LOCALAPPDATA%\GeoMagAnalyzer`).
4. **Остановка сервера в режиме `--browser` (`console=False`):** используйте кнопку **«Exit»** в верхней панели интерфейса (`POST /api/shutdown` с `X-Session-Token`) либо пункт **«Stop Server / Exit»** в контекстном меню иконки системного трея. При закрытии окна все воркеры `ProcessPoolExecutor` и `SyncManager` завершаются штатно.

### 1.2. Пользовательский установщик (`GeoMagAnalyzer_Setup.exe`, Inno Setup)
1. Запустите `GeoMagAnalyzer_Setup.exe` — установка выполняется с правами обычного пользователя (`PrivilegesRequired=lowest`, без запроса UAC) в `%LOCALAPPDATA%\Programs\GeoMagAnalyzer` с созданием ярлыков в меню «Пуск» и на рабочем столе.
2. В установленной версии файл `portable.mode` отсутствует, поэтому `runtime.sqlite`, `user_cache/`, `webview2_data/` и логи всегда хранятся отдельно в `%LOCALAPPDATA%\GeoMagAnalyzer\`.

---

## 2. Сборка дистрибутивов из исходного кода (для разработчика)

### Предварительные требования на сборочной машине Windows x64
- Python `>=3.11` и виртуальное окружение `.venv` с установленным пакетом `desktop`:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -e "backend[dev,desktop]"
  ```
- Node.js `18+` и зависимости в `frontend/` (`npm ci`).
- Предварительно собранный ETL-кэш `data/processed/` (`.\.venv\Scripts\python.exe -m geomag_etl.build`).
- [Inno Setup 6/7](https://jrsoftware.org/isinfo.php) (`ISCC.exe`) для компиляции `GeoMagAnalyzer_Setup.exe`.

### Команда полной сборки
```powershell
.\.venv\Scripts\python.exe packaging\build_windows.py
```
Скрипт выполняет 4 шага:
1. Генерирует `packaging/geomag_analyzer.ico` (при отсутствии) и собирает production SPA `frontend/dist/`.
2. Проверяет целостность `data/processed/` (`catalog.sqlite`, `manifest.parquet`, 4 штормовых файла `geomag/*.parquet`, `solar1_l1_1min.parquet`, `omni_bsn_1min.parquet`, `oulu_nm_1min.parquet`).
3. Собирает `--onedir` бандл `dist/GeoMagAnalyzer/` по спецификации `packaging/geomag_analyzer.spec` (`console=False`, `upx=False`, `multiprocessing.freeze_support()`, без `portable.mode`).
4. Упаковывает `dist/GeoMagAnalyzer_Portable_Win64.zip` (с `portable.mode`, `GeoMagAnalyzer_Browser.bat`, `README_PORTABLE.txt`), компилирует `dist/GeoMagAnalyzer_Setup.exe` через `ISCC.exe` и формирует `dist/SHA256SUMS.txt`.

Для проверки собранных артефактов в изолированной среде без `PYTHONPATH` и установленного Python:
```powershell
.\.venv\Scripts\python.exe packaging\e2e_verify.py
```

---

## 3. Развёртывание в режиме веб-сервиса (`Headless Server Mode`)

### 3.1. Архитектурные требования
- **Обязателен ровно один API-процесс (`--workers 1`)**: `JobManager` хранит пул воркеров `ProcessPoolExecutor`, события отмены `multiprocessing.Manager().Event()` и подписки WebSocket/SSE в памяти процесса.
- **Изоляция потоков BLAS:** в окружении сервиса и внутри каждого воркера выставляются `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- **Квоты Tier 3:** задаются переменными `GEOMAG_MAX_WORKERS`, `GEOMAG_MAX_MC_SURROGATES` (по умолчанию `<= 1000`) и `GEOMAG_MAX_ACTIVE_JOBS`.
- **CSRF и обратный прокси:**
  - задайте `GEOMAG_ALLOWED_HOSTS` (доменное имя сервера) и `GEOMAG_ALLOWED_ORIGINS` (например `https://geomag.example.edu`);
  - задайте `GEOMAG_COOKIE_SECURE=1` при терминации TLS на `Nginx`/`Caddy` и передавайте `X-Forwarded-Proto $scheme`;
  - в headless-режиме `POST /api/shutdown` отключён (возвращает HTTP `501`, в `GET /api/health` поле `can_shutdown = false`).

### 3.2. Запуск под `uvicorn`
Пример настройки переменных окружения приведён в [`packaging/web/geomag-analyzer.env.example`](../packaging/web/geomag-analyzer.env.example), unit-файл systemd — в [`packaging/web/geomag-analyzer.service`](../packaging/web/geomag-analyzer.service):

```bash
uvicorn geomag_api.main:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers
```

### 3.3. Конфигурации обратного прокси
- **Nginx:** [`packaging/web/nginx.conf`](../packaging/web/nginx.conf) — настроен на TLS, проброс `Upgrade`/`Connection` для `/ws/` (`proxy_read_timeout 3600s`) и отключение буферизации (`proxy_buffering off`) для SSE-потока `/api/jobs/*/events`.
- **Caddy:** [`packaging/web/Caddyfile`](../packaging/web/Caddyfile) — автоматический TLS, нативный проброс WebSocket и `flush_interval -1` для SSE `/api/jobs/*/events`.

---

## 4. Чеклист чистой офлайн-приёмки на Windows 10/11

1. Перенести `GeoMagAnalyzer_Portable_Win64.zip` или `GeoMagAnalyzer_Setup.exe` на чистую машину Windows 10/11 **без Python, без прав администратора и с отключённой сетью**.
2. Убедиться, что запуск `GeoMagAnalyzer.exe` проходит без предупреждений блокировки и открывает окно `pywebview` (или системный браузер при `--browser` / отсутствии WebView2).
3. Проверить загрузку 49 магнитометрических станций и карты `D3-geo` (`world-110m.topojson`) при полностью отключённом интернете (0 внешних сетевых запросов).
4. Запустить задачу Tier 3 (`XWT/WTC`), проверить прогресс и отмену, затем довести задачу до завершения и убедиться в мгновенном повторном `cache_hit` из `user_cache/` и обновлении `runtime.sqlite`.
5. Экспортировать график/карту в `SVG` и таблицу в `CSV`, затем закрыть приложение (кнопкой `Exit` или закрытием окна) и убедиться в отсутствии зависших процессов в диспетчере задач.
