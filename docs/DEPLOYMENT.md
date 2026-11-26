# Руководство по сборке и автономному запуску GeoMag Analyzer

Система поставляется как **автономное настольное приложение Windows x64** (`PyInstaller --onedir` + `pywebview Edge WebView2` с автоматическим фолбэком `--browser`), не требующее установленного Python, прав администратора и доступа к интернету.

---

## 1. Портативная версия (`GeoMagAnalyzer_Portable_Win64.zip`)

1. Распакуйте `GeoMagAnalyzer_Portable_Win64.zip` в любую пользовательскую папку (включая путь с пробелами или USB-накопитель; права администратора не требуются).
2. Запустите `GeoMagAnalyzer.exe`:
   - при наличии системного компонента **Microsoft Edge WebView2 Runtime** открывается нативное окно приложения;
   - при отсутствии WebView2 (или при передаче ключа `--browser` / запуске `GeoMagAnalyzer_Browser.bat`) автоматически запускается системный браузер по умолчанию с адресом `http://127.0.0.1:<port>/`.
3. **Разделение статики и пользовательских данных:**
   - папка `_internal/data/processed/` (`catalog.sqlite`, `.parquet`) и `_internal/frontend/dist/` (включая локальные шрифты и `world-110m.topojson`) используются строго **read-only**;
   - рядом с `GeoMagAnalyzer.exe` в портативной сборке лежит маркер `portable.mode`, поэтому изменяемые `runtime.sqlite`, `user_cache/` и `logs/` записываются в `./user_data/` (при отсутствии прав записи выполняется автоматический фолбэк в `%LOCALAPPDATA%\GeoMagAnalyzer`).
4. **Остановка сервера в режиме `--browser` (`console=False`):** используйте кнопку **«Exit»** в верхней панели интерфейса (`POST /api/shutdown`). При закрытии настольного окна или нажатии `Exit` все воркеры `ProcessPoolExecutor` и `SyncManager` завершаются штатно.

---

## 2. Сборка дистрибутива и запуск из исходного кода (для разработчика)

### Предварительные требования на сборочной машине Windows x64
- Python `>=3.11` и виртуальное окружение `.venv` с установленным пакетом `desktop`:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -e "backend[dev,desktop]"
  ```
- Node.js `18+` и зависимости в `frontend/` (`npm ci`).
- Предварительно собранный ETL-кэш `data/processed/` (`.\.venv\Scripts\python.exe -m geomag_etl.build`).

### Запуск из исходного кода в браузере
Для локального запуска dev/release-сборки достаточно выполнить `start_browser.bat` (вызывает `backend/launcher.py --browser --port 8000`) или `setup_and_start.bat` при первом разворачивании `.venv`.

### Команда полной сборки Portable ZIP
```powershell
.\.venv\Scripts\python.exe packaging\build_windows.py
```
Скрипт выполняет 3 шага:
1. Проверяет наличие `packaging/geomag_analyzer.ico` и целостность `data/processed/` (`catalog.sqlite`, 4 штормовых файла `geomag/*.parquet`, `solar1_l1_1min.parquet`, `omni_bsn_1min.parquet`, `oulu_nm_1min.parquet`), затем собирает production SPA `frontend/dist/`.
2. Собирает `--onedir` бандл `dist/GeoMagAnalyzer/` по спецификации `packaging/geomag_analyzer.spec` (`console=False`, `upx=False`, `multiprocessing.freeze_support()`, без `portable.mode`).
3. Упаковывает `dist/GeoMagAnalyzer_Portable_Win64.zip` (с `portable.mode`, `GeoMagAnalyzer_Browser.bat`, `README_PORTABLE.txt`) и формирует `dist/SHA256SUMS.txt`.

Для проверки собранного архива в изолированной среде без `PYTHONPATH` и установленного Python:
```powershell
.\.venv\Scripts\python.exe packaging\e2e_verify.py
```

---

## 3. Чеклист чистой офлайн-приёмки на Windows 10/11

1. Перенести `GeoMagAnalyzer_Portable_Win64.zip` на чистую машину Windows 10/11 **без Python, без прав администратора и с отключённой сетью** и распаковать.
2. Убедиться, что запуск `GeoMagAnalyzer.exe` проходит без предупреждений блокировки и открывает окно `pywebview` (или системный браузер при `--browser` / отсутствии WebView2).
3. Проверить загрузку 49 магнитометрических станций и карты `D3-geo` (`world-110m.topojson`) при полностью отключённом интернете (0 внешних сетевых запросов).
4. Запустить задачу Tier 3 (`XWT/WTC`), проверить прогресс и отмену, затем довести задачу до завершения и убедиться в мгновенном повторном `cache_hit` из `user_cache/` и обновлении `runtime.sqlite`.
5. Экспортировать график/карту в `SVG` и таблицу в `CSV`, затем закрыть приложение кнопкой `Exit` (или закрытием окна `pywebview`) и убедиться в отсутствии зависших процессов в диспетчере задач.
