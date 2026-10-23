# Кластеризация AF3 (multi-seed) против single-seed

Кластеризует модели AF3 каждой комплексообразующей пары по RMSD антигена (HDBSCAN по
матрице RMSD после Kabsch-совмещения по фреймворку) и отвечает на вопрос: даёт ли выбор
лучшего кластера из 500 сидов больше, чем один запуск с одним сидом — «вслепую» (по среднему
ranking_score) и «в оркуле» (по среднему DockQ). Итог — HTML-отчёт на каждый комплекс и сводная страница.

## Пайплайн (`run_clustering.sh` прогоняет его для каждого комплекса)

`parser(multi)` → `matrix` → `cluster` → `dockq_conf` → `parser(single)` → `knn_assign`
→ `collect` → `analysis` → `plots_upd` → `plots` → `build_report`; затем один раз для всех —
`build_index`. Шаги 4, 6, 10 в обвязке — «мягкие» (сбой только предупреждает).

Поток данных (⭧ = точка входа с CLI, ▫ = модуль-хелпер, тоже импортируется):

1. ▫ `parser.py` — читает `<root>/<pdb>/seed-<s>_sample-<k>/*_model.cif` (gemmi) + спецификацию
   остатков ANARCI → `<pdb>_ca_<run>.npz` (CA-координаты групп fit/cdr3/anchor/ag) в
   `/mnt/f0771a75-.../clustering`. **Блока `__main__` нет** — запускать через `python -c` (см. ниже).
2. ▫ `matrix.py` — Kabsch-совмещение по фреймворку, RMSD антигена с pLDDT-маской →
   `<pdb>_matrixA.npy`, `_agmask.npy`, `_frame.npz`.
3. ▫ `cluster.py` — HDBSCAN (precomputed, eom, min_cluster_size=15) → `_labels.tsv`, `_clusters.tsv`.
4. ⭧ `dockq_conf.py` — объединяет метки с `*_summary_confidences.json` + DockQ-таблицей интерфейсов
   (тяжёлые цепи через `anarci_residues_ref.tsv`) → `_models_eval.tsv`, `_clusters_eval.tsv`.
5. ▫ `knn_assign.py` — относит 5 single-seed моделей к кластерам (kNN-голосование + порог
   расстояния 95% leave-one-out) → `<pdb>_single_assign.tsv`.
6. ⭧ `collect.py` — таблица по моделям (DockQ, ranking, iptm, ptm, средний pLDDT) для обоих деревьев →
   `models_<multi|single>_<pdb>.tsv` в `/mnt/1857e392-.../results/reports/`.
7. ⭧ `analysis.py` — метрики по кластерам, лучшие кластеры (dockq/rank/plddt), дельты single-vs-multi
   → `stats_<pdb>.json`, `multi_/single_/clusters_<pdb>.tsv`, `single_dist_<pdb>.npy`, `summary.tsv`.
8. ⭧ `plots_upd.py` — plotly-фигуры (MDS-вложение, расстояние до медоида, ящики DockQ,
   rank-vs-DockQ, размеры кластеров); `__main__` пишет `preview_<pdb>.html`; `build_report.py`
   переиспользует `complex_figures()`.
9. ⭧ `plots.py` — статичные PNG (MDS, тепловая карта RMSD, DockQ) → `/mnt/f0771a75-.../clustering/`.
10. ⭧ `build_report.py` — `<pdb>.html` в `results/reports/html/`.
11. ⭧ `build_index.py` — сводная `html/index.html` + `report_summary.tsv` по всем `stats_*.json`.

Вход: деревья фолдинга `/mnt/f0771a75-.../{500-seeds,single-seed}/<pdb>/seed-<s>_sample-<k>/`
(`*_model.cif`, `*_confidences.json`, `*_summary_confidences.json`); спецификации в
`/mnt/1857e392-.../results/reports/`: `anarci_residues.tsv` (нужна нумерация ANARCI),
`anarci_residues_ref.tsv`, DockQ-таблицы интерфейсов. Зависимости: numpy/pandas/scipy/sklearn/gemmi/plotly/matplotlib.

## Запуск

**cwd должен быть этой каталог** — модули импортируют друг друга (`from parser import OUT_DIR` и
т.д.), а `parser.py` добавляет `/mnt/1857e392-.../scripts` в `sys.path` ради `model_cif_to_fasta`.

### run_clustering.sh (полная цепочка)
```bash
cd /mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/scripts/clustering
./run_clustering.sh -w 8 8jow 8ker     # без pdb-аргументов = все комплексы из 500-seeds
```
Флаги: `-w N` — воркеры парсера (по умолч. 1), `-t 2h` — таймаут шага, `-f` — пересобрать
готовые отчёты, `-h` — справка. Переменные: `DOCKQ_MULTI` / `DOCKQ_SINGLE` переопределяют
DockQ-таблицы (умолч.: `dockq_*_75comp_all5_interfaces_20261019.tsv`); экспортируются python-шагам
как `DOCKQ_500` / `DOCKQ_SINGLE`. Внутри: `PY=/opt/conda/bin/python3.10`. Логи: `/tmp/clustering_logs/<pdb>.log`.

### Отдельные скрипты (везде `/opt/conda/bin/python3.10`, cwd = этот каталог)
```bash
# 1) парсинг моделей (CLI нет; вызов как в shell-скрипте):
$PY -c "from parser import parse_complex; parse_complex('/mnt/f0771a75-.../500-seeds','8jow','<...>/anarci_residues.tsv',workers=8)"
# 2,3,5) argparse нет; __main__ запускает жёстко заданный 8jow. Функции: matrix_A(pdb,tsv,run=),
#        cluster_complex(pdb,...), assign_single(pdb)
# 4) dockq_conf.py [pdb_id] --tag --root --dockq --ref --cluster-dir --out   (умолч. pdb 8jow)
# 6) collect.py [pdb_ids...] --workers 8 --force      (умолч.: комплексы в обоих деревьях)
# 7) analysis.py [pdb_ids...]                          (умолч.: оба дерева; пишет summary.tsv)
# 8) plots_upd.py [pdb_id]                             (умолч. 8jow -> preview_<pdb>.html)
# 9) plots.py [pdb_id]                                 (умолч. 8jow -> 3 PNG)
# 10) build_report.py [pdb_ids...]                     (умолч.: все найденные stats_*.json)
# 11) build_index.py                                   (без аргументов; собирает все stats_*.json)
```

Примечания: в докстрингах `parser.py`/`matrix.py` упоминаются версионированные имена с `<n>` и
CLI у парсера — фактически код пишет простые имена `<pdb>_...`; `analysis.py`/`plots_upd.py`
принимают и те и другие, если версионированные файлы есть. `build_report.py` вшивает git-хэш
из этого каталога (он является git-репозиторием).
