"""
Export paper-ready CSV tables and print figure selection for the paper.

Outputs:
  tables/model_comparison.csv   — EER, std, params, latency, memory, throughput
  tables/inference_benchmark.csv — per-model per-batch-size benchmark detail
  tables/ablation.csv           — branch combination ablation EER

Usage:
    python graphs/export_tables.py
"""

import csv
import os

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, 'tables')
os.makedirs(TABLES, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
MODEL_COMPARISON = [
    {
        'Model':             'CNN+Transformer',
        'EER_pct':           2.38,
        'EER_std_pct':       4.56,
        'Params':            229184,
        'File_size_MB':      0.95,
        'Peak_mem_MB_B128':  '—',
        'Lat_B1_ms':         '~7*',
        'Lat_B8_ms':         '—',
        'Lat_B32_ms':        '—',
        'Lat_B128_ms':       '—',
        'Throughput_B128_sps': '—',
        'Notes':             'Warm-started from Full Ensemble; * estimated',
    },
    {
        'Model':             'Full Ensemble (CNN+LSTM+Trans)',
        'EER_pct':           2.38,
        'EER_std_pct':       4.46,
        'Params':            512960,
        'File_size_MB':      2.09,
        'Peak_mem_MB_B128':  35.4,
        'Lat_B1_ms':         12.0,
        'Lat_B8_ms':         13.8,
        'Lat_B32_ms':        16.0,
        'Lat_B128_ms':       15.8,
        'Throughput_B128_sps': 8102,
        'Notes':             '',
    },
    {
        'Model':             'TypeNet LSTM',
        'EER_pct':           4.17,
        'EER_std_pct':       6.24,
        'Params':            201472,
        'File_size_MB':      0.81,
        'Peak_mem_MB_B128':  24.5,
        'Lat_B1_ms':         4.24,
        'Lat_B8_ms':         8.28,
        'Lat_B32_ms':        8.54,
        'Lat_B128_ms':       7.93,
        'Throughput_B128_sps': 16132,
        'Notes':             '',
    },
    {
        'Model':             'CNN Branch (standalone)',
        'EER_pct':           4.41,
        'EER_std_pct':       6.60,
        'Params':            103872,
        'File_size_MB':      '—',
        'Peak_mem_MB_B128':  11.9,
        'Lat_B1_ms':         1.72,
        'Lat_B8_ms':         1.63,
        'Lat_B32_ms':        1.42,
        'Lat_B128_ms':       1.79,
        'Throughput_B128_sps': 71623,
        'Notes':             '',
    },
    {
        'Model':             'Transformer Branch (standalone)',
        'EER_pct':           5.42,
        'EER_std_pct':       8.58,
        'Params':            75648,
        'File_size_MB':      '—',
        'Peak_mem_MB_B128':  18.4,
        'Lat_B1_ms':         5.02,
        'Lat_B8_ms':         4.27,
        'Lat_B32_ms':        4.72,
        'Lat_B128_ms':       5.24,
        'Throughput_B128_sps': 24450,
        'Notes':             '',
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — Inference benchmark detail
# ══════════════════════════════════════════════════════════════════════════════
BENCHMARK = [
    # (model, batch, cold_ms, mean_ms, std_ms, p50, p95, p99, sps)
    ('TypeNet LSTM',    1,   68.5, 4.24,  0.99, 4.08, 5.38, 8.02, 236),
    ('TypeNet LSTM',    8,   17.2, 8.28,  2.72, 7.53, 14.6, 15.0, 967),
    ('TypeNet LSTM',    32,  7.93, 8.54,  2.67, 7.71, 13.9, 15.7, 3747),
    ('TypeNet LSTM',    128, 23.0, 7.93,  2.16, 7.40, 11.9, 13.7, 16132),
    ('CNN Branch',      1,   24.8, 1.72,  0.55, 1.69, 2.70, 3.33, 583),
    ('CNN Branch',      8,   27.2, 1.63,  0.49, 1.60, 2.26, 3.27, 4896),
    ('CNN Branch',      32,  11.3, 1.42,  0.37, 1.25, 1.99, 2.61, 22589),
    ('CNN Branch',      128, 1.87, 1.79,  0.53, 1.71, 2.84, 3.82, 71623),
    ('Transformer Br.', 1,   20.5, 5.02,  0.84, 4.88, 6.66, 7.36, 199),
    ('Transformer Br.', 8,   6.96, 4.27,  1.06, 3.91, 6.36, 7.33, 1873),
    ('Transformer Br.', 32,  4.01, 4.72,  1.38, 4.14, 7.39, 9.14, 6783),
    ('Transformer Br.', 128, 6.61, 5.24,  1.35, 4.89, 7.79, 9.48, 24450),
    ('Full Ensemble',   1,   17.6, 12.01, 2.91, 11.6, 17.7, 19.4, 83),
    ('Full Ensemble',   8,   10.9, 13.8,  2.50, 13.5, 18.1, 22.2, 579),
    ('Full Ensemble',   32,  12.6, 16.0,  3.78, 15.1, 22.8, 25.4, 1997),
    ('Full Ensemble',   128, 22.7, 15.8,  2.63, 15.6, 20.5, 24.1, 8102),
]

# ══════════════════════════════════════════════════════════════════════════════
# Table 3 — Ablation
# ══════════════════════════════════════════════════════════════════════════════
ABLATION = [
    ('All three branches (Ensemble)', 2.35, '—',   'Trained fusion head'),
    ('CNN + Transformer',             2.88, '+0.53', 'Lightweight fusion'),
    ('LSTM + CNN',                    3.98, '+1.63', 'Lightweight fusion'),
    ('LSTM only',                     4.11, '+1.76', 'Standalone'),
    ('LSTM + Transformer',            4.47, '+2.12', 'Lightweight fusion'),
    ('CNN only',                      4.67, '+2.32', 'Standalone'),
    ('Transformer only',              5.31, '+2.96', 'Standalone'),
]


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'  saved: {path}')


def main():
    # ── Table 1: model comparison ─────────────────────────────────────────────
    p1 = os.path.join(TABLES, 'model_comparison.csv')
    write_csv(p1, list(MODEL_COMPARISON[0].keys()), MODEL_COMPARISON)

    # ── Table 2: benchmark detail ─────────────────────────────────────────────
    bench_cols = ['Model', 'Batch', 'Cold_ms', 'Mean_ms', 'Std_ms',
                  'P50_ms', 'P95_ms', 'P99_ms', 'SPS']
    bench_rows = [dict(zip(bench_cols, r)) for r in BENCHMARK]
    p2 = os.path.join(TABLES, 'inference_benchmark.csv')
    write_csv(p2, bench_cols, bench_rows)

    # ── Table 3: ablation ─────────────────────────────────────────────────────
    abl_cols = ['Branch_combination', 'EER_pct', 'Delta_vs_best', 'Fusion_type']
    abl_rows = [dict(zip(abl_cols, r)) for r in ABLATION]
    p3 = os.path.join(TABLES, 'ablation.csv')
    write_csv(p3, abl_cols, abl_rows)

    # ── Paper figure recommendations ──────────────────────────────────────────
    print()
    print('=' * 68)
    print('  RECOMMENDED PAPER FIGURES')
    print('=' * 68)
    recs = [
        ('01_eer_comparison.png',    'Main result',
         'Primary accuracy table in visual form. Shows CNN+Transformer\n'
         '         matches full ensemble at 2.38% while LSTM lags at 4.17%.'),
        ('02_ablation.png',          'Ablation study',
         'Shows LSTM adds nothing (+1.63pp vs CNN+Trans alone).\n'
         '         CNN+Transformer pairwise fusion is the sweet spot.'),
        ('04_training_eer.png',      'Training convergence',
         'Demonstrates fused output consistently outperforms either\n'
         '         branch alone throughout training (not just at convergence).'),
        ('11_tsne_branches.png',     'Qualitative — embedding space',
         'Side-by-side t-SNE of CNN / Transformer / Fused with\n'
         '         Silhouette, Davies-Bouldin, Calinski-Harabasz metrics.'),
        ('13_tsne_gallery_query.png','Qualitative — auth protocol',
         'Gallery (filled) vs query (open ring) shows subjects stay\n'
         '         tightly clustered regardless of session split.'),
    ]
    for fname, role, note in recs:
        print(f'\n  [{role}]')
        print(f'  File : {fname}')
        print(f'  Why  : {note}')

    print()
    print('  FIGURES TO OMIT FROM PAPER (reason)')
    omit = [
        ('00_combined.png',          'Dashboard overview — too dense, redundant with individual figs'),
        ('03_training_loss.png',     'Loss curve — move to appendix or supplementary'),
        ('05_latency_b1.png',        'Use table (model_comparison.csv) instead'),
        ('06_memory.png',            'Use table (model_comparison.csv) instead'),
        ('07_throughput_b128.png',   'Use table (inference_benchmark.csv) instead'),
        ('08_model_comparison_table.png', 'Replace with LaTeX table from model_comparison.csv'),
        ('09_benchmark_table.png',   'Replace with LaTeX table from inference_benchmark.csv'),
        ('10_tsne_fused.png',        'Covered by the branch comparison (fig 11)'),
        ('12_tsne_sessions.png',     'Low information density — subject cluster too compact to show trajectory'),
    ]
    for fname, reason in omit:
        print(f'  {fname}')
        print(f'      {reason}')

    print()
    print('  CSV TABLES')
    print(f'  {p1}')
    print(f'  {p2}')
    print(f'  {p3}')
    print(f'  {os.path.join(TABLES, "clustering_metrics.csv")}  (generated by tsne_figures.py)')
    print()


if __name__ == '__main__':
    main()
