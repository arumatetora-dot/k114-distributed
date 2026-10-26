name: K114 Distributed Sieve Cluster

on:
  workflow_dispatch: # 手動で「Run workflow」ボタンを押して一斉起動

jobs:
  distributed-search:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        # 10台の独立したクラウドVMを同時にスピンアップ
        worker_id: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    steps:
      - name: リポジトリのチェックアウト
        uses: actions/checkout@v4

      - name: Python 3.12 のセットアップ
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: 数論・JIT依存パッケージのインストール
        run: |
          python -m pip install --upgrade pip
          pip install numpy numba gmpy2

      - name: 分散探索 Worker ${{ matrix.worker_id }} の実行 (10分間)
        run: |
          # 2 vCPU をフルに使い切る設定
          export NUMBA_NUM_THREADS=2
          python k114_worker.py --worker_id ${{ matrix.worker_id }} --total_workers 10 --time_limit 600

      - name: 完全解が発見された場合のログ保存
        if: always()
        run: |
          mkdir -p results
          # 発見ファイルが存在する場合のみ格納
          cp -f K114_WORKER_*.txt results/ 2>/dev/null || true

      - name: 探索ログのアーティファクト出力
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: worker-log-${{ matrix.worker_id }}
          path: results/
