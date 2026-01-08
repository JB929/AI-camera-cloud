# ST-GCN + LSTM Hybrid Action Recognition

## 1. Prepare data
Convert raw `data/cctv_raw/*` sequences into clean `(T,17,3)`:

```bash
python3 src/core/stgcn/prepare_data.py --data data/cctv_raw --out data_ready

