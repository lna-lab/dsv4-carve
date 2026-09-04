# LNA-CANON — この pack の正典（2026-09-04 更新、Lna-Lab / YUKI with Ken）

## 制式（ケン決裁）
- 2026-09-03: **オオタニ** = この pack を TP8（RTX PRO 2000 16GB ×8: GPU 0,1,2,3,5,7,8,9）、`-c 389120`・4 流・DSpark3・APC ON で給仕。
- **2026-09-04: routed experts の decode カーネルを lna2 に制式認定**（「Lna2 を制式認定します」）。重み・pack・像は同じ。

## 起動（制式、DATA1 で）
```
export EXT_SO=$PWD/vllm-exl3-lab/exllamav3-src/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
export PLUGIN_SRC=$PWD/vllm-exl3-lab/vllm-exl3/src/vllm_exl3
export NATIVE_SO=$PWD/vllm-exl3-v030-port/src/vllm_exl3_c.cpython-312-x86_64-linux-gnu.so
AUX_STREAMS=0 IMAGE=dsv4-dense MOE_KERNEL=lna2 \
NCCL_EXTRA="-e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 -e VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1 -e VLLM_EXL3_MOE_STRICT=1" \
MODEL=$PWD/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6 UTIL=0.97 MAXLEN=389120 BT=512 SEQS=4 \
SPEC='{"method":"dspark","num_speculative_tokens":3}' bash vllm-exl3-lab/serve-dsv4-tp8.sh
```
起動の証: ログに `F3 LNA2 MoE kernel ACTIVE` ×8 rank、`GPU KV cache size: 395,069 tokens`、STRICT=1 で raise なし。

## 実測（2026-09-04、席の門 4、rank0 graph ON）
| | 現行 exllamav3 | **lna2（制式）** |
|---|---|---|
| 1 流 code / en / ja | 86 / 60 / 56 tok/s | **108.6 / 78.6 / 71.1** |
| 4 流 code / en / ja | 125〜179 / 96 / 83（揺れ大） | 193.8 / 146.4 / 129.0 |
| routed カーネル | 232.6 µs/launch | **104.0**（2.24×） |
| decode 総 | 27.46 ms/step | 21.03 |
| KV 窓 | 396,656 tok | 395,069（−0.4%） |
| 166k 針 | 正答、TTFT 167 s | 正答、**119.6 s** |
| ppl（同条件、spec off） | 4.7630 | 4.7647 |
| 受理長 en/ja/code | 2.08/1.86/3.05 | 不変 |

## lna2 とは
常駐 expert-team＋動的 ticket（exllamav3 `exl3_moe` と同じ外側、grid 全体の barrier 無し）に、行数 R_e∈{1,2,4,8}（m≤24）の small-R 内側（`exl3_gemv_kernel` の核）を入れたもの。scratch は device あたり 1 本、2 stream 同時は既定で禁止（`LNA2_CONCURRENT_STREAMS=1`）。机上 parity 171 件（rel ≤1.64e-3、bitwise 再現）、null-decode 変種で 270 GB/s（カードの cold read 天井）に張り付く＝残る天井は trellis 復号。
出典: `vllm-exl3-v030-port/csrc/lna_moe_ticket.cu(.cuh)`, `lna_gemv_core.cuh`, plugin hunk = `vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py`（`get_moe_kernel_backend` / `_apply_lna2_moe`）。公開 = github.com/lna-lab/dsv4-carve `recipe-lna/lna2/`。

## 含めないもの
- `LNA_EXL3_DENSE_GROUP`（密線形の融合 T0/T1）: T0 は席の門合格（密 1.21×、e2e +1〜4%）だが、T1 は共有プールの排他で不動作・4 流悪化（修正中）。制式には入れない。
- F1 初号（`lna`、barrier 型）: 正しさ合格・速度不合格で閉じた。opt-in の実験経路として残す。

## 教訓（09-04）
- 無音の fallback は門を通る。**経路が取られた証明（ACTIVE ログ・カウンタ・strict）を門に**。
- 派生 pack の小ファイルは symlink でなく写す（原本を消したら席が再起動不能一歩手前になった）。
- 「もっともらしい数字」は構成の証拠にならない。mount と env を数える。
