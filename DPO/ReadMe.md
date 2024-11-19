## DPO Loss Function

$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \cdot \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$

## $\log\pi(y|x)$の計算方法

### $\pi(y|x)$の計算式

$\log \pi(y|x) = \displaystyle\sum_{t=1}^{T} \log \pi(y_t | y_{<t})$


### outputs.logits

`outputs.logits` は、モデルの出力であり、形状は `(batch_size, sequence_length, vocab_size)` です。
  - `batch_size`: バッチ内のサンプル数
  - `sequence_length`: 入力シーケンスの長さ
  - `vocab_size`: 語彙のサイズ（ソフトマックスを使ったら、各トークンの確率分布を表す）


### 各時間tにおける次のトークンの推測

次のトークンの確率分布

outputs.logits[:, t, :]


