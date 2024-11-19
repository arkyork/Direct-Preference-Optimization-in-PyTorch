## DPO Loss Function

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \cdot \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

## $\log\pi(y|x)$の計算方法

### outputs.logits

outputs.logits の形状

(batch_size, sequence_length, vocab_size)

### 各時間tにおける次のトークンの推測$x_{t+1}$

outputs.logits[:, t, :]

### 


