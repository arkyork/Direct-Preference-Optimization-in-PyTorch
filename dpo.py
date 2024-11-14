from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
# 対数確率を計算する関数
def log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

class DPO:
    def __init__(self, model_id, model, ref_model, beta):
        self.beta = beta
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def compute_loss(self, inputs, y_w, y_l):


        # 現在のポリシーの出力
        outputs_w = self.model(**inputs)
        outputs_l = self.model(**inputs)
        log_pi_theta_w = log_prob(outputs_w.logits, y_w)
        log_pi_theta_l = log_prob(outputs_l.logits, y_l)

        # 参照モデルの出力
        ref_outputs_w = self.ref_model(**inputs)
        ref_outputs_l = self.ref_model(**inputs)
        log_pi_ref_w = log_prob(ref_outputs_w.logits, y_w)
        log_pi_ref_l = log_prob(ref_outputs_l.logits, y_l)

        # DPOの目的関数
        log_w = self.beta * (log_pi_theta_w - log_pi_ref_w)
        log_l = self.beta * (log_pi_theta_l - log_pi_ref_l)
        sigma = torch.sigmoid(log_w - log_l)
        loss = -torch.mean(torch.log(sigma))

        return loss

    def train(self, dataset, optimizer, num_epochs=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.ref_model.to(device)
        
        self.model.train()
        self.ref_model.eval()

        # 訓練ループ
        for epoch in range(num_epochs):
            for batch in dataset:
                inputs = self.tokenizer(batch["prompt"], return_tensors="pt", padding=True).to(device)
                y_w = self.tokenizer(batch["chosen"], return_tensors="pt", padding=True).to(device)
                y_l = self.tokenizer(batch["rejected"], return_tensors="pt", padding=True).to(device)

                # 順伝播と損失計算
                loss = self.compute_loss(inputs, y_w, y_l)

                # 逆伝播と最適化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        return loss.item()
