import torch
import torch.nn.functional as F
from torch.optim import AdamW

from tqdm import tqdm
class DPO:
    def __init__(self, model, ref_model, beta):
        self.beta = beta
        self.model = model
        self.ref_model = ref_model
        self.optimizer = AdamW(self.model.parameters(), lr=1e-6)
    # 対数確率を計算する関数
    def log_prob(self,chosen_logits,chosen_ids,rejected_logits, rejected_ids):
        ## DPOのlog_Pi.ipynbで動作を確認

        """
            Args:
            chosen_logits (torch.Tensor): 選ばれたシーケンスの logits (batch_size, seq_len, vocab_size)
            chosen_ids (torch.Tensor): 選ばれたシーケンスのラベル IDs (batch_size, seq_len)
            rejected_logits (torch.Tensor): 却下されたシーケンスの logits (batch_size, seq_len, vocab_size)
            rejected_ids (torch.Tensor): 却下されたシーケンスのラベル IDs (batch_size, seq_len)    
        """
        # Chosen の log probabilities を計算
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        chosen_selected_log_probs = torch.gather(
            chosen_log_probs, dim=-1, index=chosen_ids.unsqueeze(-1)  # (batch_size, seq_len, 1)
        ).squeeze(-1)  # (batch_size, seq_len)
        chosen_log_prob = chosen_selected_log_probs.sum(dim=-1)  # (batch_size,)

        # Rejected の log probabilities を計算
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        rejected_selected_log_probs = torch.gather(
            rejected_log_probs, dim=-1, index=rejected_ids.unsqueeze(-1)  # (batch_size, seq_len, 1)
        ).squeeze(-1)  # (batch_size, seq_len)
        rejected_log_prob = rejected_selected_log_probs.sum(dim=-1)  # (batch_size,)
        return chosen_log_prob, rejected_log_prob
    def compute_loss(self, y_w,y_w_attention, y_l,y_l_attention):

        # 現在のポリシーの出力
        outputs_w = self.model(input_ids=y_w,attention_mask=y_w_attention)
        outputs_l = self.model(input_ids=y_l,attention_mask=y_l_attention)
        log_pi_theta_w,log_pi_theta_l = self.log_prob(outputs_w.logits, y_w,outputs_l.logits, y_l)

        # 参照モデルの出力
        ref_outputs_w = self.ref_model(input_ids=y_w,attention_mask=y_w_attention)
        ref_outputs_l = self.ref_model(input_ids=y_l,attention_mask=y_l_attention)
        log_pi_ref_w,log_pi_ref_l = self.log_prob(ref_outputs_w.logits, y_w,ref_outputs_l.logits, y_l)
        # DPOの目的関数
        log_w = (log_pi_theta_w - log_pi_ref_w)
        log_l = (log_pi_theta_l - log_pi_ref_l)
        sigma = -F.logsigmoid(self.beta*(log_w - log_l))

        loss = torch.mean(sigma)
        print(loss)
        return loss
    def train(self, dataset, num_epochs=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(device)
        self.ref_model.to(device)
        
        self.model.train()
        self.ref_model.eval()
        
        for param in self.ref_model.parameters():
            param.requires_grad = False


        # 訓練ループ
        for epoch in tqdm(range(num_epochs)):
            for batch in dataset:

                self.optimizer.zero_grad()
                
                y_w = batch['chosen_tokenizer']['input_ids'].clone().detach().to(device)
                y_w_attention = batch['chosen_tokenizer']['attention_mask'].clone().detach().to(device)
                y_l = batch['rejected_tokenizer']['input_ids'].clone().detach().to(device)
                y_l_attention = batch['rejected_tokenizer']['attention_mask'].clone().detach().to(device)

                # 順伝播と損失計算
                loss = self.compute_loss(y_w,y_w_attention, y_l,y_l_attention)

                # 逆伝播と最適化

                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        return loss.item()
