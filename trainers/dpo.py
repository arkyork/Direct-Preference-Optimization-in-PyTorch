from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

from tqdm import tqdm

class DPO:
    def __init__(self, model_id, model, ref_model, beta):
        self.beta = beta
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        log_pi_theta_w,log_pi_theta_l = log_prob(outputs_w.logits, y_w,outputs_l.logits, y_l)

        # 参照モデルの出力
        ref_outputs_w = self.ref_model(input_ids=y_w,attention_mask=y_w_attention)
        ref_outputs_l = self.ref_model(input_ids=y_l,attention_mask=y_l_attention)
        log_pi_ref_w,log_pi_ref_l = log_prob(ref_outputs_w.logits, y_w,ref_outputs_l.logits, y_l)
        # DPOの目的関数
        log_w = self.beta * (log_pi_theta_w - log_pi_ref_w)
        log_l = self.beta * (log_pi_theta_l - log_pi_ref_l)
        sigma = torch.sigmoid(log_w - log_l)
        loss = -torch.mean(torch.log(sigma))
        return loss
    def tokenize_dataset(self,batch):
        # Tokenize prompt, chosen, and rejected
        prompt_token = self.tokenizer(batch['prompt'], padding=True, return_tensors="pt")
        chosen_token = self.tokenizer(batch['chosen'], padding=True, return_tensors="pt")
        rejected_token = self.tokenizer(batch['rejected'], padding=True, return_tensors="pt")
        
        # Add eos_token
        chosen_token['input_ids'] = torch.cat(
            [chosen_token['input_ids'], torch.tensor([[self.tokenizer.eos_token_id]])], dim=1
        )
        chosen_token['attention_mask'] = torch.cat(
            [chosen_token['attention_mask'], torch.tensor([[1]])], dim=1
        )

        rejected_token['input_ids'] = torch.cat(
            [rejected_token['input_ids'], torch.tensor([[self.tokenizer.eos_token_id]])], dim=1
        )
        rejected_token['attention_mask'] = torch.cat(
            [rejected_token['attention_mask'], torch.tensor([[1]])], dim=1
        )

        # Concatenate prompt with chosen and rejected
        batch['chosen_tokenizer'] = {
            'input_ids': torch.cat([prompt_token['input_ids'], chosen_token['input_ids']], dim=1),
            'attention_mask': torch.cat([prompt_token['attention_mask'], chosen_token['attention_mask']], dim=1),
        }
        batch['rejected_tokenizer'] = {
            'input_ids': torch.cat([prompt_token['input_ids'], rejected_token['input_ids']], dim=1),
            'attention_mask': torch.cat([prompt_token['attention_mask'], rejected_token['attention_mask']], dim=1),
        }
        batch["chosen"]=batch["prompt"]+batch["chosen"]
        batch["rejected"]=batch["prompt"]+batch["rejected"]

        return batch
    def train(self, dataset, optimizer, num_epochs=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.ref_model.to(device)
        
        self.model.train()
        self.ref_model.eval()

        dataset=dataset.map(self.tokenize_dataset)
        # 訓練ループ
        for epoch in tqdm(range(num_epochs)):
            for batch in tqdm(dataset):

                optimizer.zero_grad()
                
                y_w = torch.tensor(batch['chosen_tokenizer']['input_ids']).to(device)
                y_w_attention = torch.tensor(batch['chosen_tokenizer']['attention_mask']).to(device)
                y_l = torch.tensor(batch['rejected_tokenizer']['input_ids']).to(device)
                y_l_attention = torch.tensor(batch['rejected_tokenizer']['attention_mask']).to(device)

                # 順伝播と損失計算
                loss = self.compute_loss(y_w,y_w_attention, y_l,y_l_attention)

                # 逆伝播と最適化

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        return loss.item()
