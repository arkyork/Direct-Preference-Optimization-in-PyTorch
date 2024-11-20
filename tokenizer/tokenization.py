import torch
from torch.nn.utils.rnn import pad_sequence


def tokenize_dataset(tokenizer,batch):
    # Tokenize prompt, chosen, and rejected
    batch['prompt'] = batch['prompt']+"\n"
    
    prompt_token = tokenizer(batch['prompt'], padding=True, return_tensors="pt")
    chosen_token = tokenizer(batch['chosen'], padding=True, return_tensors="pt")
    rejected_token = tokenizer(batch['rejected'], padding=True, return_tensors="pt")
    
    # Add eos_token
    chosen_token['input_ids'] = torch.cat(
        [chosen_token['input_ids'], torch.tensor([[tokenizer.eos_token_id]])], dim=1
    )
    chosen_token['attention_mask'] = torch.cat(
        [chosen_token['attention_mask'], torch.tensor([[1]])], dim=1
    )

    rejected_token['input_ids'] = torch.cat(
        [rejected_token['input_ids'], torch.tensor([[tokenizer.eos_token_id]])], dim=1
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


def collate_fn(batch):
    """
    バッチ内のすべてのサンプルをパディングしてサイズを揃える
    """
    # 各サンプルの 'chosen_tokenizer' と 'rejected_tokenizer' を取り出し、リストに格納
    chosen_input_ids = [torch.tensor(sample['chosen_tokenizer']['input_ids'][0]) for sample in batch]
    chosen_attention_mask = [torch.tensor(sample['chosen_tokenizer']['attention_mask'][0]) for sample in batch]

    rejected_input_ids = [torch.tensor(sample['rejected_tokenizer']['input_ids'][0]) for sample in batch]
    rejected_attention_mask = [torch.tensor(sample['rejected_tokenizer']['attention_mask'][0]) for sample in batch]

    # パディングしてバッチ全体のサイズを揃える
    padded_chosen_input_ids = pad_sequence(chosen_input_ids, batch_first=True, padding_value=0)
    padded_chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)

    padded_rejected_input_ids = pad_sequence(rejected_input_ids, batch_first=True, padding_value=0)
    padded_rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

    # 結果を辞書として返す
    return {
        'chosen_tokenizer': {
            'input_ids': padded_chosen_input_ids,
            'attention_mask': padded_chosen_attention_mask,
        },
        'rejected_tokenizer': {
            'input_ids': padded_rejected_input_ids,
            'attention_mask': padded_rejected_attention_mask,
        },
    }