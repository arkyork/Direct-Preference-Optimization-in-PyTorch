import torch
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