import torch
import json
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from fastchat.conversation import get_conv_template

class SafetyDataset(Dataset):

    def __init__(self,
                 args,
                 data,
                 tokenizer
                 ):
       
        self.max_input_length = args.max_input_length

        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if 'vicuna' in tokenizer.name_or_path:
            self.conversation = get_conv_template("vicuna_v1.1")
        else:
            NotImplementedError("Model not implemented")

    def __len__(self) -> int:
        return len(self.data)

    def add_template(self, text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
            
        conversation = self.conversation.copy()
        if messages[0]['role'] == 'system':
            conversation.set_system_message(messages[0]['content'])
            messages = messages[1:]
        for msg in messages:
            conversation.append_message(msg['role'], msg['content'])
        # conversation.append_message(self.conversation.roles[1], None)
        
        return conversation.get_prompt()
    
        # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # return prompt
    
    def tokenize_text(self, text: str, max_length, padding='max_length', add_special_tokens=False) -> tuple:
        encoded_inputs = self.tokenizer(text, add_special_tokens=add_special_tokens, max_length=max_length, padding=padding, truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        d = self.data[idx]
        
        text_input = d['input']
        
        text_input = self.add_template(text_input)
        # print("-----")
        # print(text_input)
        
        input_ids, input_mask = self.tokenize_text(text_input, self.args.max_input_length, padding=False, add_special_tokens=False)
        
        if 'output' in d:
            output_text = d['output']
            
            output_ids, _ = self.tokenize_text(output_text + self.tokenizer.eos_token, self.args.max_length, padding=False, add_special_tokens=False)
            concat_input_ids = torch.cat([input_ids, output_ids], dim=-1)
            tot_max_len = self.args.max_length
            if len(concat_input_ids) < tot_max_len:
                padded_tokens = torch.full((tot_max_len - len(concat_input_ids), ), fill_value=self.tokenizer.eos_token_id)
                padded_input_ids = torch.cat([concat_input_ids, padded_tokens], dim=-1)
            else:
                padded_input_ids = concat_input_ids[:tot_max_len]
                
            output_ids = padded_input_ids.clone()
            concat_len = len(concat_input_ids)
            output_ids[concat_len:] = -100
            input_len = len(input_ids)
            output_ids[:input_len] = -100
            
            attention_mask = torch.zeros_like(padded_input_ids)
            attention_mask[:concat_len] = 1.
            
            data = dict(
                input_ids=padded_input_ids,
                labels=output_ids,
                attention_mask=attention_mask,
                loss_type=d['label']
            )

            return data
        
        else:
            raise NotImplementedError("Method not implemented") 
        
        
def create_dataset(args, tokenizer):
    with open(f'{args.dataset}/train.json', 'r', encoding='utf8') as f:
        train_data = json.load(f)
        
    train_dataset = SafetyDataset(args, train_data, tokenizer)
    
    with open(f'{args.dataset}/valid.json', 'r', encoding='utf8') as f:
        valid_data = json.load(f)
        
    valid_dataset = SafetyDataset(args, valid_data, tokenizer)
    
    return train_dataset, valid_dataset