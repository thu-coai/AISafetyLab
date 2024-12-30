import torch

def generate(model, tokenizer, input_ids, gen_config=None, batch=False):
    if not batch:
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 64
        
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]
        return output_ids[input_ids.size(1):]
    
    else:
        raise NotImplementedError
 