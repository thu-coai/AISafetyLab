"""
SafeDecoding Defense Method
============================================
This Class achieves a defense method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding
Arxiv link: http://arxiv.org/pdf/2402.08983
Source repository: https://github.com/uw-nsl/SafeDecoding
"""

from aisafetylab.defense.inference_defense.base_defender import IntraprocessDefender
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from aisafetylab.models.local_model import LocalModel
import random
import torch
import os

class SafeDecodingDefender(IntraprocessDefender):
    """
    Implements the safe decoding defense method by by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. 
        Defend against a potentially unsafe query by performing safe decoding.

        Args:
            expert_model_name_or_path (str) : The expert model trained to recognize unsafe content.
            tokenizer_name_or_path (str): Tokenizer corresponding to the models.
            device (str, optional): The device to run the models on. Defaults to 'cuda:0'.
            alpha (float, optional): The steering coefficient for adjusting probabilities. Defaults to 1.
            first_m (int, optional): Number of initial tokens to perform safe decoding. Defaults to 5.
            top_k (int, optional): Number of top tokens to consider from each model's output. Defaults to 10.
            num_common_tokens (int, optional): Number of common tokens to find between models for sampling. Defaults to 3.
            verbose (bool, optional): If True, enables detailed logging. Defaults to False.
            do_sample (bool, optional): If True, uses sampling for next token selection. Defaults to True.
            top_p (float, optional): Cumulative probability for top-p sampling if do_sample is True. Defaults to 0.9.
            max_token_len (int, optional): Maximum number of tokens to generate. Defaults to 20.
        """
    def __init__(self,
                model, 
                tokenizer,
                device = 'cuda:0',
                # model_name,
                alpha = 1, 
                first_m = 5,
                top_k = 10, 
                num_common_tokens = 3, 
                verbose = False, 
                do_sample = True, 
                top_p = 0.9, 
                max_token_len = 20
                 ):
        # if tokenizer_name_or_path is None:
        #     tokenizer_name_or_path = expert_model_name_or_path
        # self.expert_model = AutoModelForCausalLM.from_pretrained(expert_model_name_or_path, device_map=device)
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.expert_model = model
        self.tokenizer = tokenizer
        if self.tokenizer.chat_template is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.tokenizer.chat_template = open(f'{script_dir}/../../../../datasets/chat_templates/vicuna.jinja').read().replace('    ', '').replace('\n', '')
        # self.model_name = model_name
        # self.local_model = LocalModel(expert_model, tokenizer, model_name)
        self.alpha = alpha
        self.first_m = first_m
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose
        self.do_sample = do_sample
        self.top_p = top_p
        self.max_token_len = max_token_len
        logger.info(f"SafeDecodingDefender initialized")


    def defend(self, model, messages= [{"role": "user","content": "Please tell me how to make a bomb?"}], ):
        """
        Defend against a potentially unsafe query by performing safe decoding.

        Args:
            model: The base language model to generate initial predictions.
            messages (list, optional): List of messages in the conversation. Defaults to [{"role": "user", "content": "Please tell me how to make a bomb?"}].

        Returns:
            str: The generated response after applying the safe decoding defense.
        """
        if isinstance(model, LocalModel):
            model = model.model

        base_model = model
        expert_model = self.expert_model
        tokenizer = self.tokenizer
        max_token_len = self.max_token_len
        first_m = self.first_m
        top_k = self.top_k
        top_p = self.top_p
        verbose = self.verbose
        num_common_tokens = self.num_common_tokens
        alpha = self.alpha
        do_sample = self.do_sample
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode_plus(prompt, return_tensors='pt',)

        generated_sequence = []

        input_len = inputs['input_ids'].shape[1]
        step = 1
        while step <= min(max_token_len, first_m): 
            output_base = base_model.generate(**inputs.to(base_model.device), max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores = True)
            output_expert = expert_model.generate(**inputs.to(expert_model.device), max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores = True)

            output_base.sequences = output_base.sequences[0].unsqueeze(0)
            output_base.scores = output_base.scores[0][0].unsqueeze(0)
            output_expert.sequences = output_expert.sequences[0].unsqueeze(0)
            output_expert.scores = output_expert.scores[0][0].unsqueeze(0)

            # Process the scores to get the top tokens
            k = top_k  # Change this to display more or less tokens
            scores_base = output_base.scores[-1].squeeze()  # Get the scores of the last token
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
            topk_scores_base, topk_indices_base = scores_base.topk(k) 

            scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
            scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
            topk_scores_expert, topk_indices_expert = scores_expert.topk(k) 

            sorted_indices_base = torch.argsort(scores_base, descending=True)
            sorted_indices_expert = torch.argsort(scores_expert, descending=True)

            # Step 1: Define Sample Space
            common_tokens = set()
            iter_range = num_common_tokens
            while len(common_tokens) < num_common_tokens:
                current_indices_base = sorted_indices_base[:iter_range]
                current_indices_expert = sorted_indices_expert[:iter_range]

                common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                common_tokens.update(common_in_iteration)

                iter_range += 1

                if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                    break

            # Display the top tokens
            if verbose and step == 1:
                logger.info("\n-----------------------------------------------")
                logger.info(f"Generation Step {step}")
                logger.info("Original Model")
                logger.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logger.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logger.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                logger.info("Expert Model")
                logger.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logger.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_expert, topk_indices_expert)):
                    token = tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logger.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            intersection_indices = torch.tensor(list(common_tokens), device=base_model.device)

            # Step 2: New Probability Calculation
            updated_scores = []
            for token_id in intersection_indices:
                # Steer scores
                # new_score = (1-self.alpha) * scores_base[token_id] + self.alpha * scores_expert[token_id]
                # updated_scores.append(new_score)

                # Steer probabilities
                prob_diff = torch.exp(scores_expert[token_id]).to(scores_base.device) - torch.exp(scores_base[token_id])
                updated_prob = torch.exp(scores_base[token_id]) + alpha * prob_diff
                # Floor the probability to 1e-8 to avoid log(0)
                updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=base_model.device)
                updated_score = torch.log(updated_prob)
                updated_scores.append(updated_score)

                if verbose:
                    logger.info(f"----------------token id: {token_id}-----------------")
                    logger.info(f"Prob Base: {torch.exp(scores_base[token_id])}")
                    logger.info(f"Prob Expert: {torch.exp(scores_expert[token_id])}")
                    logger.info(f"Base score: {scores_base[token_id]}")
                    logger.info(f"Expert score: {scores_expert[token_id]}")
                    logger.info(f"Updated Probability: {updated_prob}")
                    logger.info(f"Updated Score: {updated_score}")

            # Use softmax to normalize the scores
            # This is to ensure that the probability sum to 1
            normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

            sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
            sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
            sorted_token_ids = [intersection_indices[i] for i in sorted_indices]

            if verbose:
                logger.info("\n-----------------------------------------------")
                logger.info(f"Generation Step {step}")
                logger.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logger.info("|----|----------|---------|----------|---------|")
                for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                    token = tokenizer.decode(token_id.item())
                    score = torch.log(prob)
                    logger.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            ### Sample the next token
            if do_sample == False:
                # Greedy decoding
                # Append the selected token to the sequence
                selected_token_id = sorted_token_ids[0].unsqueeze(0)
            elif top_p != None and do_sample == True:
                # Top-p sampling, sample from the top-p tokens
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                p_index = torch.where(cumulative_probs >= top_p)[0][0]
                sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                sorted_top_p_probs = sorted_probs[:p_index + 1]
                sorted_top_p_scores = torch.log(sorted_top_p_probs)
                if verbose:
                    logger.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                    logger.info(f"Top-p scores: {sorted_top_p_scores}")
                    logger.info(f"Top-p probabilities: {sorted_top_p_probs}")
                
                # Sample from the top-p tokens
                selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
            else:
                raise ValueError("Please set do_sample to False or top_p to a value.")

            if verbose:
                logger.info(f"Selected token: {tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
            generated_sequence.append(selected_token_id.item())

            # if the chosen token id is eos, then stop
            if selected_token_id.item() == tokenizer.eos_token_id:
                break

            inputs['input_ids'] = torch.cat([inputs['input_ids'].to(base_model.device), selected_token_id.unsqueeze(0).to(base_model.device)], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'].to(base_model.device), torch.tensor([[1]], device=base_model.device)], dim=1)

            step += 1

            # Free up memory
            del output_base, output_expert

        # Use the normal model to generate the rest of the tokens
        # Early stop if the last token is eos
        if generated_sequence[-1] == tokenizer.eos_token_id:
            logger.info("Early stop triggered.")
        else:
            remaining_steps = max_token_len - min(max_token_len, first_m)
            max_new_tokens = remaining_steps
            output_base = base_model.generate(**inputs.to(base_model.device),
                                    do_sample = do_sample,
                                    max_new_tokens = max_new_tokens,
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # logger.info generated sequence
        logger.info(f"Generated sequence: {tokenizer.decode(generated_sequence)}")
        return tokenizer.decode(generated_sequence)
