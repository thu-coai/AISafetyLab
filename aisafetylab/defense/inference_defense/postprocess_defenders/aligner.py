from aisafetylab.defense.inference_defense.base_defender import PostprocessDefender
from aisafetylab.defense.inference_defense.defender_texts import ALIGNER_DEFAULT_TEXT
from aisafetylab.models.base_model import Model
from loguru import logger

class AlignerDefender(PostprocessDefender):
    def __init__(self, model: Model, tokenizer, question, answer, aligner_prompt=ALIGNER_DEFAULT_TEXT):
        """
        Initialize AlignerDefender with external model and tokenizer
        
        Args:
            model: Pre-initialized language model
            tokenizer: Pre-initialized tokenizer
            question: Original question to align
            answer: Original answer to align
            aligner_prompt: Template for alignment prompt
        """
        self.model = model
        self.tokenizer = tokenizer
        self.aligner_prompt = aligner_prompt

    def defend(self, output, query):
        """
        Align the model output using the alignment model
        
        Args:
            output: Original model output to be aligned
            
        Returns:
            str: Aligned model output
        """
        if isinstance(query, str):
            query = [
                {
                    "role": "user",
                    "content": query
                }
            ]
        
        # Construct full prompt with the model output
        full_prompt = self.aligner_prompt.format(
            question=query,
            answer=output
        )
        
        # Tokenize and generate aligned output
        input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )[0]
        
        aligned_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return aligned_output
        
        