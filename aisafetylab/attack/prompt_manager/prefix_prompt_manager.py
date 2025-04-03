import torch
from loguru import logger
from aisafetylab.attack.feedback import GradientFeedback, LogitsFeedback, LossFeedback
from aisafetylab.evaluation.scorers.pattern_scorer import PatternScorer

class PrefixPromptManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
        r"""
        :param ~str instruction: the harmful query.
        :param ~str target: the target response for the query.
        :param ~str adv_string: the jailbreak prompt.
        """
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
        
    def get_real_end(self, toks):
        for k in range(len(toks)-1, len(toks)-5, -1):
            if toks[k] == self.tokenizer.eos_token_id:
                return k
        return len(toks)
    
    def fix_ids(self, ids):
        # check if there are multiple bos tokens or eos tokens
        # if so, remove all but the first bos token and all but the last eos token
        new_start = 0
        new_end = len(ids)
        for i in range(1, len(ids)):
            if ids[i] == self.tokenizer.bos_token_id:
                new_start = i
            else:
                break
        for i in range(len(ids)-1, -1, -1):
            if ids[i] == self.tokenizer.eos_token_id:
                new_end = i + 1
            else:
                break
        
        return ids[new_start:new_end]

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string} {self.instruction}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        tot_toks = encoding.input_ids
        
        if self.conv_template.name in ['llama-2', 'mistral']: # use_fast=True for mistral
            self.conv_template.messages = []

            # self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
            # logger.info(f'prompt: {self.conv_template.get_prompt()}, toks: {self.tokenizer.convert_ids_to_tokens(toks)}')
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.append_message(self.conv_template.roles[0], self.adv_string)
            # self.conv_template.update_last_message(f"{self.goal}")
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
            self._control_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, self.get_real_end(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.adv_string}{separator}{self.instruction}")
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
            self._goal_slice = slice(self._control_slice.stop, self.get_real_end(toks))
            # logger.info(f'control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, len control_slice: {len(toks[self._control_slice])}')

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            # logger.debug(f'prompt: {self.conv_template.get_prompt()}')
            # logger.debug(f'tokenizer: {self.tokenizer}')
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            
            target_end_pos = self.get_real_end(toks)
            
            self._target_slice = slice(self._assistant_role_slice.stop, target_end_pos)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, target_end_pos - 1)
            
            # logger.debug(f'toks: {self.tokenizer.convert_ids_to_tokens(toks)}, user_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._user_role_slice])}, goal_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._goal_slice])}, control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, assistant_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._assistant_role_slice])}')
        
        else:
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            # logger.info(f'prompt: {self.conv_template.get_prompt()}, toks: {self.tokenizer.convert_ids_to_tokens(toks)}')
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.adv_string}")
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            self._control_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, self.get_real_end(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.adv_string}{separator}{self.instruction}")
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            self._goal_slice = slice(self._control_slice.stop, self.get_real_end(toks))
            # logger.info(f'control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, len control_slice: {len(toks[self._control_slice])}')

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            # logger.debug(f'prompt: {self.conv_template.get_prompt()}')
            # logger.debug(f'tokenizer: {self.tokenizer}')
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            
            target_end_pos = self.get_real_end(toks)
            
            self._target_slice = slice(self._assistant_role_slice.stop, target_end_pos)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, target_end_pos - 1)

        self.conv_template.messages = []

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.fix_ids(self.tokenizer(prompt).input_ids)
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        # logger.debug(f'tot input: {self.tokenizer.decode(input_ids)}\nprompt: {self.tokenizer.decode(input_ids[self._assistant_role_slice.stop])}')
        return input_ids


class GCGAttackPrompt(object):
    """
    A class used to generate an attack prompt. 
    """
    
    def __init__(self,
        goal,
        target,
        tokenizer,
        conv_template,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        *args, **kwargs
    ):
        """
        Initializes the AttackPrompt object with the provided parameters.

        Parameters
        ----------
        goal : str
            The intended goal of the attack
        target : str
            The target of the attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        """
        
        self.goal = goal
        self.target = target
        self.control = control_init
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.test_prefixes = test_prefixes

        self.conv_template.messages = []

        self.test_new_toks = len(self.tokenizer(self.target).input_ids) + 2 # buffer
        for prefix in self.test_prefixes:
            self.test_new_toks = max(self.test_new_toks, len(self.tokenizer(prefix).input_ids))

        self._update_ids()

        self.gradient_feedbacker = GradientFeedback()
        self.logits_feedbacker = LogitsFeedback()
        self.loss_feedbacker = LossFeedback()
        
    def get_real_end(self, toks):
        for k in range(len(toks)-1, len(toks)-5, -1):
            if toks[k] == self.tokenizer.eos_token_id:
                return k
        return len(toks)
    
    def fix_ids(self, ids):
        # check if there are multiple bos tokens or eos tokens
        # if so, remove all but the first bos token and all but the last eos token
        new_start = 0
        new_end = len(ids)
        for i in range(1, len(ids)):
            if ids[i] == self.tokenizer.bos_token_id:
                new_start = i
            else:
                break
        for i in range(len(ids)-1, -1, -1):
            if ids[i] == self.tokenizer.eos_token_id:
                new_end = i + 1
            else:
                break
        
        return ids[new_start:new_end]

    # def _update_ids(self):

    #     self.conv_template.append_message(self.conv_template.roles[0], f"{self.goal} {self.control}")
    #     self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
    #     prompt = self.conv_template.get_prompt()
    #     encoding = self.tokenizer(prompt)
    #     toks = encoding.input_ids

    #     if self.conv_template.name in ['llama-2', 'mistral']: # use_fast=True for mistral
    #         self.conv_template.messages = []

    #         # self.conv_template.append_message(self.conv_template.roles[0], None)
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
    #         # logger.info(f'prompt: {self.conv_template.get_prompt()}, toks: {self.tokenizer.convert_ids_to_tokens(toks)}')
    #         self._user_role_slice = slice(None, len(toks))

    #         self.conv_template.append_message(self.conv_template.roles[0], self.goal)
    #         # self.conv_template.update_last_message(f"{self.goal}")
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
    #         self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, self.get_real_end(toks)))

    #         separator = ' ' if self.goal and self.control[0] != ' ' else ''
    #         self.conv_template.update_last_message(f"{self.goal}{separator}{self.control}")
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
    #         self._control_slice = slice(self._goal_slice.stop, self.get_real_end(toks))
    #         # logger.info(f'control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, len control_slice: {len(toks[self._control_slice])}')

    #         self.conv_template.append_message(self.conv_template.roles[1], None)
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
    #         self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

    #         self.conv_template.update_last_message(f"{self.target}")
    #         # logger.debug(f'prompt: {self.conv_template.get_prompt()}')
    #         # logger.debug(f'tokenizer: {self.tokenizer}')
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            
    #         target_end_pos = self.get_real_end(toks)
            
    #         self._target_slice = slice(self._assistant_role_slice.stop, target_end_pos)
    #         self._loss_slice = slice(self._assistant_role_slice.stop-1, target_end_pos - 1)
            
    #         # logger.debug(f'toks: {self.tokenizer.convert_ids_to_tokens(toks)}, user_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._user_role_slice])}, goal_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._goal_slice])}, control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, assistant_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._assistant_role_slice])}')
        
    #     else:
    #         self.conv_template.messages = []

    #         self.conv_template.append_message(self.conv_template.roles[0], None)
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
    #         # logger.info(f'prompt: {self.conv_template.get_prompt()}, toks: {self.tokenizer.convert_ids_to_tokens(toks)}')
    #         self._user_role_slice = slice(None, len(toks))

    #         self.conv_template.update_last_message(f"{self.goal}")
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
    #         self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, self.get_real_end(toks)))

    #         separator = ' ' if self.goal and self.control[0] != ' ' else ''
    #         self.conv_template.update_last_message(f"{self.goal}{separator}{self.control}")
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
    #         self._control_slice = slice(self._goal_slice.stop, self.get_real_end(toks))
    #         # logger.info(f'control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, len control_slice: {len(toks[self._control_slice])}')

    #         self.conv_template.append_message(self.conv_template.roles[1], None)
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
    #         self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

    #         self.conv_template.update_last_message(f"{self.target}")
    #         # logger.debug(f'prompt: {self.conv_template.get_prompt()}')
    #         # logger.debug(f'tokenizer: {self.tokenizer}')
    #         toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            
    #         target_end_pos = self.get_real_end(toks)
            
    #         self._target_slice = slice(self._assistant_role_slice.stop, target_end_pos)
    #         self._loss_slice = slice(self._assistant_role_slice.stop-1, target_end_pos - 1)
            
    #         # logger.debug(f'toks: {self.tokenizer.convert_ids_to_tokens(toks)}, user_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._user_role_slice])}, goal_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._goal_slice])}, control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, assistant_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._assistant_role_slice])}'


    #     self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
    #     self.conv_template.messages = []
    
    def custom_add_generation_prompt(self, messages):
        TEXT = "FKDJSFJFKJDLJF"
        next_role = ''
        if len(messages) == 0:
            next_role = 'user'
        else:
            if messages[-1]['role'] == 'user':
                next_role = 'assistant'
            else:
                # system or assistant
                next_role = 'user'
        
        prompt = self.tokenizer.apply_chat_template(messages + [{'role': next_role, 'content': TEXT}], tokenize=False)
        idx = prompt.find(TEXT)
        ret = prompt[:idx]
        if ret[-1] == ' ':
            ret = ret[:-1]
        return ret
    
    def custom_apply_chat_template(self, messages, remove_special_end=False):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        if remove_special_end:
            last_text = messages[-1]['content']
            idx = prompt.rindex(last_text)
            prompt = prompt[:idx + len(last_text)]
        
        return prompt
        
    def _update_ids(self):

        # self.conv_template.append_message(self.conv_template.roles[0], f"{self.goal} {self.control}")
        # self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        messages = [{'role': 'user', 'content': f"{self.goal} {self.control}"}, {'role': 'assistant', 'content': f"{self.target}"}]
        # prompt = self.conv_template.get_prompt()
        prompt = self.custom_apply_chat_template(messages, remove_special_end=False)
        encoding = self.tokenizer(prompt)
        toks = self.fix_ids(encoding.input_ids)

        if self.conv_template.name in ['llama-2', 'mistral']: # use_fast=True for mistral
            self.conv_template.messages = []

            # self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
            # logger.info(f'prompt: {self.conv_template.get_prompt()}, toks: {self.tokenizer.convert_ids_to_tokens(toks)}')
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.append_message(self.conv_template.roles[0], self.goal)
            # self.conv_template.update_last_message(f"{self.goal}")
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, self.get_real_end(toks)))

            separator = ' ' if self.goal and self.control[0] != ' ' else ''
            self.conv_template.update_last_message(f"{self.goal}{separator}{self.control}")
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt().strip()).input_ids)
            self._control_slice = slice(self._goal_slice.stop, self.get_real_end(toks))
            # logger.info(f'control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, len control_slice: {len(toks[self._control_slice])}')

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            # logger.debug(f'prompt: {self.conv_template.get_prompt()}')
            # logger.debug(f'tokenizer: {self.tokenizer}')
            toks = self.fix_ids(self.tokenizer(self.conv_template.get_prompt()).input_ids)
            
            target_end_pos = self.get_real_end(toks)
            
            self._target_slice = slice(self._assistant_role_slice.stop, target_end_pos)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, target_end_pos - 1)
            
            # logger.debug(f'toks: {self.tokenizer.convert_ids_to_tokens(toks)}, user_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._user_role_slice])}, goal_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._goal_slice])}, control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, assistant_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._assistant_role_slice])}')
        
        else:
            # self.conv_template.messages = []

            # self.conv_template.append_message(self.conv_template.roles[0], None)
            _prompt = self.custom_add_generation_prompt([])
            toks = self.fix_ids(self.tokenizer(_prompt).input_ids)
            # logger.info(f'prompt: {self.conv_template.get_prompt()}, toks: {self.tokenizer.convert_ids_to_tokens(toks)}')
            self._user_role_slice = slice(None, len(toks))

            _prompt = self.custom_apply_chat_template([{'role': 'user', 'content': self.goal}], remove_special_end=True)
            # self.conv_template.update_last_message(f"{self.goal}")
            toks = self.fix_ids(self.tokenizer(_prompt).input_ids)
            self._goal_slice = slice(self._user_role_slice.stop, len(toks))

            separator = ' ' if self.goal and self.control[0] != ' ' else ''
            _prompt = self.custom_apply_chat_template([{'role': 'user', 'content': f"{self.goal}{separator}{self.control}"}], remove_special_end=True)
            toks = self.fix_ids(self.tokenizer(_prompt).input_ids)
            self._control_slice = slice(self._goal_slice.stop, len(toks))
            # logger.info(f'control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, len control_slice: {len(toks[self._control_slice])}')
            _prompt = self.custom_add_generation_prompt([{'role': 'user', 'content': f"{self.goal}{separator}{self.control}"}])
            toks = self.fix_ids(self.tokenizer(_prompt).input_ids)
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            _prompt = self.custom_apply_chat_template([{'role': 'user', 'content': f"{self.goal}{separator}{self.control}"}, {'role': 'assistant', 'content': self.target}], remove_special_end=True)
            # self.conv_template.update_last_message(f"{self.target}")
            # logger.debug(f'prompt: {self.conv_template.get_prompt()}')
            # logger.debug(f'tokenizer: {self.tokenizer}')
            toks = self.fix_ids(self.tokenizer(_prompt).input_ids)
            
            # target_end_pos = self.get_real_end(toks)
            target_end_pos = len(toks)
            
            self._target_slice = slice(self._assistant_role_slice.stop, target_end_pos)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, target_end_pos - 1)
            
            # logger.debug(f'toks: {self.tokenizer.convert_ids_to_tokens(toks)}, user_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._user_role_slice])}, goal_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._goal_slice])}, control_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._control_slice])}, assistant_role_slice: {self.tokenizer.convert_ids_to_tokens(toks[self._assistant_role_slice])}')


        self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
        self.conv_template.messages = []

    @torch.no_grad()
    def generate(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.do_sample = False
            gen_config.max_new_tokens = 32
        
        if gen_config.max_new_tokens > 32:
            logger.warning('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(model.device).unsqueeze(0)
        # logger.info(f'gen_config: {gen_config.__dict__}')
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id)[0]

        return output_ids[self._assistant_role_slice.stop:]
    
    def generate_str(self, model, gen_config=None):
        return self.tokenizer.decode(self.generate(model, gen_config))
    
    def test(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            # gen_config.max_new_tokens = self.test_new_toks
            gen_config.max_new_tokens = 32
            gen_config.do_sample = False
        
        gen_str = self.generate_str(model, gen_config).strip()
        input_prompt = self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop], clean_up_tokenization_spaces=False)
        logger.info(f'Input: {input_prompt}\nOutput: {gen_str}')
        # jailbroken = not any([prefix in gen_str for prefix in self.test_prefixes])
        jailbroken = PatternScorer().score(query=None, response=gen_str)['score']
        em = self.target in gen_str
        return jailbroken, int(em)

    @torch.no_grad()
    def test_loss(self, model):
        logits, ids = self.logits(model, return_ids=True)
        return self.target_loss(logits, ids).mean().item()
    
    def grad(self, model):
        return self.gradient_feedbacker.get_grad_of_control_tokens(
            model, 
            self.input_ids, 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice,
        )
    
    
    @torch.no_grad()
    def logits(self, model, test_controls=None, return_ids=False):

        if test_controls is None:
            test_controls = self.control_toks

        return self.logits_feedbacker.get_logits_with_control_tokens(
            model = model,
            input_ids = self.input_ids,
            test_controls=test_controls,
            control_slice= self._control_slice,
            tokenizer= self.tokenizer,
            return_ids=return_ids,
        )
        
    def target_loss(self, logits, ids):
        return self.loss_feedbacker.get_sliced_loss(
            logits=logits,
            ids=ids,
            target_slice=self._target_slice,
        )
    
    def control_loss(self, logits, ids):
        return self.loss_feedbacker.get_sliced_loss(
            logits=logits,
            ids=ids,
            target_slice=self._control_slice,
        )
    
    @property
    def assistant_str(self):
        return self.tokenizer.decode(self.input_ids[self._assistant_role_slice], clean_up_tokenization_spaces=False).strip()
    
    @property
    def assistant_toks(self):
        return self.input_ids[self._assistant_role_slice]

    @property
    def goal_str(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice], clean_up_tokenization_spaces=False).strip()

    @goal_str.setter
    def goal_str(self, goal):
        self.goal = goal
        self._update_ids()
    
    @property
    def goal_toks(self):
        return self.input_ids[self._goal_slice]
    
    @property
    def target_str(self):
        return self.tokenizer.decode(self.input_ids[self._target_slice], clean_up_tokenization_spaces=False).strip()
    
    @target_str.setter
    def target_str(self, target):
        self.target = target
        self._update_ids()
    
    @property
    def target_toks(self):
        return self.input_ids[self._target_slice]
    
    @property
    def control_str(self):
        return self.tokenizer.decode(self.input_ids[self._control_slice], clean_up_tokenization_spaces=False).strip()
    
    @control_str.setter
    def control_str(self, control):
        self.control = control
        self._update_ids()
    
    @property
    def control_toks(self):
        return self.input_ids[self._control_slice]
    
    @control_toks.setter
    def control_toks(self, control_toks):
        self.control = self.tokenizer.decode(control_toks, clean_up_tokenization_spaces=False)
        self._update_ids()
    
    @property
    def prompt(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice.start:self._control_slice.stop], clean_up_tokenization_spaces=False)
    
    @property
    def input_toks(self):
        return self.input_ids
    
    @property
    def input_str(self):
        return self.tokenizer.decode(self.input_ids, clean_up_tokenization_spaces=False)
    
    @property
    def eval_str(self):
        return self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop], clean_up_tokenization_spaces=False).replace('<s>','').replace('</s>','')


from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
class ModelWorker(object):
    # This class has deep connection to GCGAttackPrompt, so we put it here

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.float16,
            torch_dtype="auto",
            trust_remote_code=True,
            **model_kwargs
        ).to(device).eval()
        self.model.requires_grad_(False) # disable grads wrt weights
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue()
        self.process = None
    
    @staticmethod
    def run(model, tasks, results):
        while True:
            task = tasks.get()
            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

    def start(self):
        self.process = mp.Process(
            target=ModelWorker.run,
            args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        logger.info(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self
    
    def stop(self):
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self
