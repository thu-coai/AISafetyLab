from openai import OpenAI
from fastchat.conversation import get_conv_template
from time import sleep
from loguru import logger
from .base_model import Model
import openai
import time

class OpenAIModel(Model):
    def __init__(self, model_name, base_url, api_key, generation_config=None):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.conversation = get_conv_template('chatgpt')
        self.generation_config = generation_config if generation_config is not None else {}

        self.API_RETRY_SLEEP = 10
        self.API_ERROR_OUTPUT = "$ERROR$"
        self.API_QUERY_SLEEP = 0.5
        self.API_MAX_RETRY = 5
        self.API_TIMEOUT = 20
        self.API_LOGPROBS = True
        self.API_TOP_LOGPROBS = 20

    def set_system_message(self, system_message: str):
        """
        Sets a system message for the conversation.
        :param str system_message: The system message to set.
        """
        self.conversation.system_message = system_message
    
    def generate(self, messages, clear_old_history=True, max_try=30, try_gap=5, gap_increase=5, **kwargs):
        """
        Generates a response based on messages that include conversation history.
        :param list[dict]|list[str]|str messages: A list of messages or a single message string.
                                       User and assistant messages should alternate.
        :param bool clear_old_history: If True, clears the old conversation history before adding new messages.
        :return str: The response generated by the OpenAI model based on the conversation history.
        """
        
        ##print("messages: ", messages)
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(messages[0], dict):
            if 'role' in messages[0] and messages[0]['role'] == 'system':
                self.conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            messages = [msg['content'] for msg in messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
            
        cur = 0
        temp_gen_config = self.generation_config.copy()
        if kwargs:
            temp_gen_config.update(kwargs)
        while cur < max_try:
            try:
                logger.debug(f"model_name: {self.model_name}, messages: {self.conversation.to_openai_api_messages()}")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation.to_openai_api_messages(),
                    **temp_gen_config
                )
                logger.debug(f"response: {response}")
                content = response.choices[0].message.content
                if content is None:
                    raise Exception("Empty response")
                return content
            except Exception as e:
                logger.error(f"Failed to generate response within {self.model_name}: {e}, retrying...")
                cur += 1
                if cur < max_try:
                    sleep(try_gap)
                    try_gap += gap_increase
                else:
                    raise e
    
    def chat(self, messages, clear_old_history=True, max_try=30, try_gap=5, **kwargs):
        return self.generate(messages, clear_old_history, max_try, try_gap, **kwargs)
    
    def batch_chat(self, batch_messages, clear_old_history=True, max_try=30, try_gap=5, **kwargs):
        return [self.chat(messages, clear_old_history, max_try, try_gap, **kwargs) for messages in batch_messages]
    
    def get_response(self, prompts_list, max_n_tokens=None, no_template=False, gen_config={}):
        if isinstance(prompts_list[0], str):
            prompts_list = [[{'role': 'user', 'content': prompt}] for prompt in prompts_list]
        
        convs = prompts_list
        print(convs)
        outputs = []
        for conv in convs:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens,
                        **gen_config,
                        timeout=self.API_TIMEOUT,
                        logprobs=self.API_LOGPROBS,
                        top_logprobs=self.API_TOP_LOGPROBS,
                        seed=0,
                    )
                    response_logprobs = [
                        dict((response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].token, 
                                response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].logprob) 
                                for i_top_logprob in range(self.API_TOP_LOGPROBS)
                        )
                        for i_token in range(len(response.choices[0].logprobs.content))
                    ]
                    output = {'text': response.choices[0].message.content,
                            'logprobs': response_logprobs,
                            'n_input_tokens': response.usage.prompt_tokens,
                            'n_output_tokens': response.usage.completion_tokens,
                    }
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)
            outputs.append(output)
        return outputs
        