from aisafetylab.models import LocalModel, OpenAIModel
from loguru import logger
from time import sleep

def generate(object, messages, input_field_name='input_ids', **kwargs):
    
    if isinstance(messages, str):
        messages = [messages]
    
    if isinstance(object, LocalModel):
        # 检查是否有conversation属性，如果有则使用传统方式
        if hasattr(object, 'conversation') and hasattr(object.conversation, 'messages'):
            logger.debug('Using fschat template for local model generation...')
            object.conversation.messages = []
            for index, message in enumerate(messages):
                object.conversation.append_message(object.conversation.roles[index % 2], message)
            
            if object.conversation.roles[-1] not in object.conversation.get_prompt():
                object.conversation.append_message(object.conversation.roles[-1], None)
            prompt = object.conversation.get_prompt()
        else:
            # 使用apply_chat_template方法处理新模型
            logger.debug('Using apply_chat_template for local model generation...')
            if len(messages) == 1:
                prompt = object.apply_chat_template([{"role": "user", "content": messages[0]}])
            else:
                # 处理多轮对话
                chat_messages = []
                for index, message in enumerate(messages):
                    role = "user" if index % 2 == 0 else "assistant"
                    chat_messages.append({"role": role, "content": message})
                prompt = object.apply_chat_template(chat_messages)

        # inputs = object.tokenizer(prompt,
        #                           return_tensors='pt',
        #                           add_special_tokens=False)
        # input_ids = inputs.input_ids.to(object.model.device.index)
        # attention_mask = inputs.attention_mask.to(object.model.device.index)
        # input_length = len(input_ids[0])

        # # 设置生成所需的参数
        # generate_kwargs = {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'pad_token_id': object.tokenizer.pad_token_id
        # }
        # gen_config = object.generation_config
        # gen_config.update(kwargs)
        # # logger.debug(f'Generation config: {gen_config}')
        # output_ids = object.model.generate(**generate_kwargs, **gen_config)
        # output = object.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        
        output = object.chat(prompt, use_chat_template=False, **kwargs)

    elif isinstance(object, OpenAIModel):
        # OpenAI模型处理
        max_cnt = 10
        try_cnt = 0
        while try_cnt < max_cnt:
            try_cnt += 1
            try:
                if hasattr(object, 'conversation') and hasattr(object.conversation, 'messages'):
                    object.conversation.messages = []
                    for index, message in enumerate(messages):
                        object.conversation.append_message(object.conversation.roles[index % 2], message)
                    gen_config = object.generation_config
                    gen_config.update(kwargs)
                    response = object.client.chat.completions.create(
                        model=object.model_name,
                        messages=object.conversation.to_openai_api_messages(),
                        **gen_config
                    )
                else:
                    # 直接构建消息列表
                    if len(messages) == 1:
                        api_messages = [{"role": "user", "content": messages[0]}]
                    else:
                        api_messages = []
                        for index, message in enumerate(messages):
                            role = "user" if index % 2 == 0 else "assistant"
                            api_messages.append({"role": role, "content": message})
                    gen_config = object.generation_config
                    gen_config.update(kwargs)
                    response = object.client.chat.completions.create(
                        model=object.model_name,
                        messages=api_messages,
                        **gen_config
                    )
                output = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"API calling fails. Retry: ({try_cnt}/{max_cnt}). Error message: {e}")
                if try_cnt >= max_cnt:
                    logger.error("Reach max try times for api calling. Exit.")
                    break
                sleep(3)
                continue
            else:
                break
        

    return output