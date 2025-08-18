from aisafetylab.models import LocalModel, OpenAIModel

def generate(object, messages, input_field_name='input_ids', **kwargs):
    
    if isinstance(messages, str):
        messages = [messages]
    
    if isinstance(object, LocalModel):
        # 检查是否有conversation属性，如果有则使用传统方式
        if hasattr(object, 'conversation') and hasattr(object.conversation, 'messages'):
            object.conversation.messages = []
            for index, message in enumerate(messages):
                object.conversation.append_message(object.conversation.roles[index % 2], message)
            
            if object.conversation.roles[-1] not in object.conversation.get_prompt():
                object.conversation.append_message(object.conversation.roles[-1], None)
            prompt = object.conversation.get_prompt()
        else:
            # 使用apply_chat_template方法处理新模型
            if len(messages) == 1:
                prompt = object.apply_chat_template([{"role": "user", "content": messages[0]}])
            else:
                # 处理多轮对话
                chat_messages = []
                for index, message in enumerate(messages):
                    role = "user" if index % 2 == 0 else "assistant"
                    chat_messages.append({"role": role, "content": message})
                prompt = object.apply_chat_template(chat_messages)

        inputs = object.tokenizer(prompt,
                                  return_tensors='pt',
                                  add_special_tokens=False)
        input_ids = inputs.input_ids.to(object.model.device.index)
        attention_mask = inputs.attention_mask.to(object.model.device.index)
        input_length = len(input_ids[0])

        # 设置生成所需的参数
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pad_token_id': object.tokenizer.pad_token_id
        }
        generate_kwargs.update(kwargs)
        output_ids = object.model.generate(**generate_kwargs, **object.generation_config)
        output = object.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    elif isinstance(object, OpenAIModel):
        # OpenAI模型处理
        if hasattr(object, 'conversation') and hasattr(object.conversation, 'messages'):
            object.conversation.messages = []
            for index, message in enumerate(messages):
                object.conversation.append_message(object.conversation.roles[index % 2], message)
            response = object.client.chat.completions.create(
                model=object.model_name,
                messages=object.conversation.to_openai_api_messages(),
                **kwargs,
                **object.generation_config
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
            
            response = object.client.chat.completions.create(
                model=object.model_name,
                messages=api_messages,
                **kwargs,
                **object.generation_config
            )
        output = response.choices[0].message.content
        

    return output