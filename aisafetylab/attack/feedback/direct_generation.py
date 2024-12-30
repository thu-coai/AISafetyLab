from aisafetylab.models import LocalModel, OpenAIModel

def generate(object, messages, input_field_name='input_ids', **kwargs):
    
    object.conversation.messages = []
    if isinstance(messages, str):
        messages = [messages]
    for index, message in enumerate(messages):
        object.conversation.append_message(object.conversation.roles[index % 2], message)

    if isinstance(object, LocalModel):
        if object.conversation.roles[-1] not in object.conversation.get_prompt():
            object.conversation.append_message(object.conversation.roles[-1], None)
        prompt = object.conversation.get_prompt()

        input_ids = object.tokenizer(prompt,
                                    return_tensors='pt',
                                    add_special_tokens=False).input_ids.to(object.model.device.index)
        input_length = len(input_ids[0])

        kwargs.update({input_field_name: input_ids})
        output_ids = object.model.generate(**kwargs, **object.generation_config)
        output = object.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    elif isinstance(object, OpenAIModel):
        response = object.client.chat.completions.create(
            model=object.model_name,
            messages=object.conversation.to_openai_api_messages(),
            **kwargs,
            **object.generation_config
        )
        output = response.choices[0].message.content
        

    return output