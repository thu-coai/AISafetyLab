from aisafetylab.defense.inference_defense.base_defender import (
    PreprocessDefender,
    IntraprocessDefender,
    PostprocessDefender,
)
from aisafetylab.defense.inference_defense import SORRY_RESPONSE
from loguru import logger
from tqdm import tqdm

def chat(model, messages, defenders, **kwargs):
    """Processes a chat conversation through various defenders and returns a response.

    This function first classifies defenders into Preprocess, Intraprocess and Postpreprocess.
    It then processes the messages through input defenders, generates a response using
    a generation defender or the model's chat method, and finally processes the response
    through output defenders.

    The order of the application of Preprocess and Postprocess defender is decided by the origin order.

    Args:
        model (Model): The language model to interact with.
        messages (list): The conversation history or user query.
        defenders (list[InferDefender]): A list of defender instances.
        **kwargs: Additional keyword arguments for the model's chat method.

    Returns:
        The final response after processing through all defenders.
    """
    # Classify defenders into input, generation, and output levels
    input_defenders = []
    generation_defender = None
    output_defenders = []

    # logger.debug("defenders: ", defenders)
    if defenders is not None:
        for defender in defenders:
            if isinstance(defender, PreprocessDefender):
                input_defenders.append(defender)
            elif isinstance(defender, IntraprocessDefender):
                if generation_defender is not None:
                    raise ValueError("Only one IntraDefender is supported.")
                generation_defender = defender
            elif isinstance(defender, PostprocessDefender):
                output_defenders.append(defender)
            else:
                raise TypeError(f"Unknown defender type: {type(defender)}")

    # Stage 1: Preprocess Defenders
    for defender in input_defenders:
        messages, reject = defender.defend(messages, **kwargs)
        if reject:
            return SORRY_RESPONSE

    # Stage 2: Infraprocess Defender
    if generation_defender:
        response = generation_defender.defend(model, messages, **kwargs)
    else:
        # No generation defender, use the model's chat method
        response = model.chat(messages, **kwargs)

    # Stage 3: Postprocess Defenders
    for defender in output_defenders:
        logger.debug(f"start judging")
        response = defender.defend(response, **kwargs)

    return response

def batch_chat(model, messages, defenders, batch_size, **kwargs):
    # Classify defenders into input, generation, and output levels
    input_defenders = []
    generation_defender = None
    output_defenders = []

    for defender in defenders:
        if isinstance(defender, PreprocessDefender):
            input_defenders.append(defender)
        elif isinstance(defender, IntraprocessDefender):
            if generation_defender is not None:
                raise ValueError("Only one IntraDefender is supported.")
            generation_defender = defender
        elif isinstance(defender, PostprocessDefender):
            output_defenders.append(defender)
        else:
            raise TypeError(f"Unknown defender type: {type(defender)}")

    # Stage 1: Preprocess Defenders
    new_messages = []
    reject_res = []
    for defender in input_defenders:
        for message in tqdm(messages):
            new_message, reject = defender.defend(message, **kwargs)
            new_messages.append(new_message)
            reject_res.append(reject)
    
    if len(new_messages) == 0:
        new_messages = messages
    if len(reject_res) == 0:
        reject_res = [False] * len(new_messages)

    # Stage 2: Infraprocess Defender
    responses = []
    if generation_defender:
        for message in tqdm(new_messages):
            response = generation_defender.defend(model, message, **kwargs)
            responses.append(response)
    else:
        # No generation defender, use the model's chat method
        responses = model.batch_chat(new_messages, batch_size)
    
    for i in range(len(responses)):
        reject, response = reject_res[i], responses[i]
        if reject:
            responses[i] = SORRY_RESPONSE

    # Stage 3: Postprocess Defenders
    new_responses = []
    for defender in output_defenders:
        logger.debug(f"start judging")
        for query, response in tqdm(zip(new_messages, responses)):
            new_responses.append(defender.defend(response, query, **kwargs))
            
    if len(new_responses) == 0:
        new_responses = responses

    return new_responses
