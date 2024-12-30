import random
import spacy
from tqdm import trange
import torch
from types import SimpleNamespace


def insert_adv_string(msg, adv):
    return msg + adv


class TrieNode:
    """
    Represents a node in the Trie structure, used to store characters for prefix and suffix search.

    Attributes:
    - children (dict): Dictionary of child nodes.
    - is_end_of_word (bool): True if the node marks the end of a word.
    """

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    """
    Trie structure for storing and searching words to identify prefixes and suffixes.

    Attributes:
    - root (TrieNode): Root node of the Trie.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the Trie.

        Parameters:
        - word (str): Word to be inserted into the Trie.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def is_prefix(
        self, word
    ):  # if the word is a prefix of any word in the trie, return True
        """
        Checks if the word is a prefix of any other word in the Trie.

        Parameters:
        - word (str): Word to check as prefix.

        Returns:
        - bool: True if the word is a prefix, False otherwise.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        if len(node.children) == 0:
            return False
        else:
            return True


def find_non_prefix_strings(str_list):
    """
    Finds strings in a list that are not prefixes of any other string in the list.

    Parameters:
    - str_list (list of str): List of strings to search.

    Returns:
    - list of str: Strings that are not prefixes of any other string.
    """
    trie = Trie()
    for word in str_list:
        trie.insert(word)

    result = []
    for word in str_list:
        if not trie.is_prefix(word):
            result.append(word)
    return result


def find_fuction_words(string_lists, batch_size=1000):
    """
    Identifies function words in batches of input strings using spaCy's POS tagging.

    Parameters:
    - string_lists (list of str): List of strings to process in batches.
    - batch_size (int): Number of strings to process per batch.

    Returns:
    - list of str or None: List of identified function words, or None if none are found.
    """
    nlp = spacy.load("zh_core_web_sm")

    # string_list = ["苹果", "的", "颜色", "是", "红色"]
    # string_list = ['How', "are", "you", "I", "am"]

    for i in trange(0, len(string_lists), batch_size, desc="Finding fuction words"):
        string_list = string_lists[i : i + batch_size]
        sentence = " ".join(string_list)
        doc = nlp(sentence)

        function_words_tags = {
            "ADP",
            "CCONJ",
            "PART",
            "DET",
            "PRON",
            "SCONJ",
            "AUX",
            "ADV",
            "INTJ",
        }

        function_words = [
            token.text for token in doc if token.pos_ in function_words_tags
        ]

        if len(function_words) > 0:
            return function_words
    return None


def find_separators(tokenizer):
    """
    Finds suitable prefix and suffix separators, and an adversarial token, based on tokenizer's vocabulary.

    Parameters:
    - tokenizer: Tokenizer with vocabulary fortokenization and encoding.

    Returns:
    - tuple of str: A tuple containing the selected prefix separator, suffix separator, and an adversarial token.
    """
    str_list = list(tokenizer.vocab.keys())
    non_prefix_strings = find_non_prefix_strings(str_list)
    non_prefix_strings = sorted(non_prefix_strings, key=len)

    str_list = list(tokenizer.vocab.keys())
    str_list = ["".join([char for char in reversed(word)]) for word in str_list]
    non_suffix_strings = find_non_prefix_strings(str_list)
    non_suffix_strings = [
        "".join([char for char in reversed(word)]) for word in non_suffix_strings
    ]
    non_suffix_strings = sorted(non_suffix_strings, key=len)
    print(f"不是任何其他字符串后缀的字符串: {non_suffix_strings}")

    non_prefix_function_words = find_fuction_words(non_prefix_strings)
    non_suffix_function_words = find_fuction_words(non_suffix_strings)
    non_pre_non_suf_function_words = list(
        set(non_prefix_function_words) & set(non_suffix_function_words)
    )
    prefix_seperator, suffix_seperator, adv_token = random.choices(
        non_pre_non_suf_function_words, k=3
    )
    return prefix_seperator, suffix_seperator, adv_token


def get_pure_token_ids(text, prefix_separator, suffix_separator, tokenizer):
    """
    Encodes the text with specified prefix and suffix separators, and then extracts the
    token IDs for the content between the separators.

    Parameters:
    - text (str): The text to encode.
    - prefix_separator (str): The prefix separator used to mark the start of the main content.
    - suffix_separator (str): The suffix separator used to mark the end of the main content.
    - tokenizer: Tokenizer object used for encoding and decoding text.

    Returns:
    - torch.Tensor: Token IDs for the content between the specified prefix and suffix separators.
    """
    # prefix_separator_id = tokenizer.convert_tokens_to_ids([prefix_separator])[0]
    # suffix_separator_id = tokenizer.convert_tokens_to_ids([suffix_separator])[0]

    prefix_separator_id = tokenizer.convert_tokens_to_ids(prefix_separator)
    suffix_separator_id = tokenizer.convert_tokens_to_ids(suffix_separator)

    token_ids = tokenizer.encode(
        prefix_separator + text + suffix_separator,
        add_special_tokens=False,
        return_tensors="pt",
    )[0]
    # skip the prefix_separator and the suffix_separator
    prefix_separator_index = (token_ids == prefix_separator_id).nonzero()[0].item()
    suffix_separator_index = (token_ids == suffix_separator_id).nonzero()[-1].item()
    return token_ids[prefix_separator_index + 1 : suffix_separator_index]


def pad_or_truncate(tensor, target_length):
    """
    Pads or truncates a tensor to a target length by duplicating or removing tokens.

    Parameters:
    - tensor (torch.Tensor): The tensor to pad or truncate.
    - target_length (int): The desired length of the tensor.

    Returns:
    - torch.Tensor: Tensor padded or truncated to the specified length.
    """
    current_length = tensor.size(0)
    # pad_length_list = []

    if current_length > target_length:
        return SimpleNamespace(tensor=tensor[:target_length], pad_length=0)
    elif current_length < target_length:
        pad_length = target_length - current_length
        padding = tensor[-1].expand(pad_length)
        # pad_length_list.append(pad_length)
        return SimpleNamespace(
            tensor=torch.cat((tensor, padding), dim=0), pad_length=pad_length
        )
    else:
        return SimpleNamespace(
            tensor=tensor,
            pad_length=0,
        )


def tidify(adv_ids, target_length, tokenizer, prefix_separator, suffix_separator):
    """
    The corresponding tokens of adv_ids may not be valid because of token merges, to solve this problem, this function first tranlate the token ids into text and then encode the text into corresponding token ids, which will then be padded or truncated to a target length.

    Parameters:
    - adv_ids (torch.Tensor): Token IDs to process.
    - target_length (int): Desired length for the output.
    - tokenizer: Tokenizer used for encoding/decoding.
    - prefix_separator (str): Prefix separator to remove.
    - suffix_separator (str): Suffix separator to remove.

    Returns:
    - torch.Tensor: Processed token IDs of the specified length.
    """
    # print(f"Before tidify:{adv_ids}")
    adv_text = tokenizer.decode(adv_ids)
    adv_ids = get_pure_token_ids(
        adv_text, prefix_separator, suffix_separator, tokenizer
    )
    # print(f"After tidify:{adv_ids}")
    adv_ids = pad_or_truncate(adv_ids, target_length).tensor
    # print(f"After pad/truncate: {adv_ids}")
    return adv_ids


def batch_tidify(
    variants_tensor, target_length, tokenizer, prefix_separator, suffix_separator
):
    """
    Applies the tidify process to each tensor in a batch, ensuring all have the same target length.

    Parameters:
    - variants_tensor (list of torch.Tensor): Batch of token IDs to process.
    - target_length (int): Desired length for each output tensor.
    - tokenizer: Tokenizer used for encoding/decoding.
    - prefix_separator (str): Prefix separator to remove.
    - suffix_separator (str): Suffix separator to remove.

    Returns:
    - torch.Tensor: Batch of processed token IDs of the specified length.
    """
    tidified_variants_tensor = []
    for adv_ids in variants_tensor:
        adv_ids = tidify(
            adv_ids, target_length, tokenizer, prefix_separator, suffix_separator
        )
        tidified_variants_tensor.append(adv_ids)
    variants_tensor = torch.stack(tidified_variants_tensor)
    return variants_tensor


def find_adv_indices_and_output_indices(
    tokenizer,
    prefix_separator_id,
    suffix_separator_id,
    prefix_separator,
    suffix_separator,
    messages_components,
):
    """
    Identifies the indices of adversarial tokens and the goal output tokens in a list of messages, based on separator tokens.

    Parameters:
    - tokenizer: Tokenizer used for encoding and decoding text.
    - prefix_separator_id (int): Token ID for the prefix separator.
    - suffix_separator_id (int): Token ID for the suffix separator.
    - prefix_separator (str): Prefix separator string.
    - suffix_separator (str): Suffix separator string.
    - messages_components (list of dict): List of message dictionaries with 'role' and 'content' keys.

    Returns:
    - SimpleNameSpace: Dictionary containing encoded prompt, adversarial token indices, output indices, and target length.
    """

    """
    PARAMETER EXAMPLE:

    messages_components = [
            {
                "role": "user",
                "content": [
                    {"type": "prompt", "text": instruct},
                    {"type": "prefix_separator", "text": world.prefix_separator}, 
                    {"type": "adv_string", "text": init_adv_string}, 
                    {"type": "suffix_separator", "text": world.suffix_separator}, 
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "goal", "text": goal}]
            }
        ]
    """
    tokenizer = tokenizer
    prefix_separator_id = prefix_separator_id
    suffix_separator_id = suffix_separator_id
    prefix_separator = prefix_separator
    suffix_separator = suffix_separator

    messages = []
    adv_slices = []

    for message in messages_components:
        message_reg = {}  
        message_reg["role"] = message["role"]
        message_components = message["content"]  
        message_content = ""  
        for component in message_components:
            text = component["text"]
            message_content += text  

            tmp_message = dict(
                role=message_reg["role"], content=message_content
            )  
            if component["type"] == "prefix_separator":  
                tmp_messages = messages + [tmp_message]
                # print("tmp_messages", tmp_messages)
                tokens = tokenizer.apply_chat_template(
                    tmp_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )[0]
                # print("tokens",tokens)
                # print("prefix_separator_id", prefix_separator_id)
                prefix_separator_index = (
                    (tokens == prefix_separator_id).nonzero()[-1].item()
                )
                component_slice = slice(prefix_separator_index + 1, None)
            elif component["type"] == "suffix_separator":  
                tmp_messages = messages + [tmp_message]
                tokens = tokenizer.apply_chat_template(
                    tmp_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )[0]
                suffix_separator_index = (
                    (tokens == suffix_separator_id).nonzero()[-1].item()
                )
                component_slice = slice(component_slice.start, suffix_separator_index)
                adv_slices.append(
                    component_slice
                ) 

        message_reg["content"] = (
            message_content 
        )
        messages.append(message_reg)

    # Finding the slice of output
    user_msg = messages
    user_msg[-1]["content"] += suffix_separator
    prompt_with_template_without_output = tokenizer.apply_chat_template(
        user_msg[:-1], add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    prompt_with_output_encoded = tokenizer.apply_chat_template(
        user_msg[:], add_generation_prompt=False, tokenize=True, return_tensors="pt"
    )
    print(prompt_with_output_encoded.shape)

    # find the adv_slices and adv_indices in the prompt_with_output_encoded


    adv_indices = []
    for adv_slice in adv_slices:
        adv_index = list(range(len(prompt_with_output_encoded[0])))[adv_slice]
        adv_indices.extend(adv_index)

    target_length = len(adv_indices)

    print(adv_indices)
    print(tokenizer.batch_decode(prompt_with_output_encoded[..., adv_indices]))
    print(f"The target length is {target_length}")
    suffix_separator_index = (
        (prompt_with_output_encoded[0] == suffix_separator_id).nonzero()[-1].item()
    )

    output_slice = slice(
        len(prompt_with_template_without_output[0]), suffix_separator_index
    )
    output_indices = list(range(len(prompt_with_output_encoded[0])))[output_slice]
    output_ids = prompt_with_output_encoded[..., output_slice]
    output_tokens = tokenizer.batch_decode(output_ids)
    print(output_tokens)
    print(output_indices)


    return SimpleNamespace(
        prompt_with_output_encoded=prompt_with_output_encoded,
        adv_indices=adv_indices,
        output_indices=output_indices,
        target_length=target_length,
    )




def expand_for_broadcast_tensor(list_of_tensors, dim=0):
    sizes = {tensor.shape[dim] for tensor in list_of_tensors}
    max_size = max(sizes)
    sizes.discard(1)
    assert len(sizes) <= 1
    shape = [-1 for _ in list_of_tensors[0].shape]
    shape[dim] = max_size
    expanded_tensors = [tensor.expand(*shape) for tensor in list_of_tensors]
    return expanded_tensors


def expand_for_broadcast_list(list_of_lists):
    sizes = {len(_list) for _list in list_of_lists}
    max_size = max(sizes)
    sizes.discard(1)
    assert len(sizes) <= 1
    expanded_lists = [
        _list if len(_list) == max_size else [_list[0] for _ in range(max_size)]
        for _list in list_of_lists
    ]
    return expanded_lists
