from aisafetylab.utils import find_adv_indices_and_output_indices
class ComponentBreaker:
    def init_input(self, tokenizer, prefix_seperator_id, suffix_seperator_id, prefix_seperator, suffix_seperator, messages_components):
        return find_adv_indices_and_output_indices(tokenizer, prefix_seperator_id, suffix_seperator_id, prefix_seperator, suffix_seperator, messages_components)