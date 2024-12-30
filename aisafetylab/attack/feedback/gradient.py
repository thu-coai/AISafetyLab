from aisafetylab.utils import prepare_labels
from aisafetylab.utils import prepare_input_embeddings
from aisafetylab.utils import forward_with_batches
from types import SimpleNamespace
import torch
from torch import nn
import gc
from aisafetylab.utils import get_embeddings, get_embedding_matrix
def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    return one_hot.grad.clone()
class GradientFeedback:
    @staticmethod
    def get_grad_of_control_tokens(
        model, 
        input_ids, 
        control_slice, 
        target_slice, 
        loss_slice = None,
        ):
        input_ids = input_ids.to(model.device)
        if loss_slice is None:
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        return token_gradients(
            model = model,
            input_ids = input_ids,
            input_slice = control_slice,
            target_slice = target_slice,
            loss_slice = loss_slice,
        )
    @staticmethod
    def compute_one_hot_gradient(model, embedding_matrix, prompt_with_output_encoded, output_indices, adv_indices):
        labels = prepare_labels(prompt_with_output_encoded, output_indices, ignore_index=-100)
        one_hot, input_embeddings = prepare_input_embeddings(
            prompt_with_output_encoded,
            embedding_matrix,
            adv_indices
        )
        outputs = forward_with_batches(
            model=model, 
            input_embeddings=input_embeddings, 
            labels=labels, 
            do_backward=True)
        one_hot = one_hot
        outputs = outputs
        torch.cuda.empty_cache()
        gc.collect()

        return SimpleNamespace(
            one_hot = one_hot,
            outputs = outputs
        )