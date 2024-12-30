from transformers import Trainer
import torch


class SafeRLHFTrainer(Trainer):
    """
    Custom Trainer class to handle Safe RLHF training tasks.
    Supports SFT, Reward Model, Cost Model, and PPO-Lag methods.
    """

    def __init__(self, *args, method="sft", **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def compute_loss(self, model, inputs):
        """
        Compute loss based on the specified training method.
        """
        if self.method == "sft":
            return self.compute_crossentropy_loss(model, inputs)
        elif self.method == "reward":
            return self.compute_reward_loss(model, inputs)
        elif self.method == "cost":
            return self.compute_cost_loss(model, inputs)
        elif self.method == "ppo":
            return self.compute_ppo_loss(model, inputs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_crossentropy_loss(self, model, inputs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def compute_reward_loss(self, model, inputs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits.squeeze(-1)  # Assuming 1D outputs
        scores = inputs.get("scores")

        if scores is None:
            raise ValueError("Input dictionary must contain 'scores'.")

        # Ensure scores are float tensors
        scores = scores.float()

        # Optionally normalize logits (if required by your task)
        logits = torch.sigmoid(logits)

        loss_fct = torch.nn.MSELoss()
        return loss_fct(logits, scores)

    def compute_cost_loss(self, model, inputs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits.squeeze(-1)
        costs = inputs.get("costs")

        if costs is None:
            raise ValueError("Input dictionary must contain 'costs'.")

        costs = costs.float()
        logits = torch.sigmoid(logits)

        loss_fct = torch.nn.MSELoss()
        return loss_fct(logits, costs)

    def compute_ppo_loss(self, model, inputs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits
        value_preds = getattr(outputs, "value", None)  # Ensure 'value' exists

        if value_preds is None:
            raise ValueError("'value' predictions are missing from model outputs.")

        actions = inputs["actions"]
        old_log_probs = inputs["log_probs"]
        advantages = inputs["advantages"]
        target_values = inputs["values"]

        action_log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = action_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        ratios = torch.exp(action_log_probs - old_log_probs)
        clipped_ratios = torch.clamp(ratios, 1.0 - self.args.clip_range_ratio, 1.0 + self.args.clip_range_ratio)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(-1), target_values)

        entropy_loss = -torch.mean(torch.sum(torch.exp(action_log_probs) * action_log_probs, dim=-1))

        total_loss = (
            policy_loss +
            self.args.critic_coeff * value_loss -
            self.args.entropy_coeff * entropy_loss
        )
        return total_loss
