"""
ThinkRL COPO Algorithm
======================

Count-based Online Preference Optimization (COPO) implementation.
Reference: "Online Preference Alignment for Language Models via Count-based Exploration" (ICLR 2025)

COPO extends Online DPO by adding a count-based exploration bonus derived from
a Coin Flipping Network (CFN). The CFN learns to predict the average of random
vectors assigned to each visitation. The magnitude of its output approximates
1/sqrt(N), serving as an exploration bonus to encourage visiting low-count states.

Author: EllanorAI
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import random
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import COPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class COPOConfig:
    """Configuration for COPO algorithm."""

    # Learning rate for Policy
    learning_rate: float = 5e-7

    # COPO Exploration parameters
    # Alpha controls the strength of the exploration bonus (see Table 3 in paper)
    alpha: float = 0.1

    # CFN Parameters (Table 4 in paper)
    cfn_learning_rate: float = 1e-4
    cfn_hidden_dim: int = 32
    cfn_output_dim: int = 20
    # Buffer size is critical for online RL to maintain visitation statistics
    cfn_buffer_size: int = 20000

    # DPO parameters
    beta: float = 0.1
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    label_smoothing: float = 0.0

    # Online generation
    num_generations_per_prompt: int = 4
    # Paper uses low temp for eval, but exploration requires sampling.
    generation_temperature: float = 1.0
    generation_top_p: float = 1.0
    max_new_tokens: int = 512

    # Training
    n_epochs: int = 1
    batch_size: int = 8
    cfn_batch_size: int = 32
    cfn_epochs: int = 2
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0

    # Reference model update
    update_reference_every: int = 1

    # Model Architecture (for CFN input size inference)
    # This should match the hidden size of the policy model (e.g. 4096 for Llama-3-8B)
    hidden_size: int = 4096


class CoinFlippingNetwork(nn.Module):
    """
    Coin Flipping Network (CFN) for pseudo-count estimation.

    Architecture based on paper Section 5.2 / Table 4:
    Input (LLM Hidden) -> Linear(32) -> LeakyReLU -> Linear(20)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Replay Buffer to store features of visited states.
    Critically, this allows the CFN to "remember" past visitations.
    As we sample the same state more times, the CFN target average converges to 0.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, features: torch.Tensor):
        """Add batch of features to buffer."""
        # Store on CPU to save VRAM
        for i in range(features.size(0)):
            self.buffer.append(features[i].detach().cpu())

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of features."""
        if len(self.buffer) < batch_size:
            items = list(self.buffer)
        else:
            items = random.sample(self.buffer, batch_size)
        return torch.stack(items)

    def __len__(self):
        return len(self.buffer)


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class COPOAlgorithm(BaseRLHFAlgorithm):
    """
    Count-based Online Preference Optimization (COPO).

    Paper Logic:
    - Objective: maximize J = J_DPO + alpha * E[1/sqrt(N)]
    - 1/sqrt(N) is approximated by ||CFN(s)|| / sqrt(d)
    - CFN is trained to regress to random Rademacher vectors {-1, 1}^d.
    - If a state is visited N times, CFN learns the average of N random vectors.
    - The magnitude of the average of N random vectors scales as 1/sqrt(N).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module | None = None,
        reward_model: nn.Module | None = None,
        config: COPOConfig | None = None,
        optimizer: Optimizer | None = None,
        tokenizer: Any | None = None,
        **kwargs,
    ):
        self.config = config or COPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=reference_model,
            optimizer=optimizer,
            learning_rate=self.config.learning_rate,
            kl_coeff=self.config.beta,
            clip_grad_norm=self.config.clip_grad_norm,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Tokenizer is mandatory for COPO (RM encoding, pairing)
        if self.tokenizer is None:
            raise ValueError("COPO requires a tokenizer to be passed to the constructor.")

        self.reward_model = reward_model
        if self.reward_model is None:
            raise ValueError("COPO requires a reward model to score generations.")

        # Ensure ref_model exists
        if self.ref_model is None:
            logger.info("No reference model provided. Creating a copy of the policy model.")
            self.ref_model = copy.deepcopy(self.policy_model)

        # Freeze reference model and reward model
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.reward_model.eval()
        self.reward_model.requires_grad_(False)

        # Initialize CFN
        self.cfn = CoinFlippingNetwork(
            input_dim=self.config.hidden_size,
            hidden_dim=self.config.cfn_hidden_dim,
            output_dim=self.config.cfn_output_dim,
        ).to(next(self.policy_model.parameters()).device)

        self.cfn_optimizer = torch.optim.AdamW(self.cfn.parameters(), lr=self.config.cfn_learning_rate)

        # CFN Replay Buffer
        self.cfn_buffer = ReplayBuffer(capacity=self.config.cfn_buffer_size)

        # Initialize Loss Function
        self.loss_fn = COPOLoss(
            beta=self.config.beta,
            alpha=self.config.alpha,
            cfn_output_dim=self.config.cfn_output_dim,
            loss_type=self.config.loss_type,
        )

        self.iteration_count = 0
        self.accumulated_loss = 0.0
        self.steps_since_update = 0

    def generate_responses(self, prompts: list[str]) -> list[list[str]]:
        """Generate multiple responses per prompt to find best/worst pairs."""
        n_samples = self.config.num_generations_per_prompt

        outputs = self.generate_rollouts(
            prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.generation_temperature,
            top_p=self.config.generation_top_p,
            num_return_sequences=n_samples,
            do_sample=True,  # Critical for exploration
            return_logprobs=False,
        )

        flat_texts = outputs["text"]

        # Reshape [P * N] -> [P, N]
        grouped_responses = []
        for i in range(0, len(flat_texts), n_samples):
            grouped_responses.append(flat_texts[i : i + n_samples])

        return grouped_responses

    def score_responses(self, prompts: list[str], responses: list[list[str]]) -> list[list[float]]:
        """Score generated responses using the reward model."""
        all_scores = []
        flat_inputs = []

        # Pylance fix: tokenizer is asserted not None in __init__
        tokenizer = self.tokenizer
        reward_model = self.reward_model
        assert tokenizer is not None
        assert reward_model is not None

        # Pre-format inputs for RM
        for prompt, res_list in zip(prompts, responses):
            for res in res_list:
                # Basic concatenation, update if chat template logic needed
                text = f"{prompt}{res}"
                flat_inputs.append(text)

        batch_size = self.config.batch_size
        device = next(reward_model.parameters()).device

        with torch.no_grad():
            for i in range(0, len(flat_inputs), batch_size):
                batch_texts = flat_inputs[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_new_tokens + 512,
                ).to(device)

                # Robust RM call
                try:
                    scores = reward_model(**inputs)
                except TypeError:
                    # Specific exception for unexpected kwargs
                    scores = reward_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                except Exception as e:
                    logger.warning(f"Reward model inference failed with: {e}. Attempting fallback.")
                    scores = reward_model(inputs["input_ids"])

                # Handle various output formats (Logits, Tuples, etc.)
                if hasattr(scores, "logits"):
                    scores = scores.logits
                elif isinstance(scores, tuple):
                    scores = scores[0]

                # Squeeze to ensure 1D tensor
                if isinstance(scores, torch.Tensor):
                    if scores.dim() > 1:
                        scores = scores.squeeze(-1)
                    scores = scores.float().cpu().tolist()
                elif isinstance(scores, list):
                    pass
                else:
                    # Scalar fallback
                    scores = [float(scores)] * len(batch_texts)

                all_scores.extend(scores)

        # Regroup scores
        grouped_scores = []
        n_samples = self.config.num_generations_per_prompt
        for i in range(0, len(all_scores), n_samples):
            grouped_scores.append(all_scores[i : i + n_samples])

        return grouped_scores

    def create_preference_pairs(
        self,
        prompts: list[str],
        responses: list[list[str]],
        rewards: list[list[float]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Create DPO pairs and collect ALL generated text for CFN.
        Returns: (Preference Pairs, All Visited Texts)
        """
        preference_data = []
        all_visited_texts = []

        for prompt, res_list, score_list in zip(prompts, responses, rewards):
            # 1. Collect all visited states for CFN
            for res in res_list:
                all_visited_texts.append(f"{prompt}{res}")

            if len(res_list) < 2:
                continue

            # 2. Create Pairs (Best vs Worst)
            paired = list(zip(res_list, score_list))
            best_res, best_score = max(paired, key=lambda x: x[1])
            worst_res, worst_score = min(paired, key=lambda x: x[1])

            # Only train if there is a margin
            if best_score > worst_score:
                preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": f"{prompt}{best_res}",
                        "rejected": f"{prompt}{worst_res}",
                        "margin": best_score - worst_score,
                    }
                )

        return preference_data, all_visited_texts

    def extract_features(self, batch_texts: list[str]) -> torch.Tensor:
        """
        Extract features (phi(x,y)) for CFN.
        Uses the last hidden state of the policy model (EOS token).
        """
        assert self.tokenizer is not None

        device = next(self.policy_model.parameters()).device
        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        policy_model = self.policy_model
        assert isinstance(policy_model, nn.Module)
        with torch.no_grad():
            outputs = policy_model(**inputs, output_hidden_states=True, return_dict=True)

        # Last hidden layer: [B, Seq, Dim]
        last_hidden = outputs.hidden_states[-1]

        # Pool EOS token
        seq_lengths = inputs.attention_mask.sum(dim=1) - 1
        seq_lengths = torch.clamp(seq_lengths, min=0, max=last_hidden.size(1) - 1)

        features = last_hidden[torch.arange(last_hidden.size(0)), seq_lengths]

        return features.detach()  # Detach to stop gradients flowing to Policy

    def train_cfn(self, all_texts: list[str]):
        """
        Train the Coin Flipping Network.
        """
        logger.info(f"Updating CFN with {len(all_texts)} new samples...")

        # 1. Feature Extraction (in batches)
        batch_size = self.config.cfn_batch_size
        new_features = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i : i + batch_size]
            feats = self.extract_features(batch)
            new_features.append(feats)

        if new_features:
            # Push new features to replay buffer
            self.cfn_buffer.push(torch.cat(new_features, dim=0))

        if len(self.cfn_buffer) < self.config.cfn_batch_size:
            return

        # 2. Training Loop
        self.cfn.train()
        device = next(self.cfn.parameters()).device
        loss_fn = nn.MSELoss()  #

        # Calculate iterations based on buffer size and epochs
        n_batches_per_epoch = max(1, len(self.cfn_buffer) // self.config.cfn_batch_size)
        n_updates = n_batches_per_epoch * self.config.cfn_epochs
        # Optional: Cap max updates if buffer gets huge
        n_updates = min(n_updates, 200)

        total_loss = 0.0
        for _ in range(n_updates):
            # Sample from buffer
            batch_feats = self.cfn_buffer.sample(self.config.cfn_batch_size).to(device)

            # Generate FRESH random targets: {-1, 1}^d
            targets = torch.randint(0, 2, (batch_feats.size(0), self.config.cfn_output_dim), device=device).float()
            targets = targets * 2 - 1  # Map [0, 1] -> [-1, 1]

            self.cfn_optimizer.zero_grad()
            preds = self.cfn(batch_feats)
            loss = loss_fn(preds, targets)
            loss.backward()
            self.cfn_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
        logger.info(f"CFN Updated ({n_updates} steps). Avg Loss: {avg_loss:.4f}")

    def prepare_batch(self, batch_data: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Tokenize batch for DPO."""
        assert self.tokenizer is not None

        chosen_texts = [x["chosen"] for x in batch_data]
        rejected_texts = [x["rejected"] for x in batch_data]

        device = next(self.policy_model.parameters()).device

        chosen_enc = self.tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        rejected_enc = self.tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Mask padding
        chosen_labels = chosen_enc["input_ids"].clone()
        chosen_labels[chosen_enc["attention_mask"] == 0] = -100
        rejected_labels = rejected_enc["input_ids"].clone()
        rejected_labels[rejected_enc["attention_mask"] == 0] = -100

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels": rejected_labels,
        }

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Compute COPO Loss = DPO Loss + Exploration Bonus.
        """
        # Unpack
        chosen_ids = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        rejected_ids = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]

        all_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        all_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        # 1. Policy Forward
        # We need hidden states for CFN
        policy_model = self.policy_model
        assert isinstance(policy_model, nn.Module)
        policy_out = policy_model(
            input_ids=all_ids,
            attention_mask=all_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        policy_log_probs = self.get_log_probs(
            policy_out.logits,
            torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0),
        )

        # 2. Reference Forward
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=all_ids, attention_mask=all_mask)
            if hasattr(ref_out, "logits"):
                ref_logits = ref_out.logits
            elif isinstance(ref_out, tuple):
                ref_logits = ref_out[0]
            else:
                ref_logits = ref_out

            ref_log_probs = self.get_log_probs(
                ref_logits,
                torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0),
            )

        # 3. DPO Loss Calculation
        batch_size = chosen_ids.size(0)

        # 4. Exploration Bonus (Only for CHOSEN responses)
        # Extract hidden features for chosen
        last_hidden = policy_out.hidden_states[-1]
        chosen_hidden = last_hidden[:batch_size]

        seq_len = chosen_mask.sum(dim=1) - 1
        seq_len = torch.clamp(seq_len, min=0, max=chosen_hidden.size(1) - 1)
        chosen_feats = chosen_hidden[torch.arange(batch_size), seq_len]

        # Compute Loss using COPOLoss
        total_loss, metrics = self.loss_fn(
            policy_log_probs=policy_log_probs,
            ref_log_probs=ref_log_probs,
            chosen_hidden_states=chosen_feats,
            cfn_model=self.cfn,
            batch_size=batch_size,
        )

        return metrics

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Training step with Gradient Accumulation."""
        self.policy_model.train()

        # Forward & Backward
        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]

        # Scale for accumulation
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        self.accumulated_loss += loss.item()
        self.steps_since_update += 1

        grad_norm = 0.0

        # Update weights if accumulation steps reached
        if self.steps_since_update >= self.config.gradient_accumulation_steps:
            if self.config.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.clip_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0
            self.accumulated_loss = 0.0

        metrics = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
        metrics["grad_norm"] = float(grad_norm)
        return metrics

    def run_iteration(self, prompts: list[str]) -> dict[str, Any]:
        """Execute one COPO iteration."""
        self.iteration_count += 1
        logger.info(f"Starting COPO Iteration {self.iteration_count}")

        # 1. Generate
        logger.info(f"Generating responses for {len(prompts)} prompts...")
        responses = self.generate_responses(prompts)

        # 2. Score
        logger.info("Scoring responses...")
        rewards = self.score_responses(prompts, responses)

        # 3. Create Pairs & Collect VISITED states
        raw_data, all_visited_texts = self.create_preference_pairs(prompts, responses, rewards)

        if not raw_data:
            logger.warning("No preference pairs created (insufficient margin/samples).")
            return {}

        # 4. Update CFN on ALL visited states
        self.train_cfn(all_visited_texts)

        # 5. Train Policy
        dataset = PreferenceDataset(raw_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.prepare_batch,
        )

        logger.info(f"Training Policy on {len(dataset)} pairs...")
        for _ in range(self.config.n_epochs):
            for batch in dataloader:
                metrics = self.training_step(batch)
                # FIX: use update_dict instead of update
                self.metrics_tracker.update_dict(metrics)

        # Gradient Accumulation Cleanup
        if self.steps_since_update > 0:
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0

        # 6. Update Reference
        if self.iteration_count % self.config.update_reference_every == 0:
            logger.info("Updating Reference Model...")
            self.ref_model.load_state_dict(self.policy_model.state_dict())

        return self.get_metrics()

    def state_dict(self) -> dict[str, Any]:
        """Save algorithm state including CFN and Buffer."""
        # Use base class implementation for common components
        state = super().state_dict()

        # Add COPO-specific state
        state.update(
            {
                "iteration_count": self.iteration_count,
                "cfn": self.cfn.state_dict(),
                "cfn_optimizer": self.cfn_optimizer.state_dict(),
                "cfn_buffer": list(self.cfn_buffer.buffer),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        """Load algorithm state."""
        # Use base class implementation
        super().load_state_dict(state, strict=strict)

        # Load COPO-specific state
        self.iteration_count = state.get("iteration_count", 0)

        if "cfn" in state:
            self.cfn.load_state_dict(state["cfn"])
        if "cfn_optimizer" in state:
            self.cfn_optimizer.load_state_dict(state["cfn_optimizer"])
        if "cfn_buffer" in state:
            self.cfn_buffer.buffer = deque(state["cfn_buffer"], maxlen=self.config.cfn_buffer_size)


def create_copo(
    policy_model,
    reference_model=None,
    reward_model=None,
    config: COPOConfig | None = None,
    **kwargs,
) -> COPOAlgorithm:
    """Factory function to create COPO algorithm."""
    return COPOAlgorithm(policy_model, reference_model, reward_model, config, **kwargs)


__all__ = ["COPOConfig", "COPOAlgorithm", "create_copo"]
