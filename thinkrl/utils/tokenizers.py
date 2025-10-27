"""
ThinkRL Tokenizer Utilities
============================

Tokenizer utilities for ThinkRL including:
- HuggingFace tokenizer integration
- Batch tokenization and encoding
- Special token handling
- Custom tokenizer support
- Tokenizer configuration and caching
- Multi-lingual tokenization

Author: Archit Sood @ EllanorAI
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import warnings

# Optional dependencies
try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    if not TYPE_CHECKING:
        # Create dummy classes for runtime when transformers is not installed
        PreTrainedTokenizer = None
        PreTrainedTokenizerFast = None
    warnings.warn(
        "transformers not available. Install with: pip install transformers"
    )

# Type checking imports
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


# Tokenizer type alias - works both with and without transformers
if TYPE_CHECKING:
    TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
else:
    TokenizerType = Any


class TokenizerConfig:
    """
    Configuration for tokenizer initialization.
    
    Attributes:
        model_name_or_path: Model name or path to tokenizer
        use_fast: Whether to use fast tokenizer
        padding_side: "right" or "left" padding
        truncation_side: "right" or "left" truncation
        max_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens
        return_attention_mask: Whether to return attention mask
        return_token_type_ids: Whether to return token type IDs
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        use_fast: bool = True,
        padding_side: str = "right",
        truncation_side: str = "right",
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        trust_remote_code: bool = False,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.use_fast = use_fast
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.trust_remote_code = trust_remote_code
        self.extra_kwargs = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "use_fast": self.use_fast,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens,
            "return_attention_mask": self.return_attention_mask,
            "return_token_type_ids": self.return_token_type_ids,
            "trust_remote_code": self.trust_remote_code,
            **self.extra_kwargs
        }


def get_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
    padding_side: str = "right",
    truncation_side: str = "right",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> TokenizerType:
    """
    Get a tokenizer from HuggingFace Hub or local path.
    
    Args:
        model_name_or_path: Model name or path (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
        use_fast: Whether to use fast (Rust-based) tokenizer
        padding_side: "right" or "left" for padding
        truncation_side: "right" or "left" for truncation
        cache_dir: Directory to cache tokenizer files
        trust_remote_code: Whether to trust remote code (for custom tokenizers)
        **kwargs: Additional arguments for AutoTokenizer.from_pretrained
        
    Returns:
        Initialized tokenizer
        
    Example:
        ```python
        # Get GPT-2 tokenizer
        tokenizer = get_tokenizer("gpt2")
        
        # Get LLaMA tokenizer with left padding
        tokenizer = get_tokenizer(
            "meta-llama/Llama-2-7b-hf",
            padding_side="left"
        )
        
        # Get Qwen tokenizer with trust_remote_code
        tokenizer = get_tokenizer(
            "Qwen/Qwen2.5-7B",
            trust_remote_code=True
        )
        ```
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers is required for tokenizer functionality. "
            "Install with: pip install transformers"
        )
    
    logger.info(f"Loading tokenizer from: {model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    # Configure padding and truncation
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"Set pad_token to unk_token: {tokenizer.pad_token}")
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Added new pad_token: [PAD]")
    
    logger.info(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"padding_side={tokenizer.padding_side}"
    )
    
    return tokenizer


def tokenize_text(
    text: Union[str, List[str]],
    tokenizer: TokenizerType,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = False,
    truncation: bool = True,
    add_special_tokens: bool = True,
    return_tensors: Optional[str] = "pt",
    return_attention_mask: bool = True,
    return_token_type_ids: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Tokenize text using a tokenizer.
    
    Args:
        text: Text string or list of strings to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy ("max_length", "longest", True, False)
        truncation: Whether to truncate sequences
        add_special_tokens: Whether to add special tokens (BOS, EOS)
        return_tensors: Format of return tensors ("pt", "np", None)
        return_attention_mask: Whether to return attention mask
        return_token_type_ids: Whether to return token type IDs
        **kwargs: Additional tokenizer arguments
        
    Returns:
        Dictionary with input_ids, attention_mask, etc.
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        # Tokenize single text
        encoded = tokenize_text(
            "Hello, world!",
            tokenizer,
            max_length=512,
            padding="max_length"
        )
        
        # Tokenize batch
        encoded = tokenize_text(
            ["Hello!", "How are you?"],
            tokenizer,
            padding=True
        )
        ```
    """
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        return_tensors=return_tensors,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        **kwargs
    )
    
    return encoded


def tokenize_batch(
    texts: List[str],
    tokenizer: TokenizerType,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = True,
    truncation: bool = True,
    add_special_tokens: bool = True,
    return_tensors: str = "pt",
    batch_size: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Tokenize a batch of texts efficiently.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        add_special_tokens: Whether to add special tokens
        return_tensors: Format of return tensors
        batch_size: Process in mini-batches if specified
        **kwargs: Additional tokenizer arguments
        
    Returns:
        Dictionary with batched encodings
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        texts = ["Text 1", "Text 2", "Text 3", ...]
        encoded = tokenize_batch(
            texts,
            tokenizer,
            max_length=512,
            padding="longest"
        )
        ```
    """
    if batch_size is None or len(texts) <= batch_size:
        # Tokenize all at once
        return tokenize_text(
            texts,
            tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs
        )
    
    # Tokenize in mini-batches
    import torch
    
    all_input_ids = []
    all_attention_mask = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenize_text(
            batch_texts,
            tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs
        )
        
        all_input_ids.append(encoded["input_ids"])
        all_attention_mask.append(encoded["attention_mask"])
    
    # Concatenate batches
    result = {
        "input_ids": torch.cat(all_input_ids, dim=0),
        "attention_mask": torch.cat(all_attention_mask, dim=0),
    }
    
    return result


def decode_tokens(
    token_ids: Union[List[int], List[List[int]]],
    tokenizer: TokenizerType,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
) -> Union[str, List[str]]:
    """
    Decode token IDs back to text.
    
    Args:
        token_ids: Token IDs or list of token ID sequences
        tokenizer: Tokenizer to use
        skip_special_tokens: Whether to skip special tokens in output
        clean_up_tokenization_spaces: Whether to clean up spaces
        **kwargs: Additional decode arguments
        
    Returns:
        Decoded text or list of texts
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        # Decode single sequence
        text = decode_tokens([1, 2, 3, 4], tokenizer)
        
        # Decode batch
        texts = decode_tokens([[1, 2, 3], [4, 5, 6]], tokenizer)
        ```
    """
    # Check if batch or single sequence
    if isinstance(token_ids[0], (list, tuple)):
        # Batch decoding
        return tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )
    else:
        # Single sequence decoding
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )


def get_special_tokens(tokenizer: TokenizerType) -> Dict[str, Any]:
    """
    Get special tokens from tokenizer.
    
    Args:
        tokenizer: Tokenizer to extract special tokens from
        
    Returns:
        Dictionary of special tokens and their IDs
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        special_tokens = get_special_tokens(tokenizer)
        print(special_tokens["pad_token"])  # "<|endoftext|>"
        print(special_tokens["pad_token_id"])  # 50256
        ```
    """
    return {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "sep_token": tokenizer.sep_token,
        "sep_token_id": tokenizer.sep_token_id,
        "cls_token": tokenizer.cls_token,
        "cls_token_id": tokenizer.cls_token_id,
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
    }


def add_special_tokens(
    tokenizer: TokenizerType,
    special_tokens_dict: Dict[str, str],
    resize_embeddings: bool = True
) -> int:
    """
    Add special tokens to tokenizer.
    
    Args:
        tokenizer: Tokenizer to add tokens to
        special_tokens_dict: Dictionary of special tokens
        resize_embeddings: Whether to return info for resizing embeddings
        
    Returns:
        Number of tokens added
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        # Add custom tokens
        num_added = add_special_tokens(
            tokenizer,
            {
                "additional_special_tokens": ["<|user|>", "<|assistant|>"]
            }
        )
        
        print(f"Added {num_added} tokens")
        # Then resize model embeddings: model.resize_token_embeddings(len(tokenizer))
        ```
    """
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added > 0:
        logger.info(f"Added {num_added} special tokens to tokenizer")
        if resize_embeddings:
            logger.info(
                f"Remember to resize model embeddings: "
                f"model.resize_token_embeddings(len(tokenizer))"
            )
    
    return num_added


def tokenize_conversation(
    messages: List[Dict[str, str]],
    tokenizer: TokenizerType,
    system_prefix: str = "",
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
    separator: str = "\n",
    add_generation_prompt: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Tokenize a conversation in chat format.
    
    Args:
        messages: List of message dicts with "role" and "content"
        tokenizer: Tokenizer to use
        system_prefix: Prefix for system messages
        user_prefix: Prefix for user messages
        assistant_prefix: Prefix for assistant messages
        separator: Separator between messages
        add_generation_prompt: Whether to add assistant prefix at end
        **kwargs: Additional tokenization arguments
        
    Returns:
        Tokenized conversation
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        encoded = tokenize_conversation(
            messages,
            tokenizer,
            add_generation_prompt=True
        )
        ```
    """
    # Try using chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            return tokenize_text(text, tokenizer, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to use chat template: {e}, falling back to manual formatting")
    
    # Manual formatting
    formatted_messages = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            formatted_messages.append(f"{system_prefix}{content}")
        elif role == "user":
            formatted_messages.append(f"{user_prefix}{content}")
        elif role == "assistant":
            formatted_messages.append(f"{assistant_prefix}{content}")
        else:
            formatted_messages.append(content)
    
    # Join messages
    text = separator.join(formatted_messages)
    
    # Add generation prompt
    if add_generation_prompt:
        text += separator + assistant_prefix
    
    return tokenize_text(text, tokenizer, **kwargs)


def prepare_input_for_generation(
    prompt: str,
    tokenizer: TokenizerType,
    max_length: int = 512,
    device: str = "cpu",
    **kwargs
) -> Dict[str, Any]:
    """
    Prepare input for text generation.
    
    Args:
        prompt: Text prompt
        tokenizer: Tokenizer to use
        max_length: Maximum input length
        device: Device to place tensors on
        **kwargs: Additional tokenization arguments
        
    Returns:
        Prepared input dictionary
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        inputs = prepare_input_for_generation(
            "Once upon a time",
            tokenizer,
            max_length=512,
            device="cuda"
        )
        
        # Use with model
        outputs = model.generate(**inputs, max_new_tokens=50)
        ```
    """
    import torch
    
    encoded = tokenize_text(
        prompt,
        tokenizer,
        max_length=max_length,
        padding=False,
        truncation=True,
        return_tensors="pt",
        **kwargs
    )
    
    # Move to device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    return encoded


def count_tokens(
    text: Union[str, List[str]],
    tokenizer: TokenizerType,
    add_special_tokens: bool = True
) -> Union[int, List[int]]:
    """
    Count number of tokens in text(s).
    
    Args:
        text: Text or list of texts
        tokenizer: Tokenizer to use
        add_special_tokens: Whether to count special tokens
        
    Returns:
        Token count or list of token counts
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        # Single text
        count = count_tokens("Hello, world!", tokenizer)
        print(f"Token count: {count}")
        
        # Batch
        counts = count_tokens(["Hello!", "How are you?"], tokenizer)
        print(f"Token counts: {counts}")
        ```
    """
    if isinstance(text, str):
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
        return len(encoded["input_ids"])
    else:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
        return [len(ids) for ids in encoded["input_ids"]]


def truncate_to_token_limit(
    text: str,
    tokenizer: TokenizerType,
    max_tokens: int,
    add_special_tokens: bool = True,
    side: str = "right"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        tokenizer: Tokenizer to use
        max_tokens: Maximum number of tokens
        add_special_tokens: Whether to account for special tokens
        side: "right" or "left" truncation
        
    Returns:
        Truncated text
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        long_text = "..." # very long text
        truncated = truncate_to_token_limit(
            long_text,
            tokenizer,
            max_tokens=512
        )
        ```
    """
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    token_ids = encoded["input_ids"]
    
    if len(token_ids) <= max_tokens:
        return text
    
    # Truncate tokens
    if side == "right":
        truncated_ids = token_ids[:max_tokens]
    else:  # left
        truncated_ids = token_ids[-max_tokens:]
    
    # Decode back to text
    truncated_text = tokenizer.decode(
        truncated_ids,
        skip_special_tokens=not add_special_tokens
    )
    
    return truncated_text


def get_tokenizer_info(tokenizer: TokenizerType) -> Dict[str, Any]:
    """
    Get comprehensive information about a tokenizer.
    
    Args:
        tokenizer: Tokenizer to inspect
        
    Returns:
        Dictionary with tokenizer information
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        info = get_tokenizer_info(tokenizer)
        
        print(f"Vocab size: {info['vocab_size']}")
        print(f"Model max length: {info['model_max_length']}")
        ```
    """
    info = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "is_fast": tokenizer.is_fast,
        "name_or_path": tokenizer.name_or_path,
        "special_tokens": get_special_tokens(tokenizer),
    }
    
    # Add tokenizer type
    if hasattr(tokenizer, "__class__"):
        info["tokenizer_class"] = tokenizer.__class__.__name__
    
    return info


def save_tokenizer(
    tokenizer: TokenizerType,
    save_directory: Union[str, Path],
    **kwargs
) -> Tuple[str, ...]:
    """
    Save tokenizer to directory.
    
    Args:
        tokenizer: Tokenizer to save
        save_directory: Directory to save to
        **kwargs: Additional save arguments
        
    Returns:
        Tuple of saved file paths
        
    Example:
        ```python
        tokenizer = get_tokenizer("gpt2")
        
        # Modify tokenizer...
        add_special_tokens(tokenizer, {"additional_special_tokens": ["<custom>"]})
        
        # Save
        saved_files = save_tokenizer(tokenizer, "./my_tokenizer")
        ```
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    
    saved_files = tokenizer.save_pretrained(str(save_directory), **kwargs)
    
    logger.info(f"Tokenizer saved to: {save_directory}")
    return saved_files


def load_tokenizer(
    load_directory: Union[str, Path],
    **kwargs
) -> TokenizerType:
    """
    Load tokenizer from directory.
    
    Args:
        load_directory: Directory to load from
        **kwargs: Additional load arguments
        
    Returns:
        Loaded tokenizer
        
    Example:
        ```python
        tokenizer = load_tokenizer("./my_tokenizer")
        ```
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required. Install with: pip install transformers")
    
    tokenizer = AutoTokenizer.from_pretrained(str(load_directory), **kwargs)
    
    logger.info(f"Tokenizer loaded from: {load_directory}")
    return tokenizer


# Public API
__all__ = [
    "TokenizerConfig",
    "get_tokenizer",
    "tokenize_text",
    "tokenize_batch",
    "decode_tokens",
    "get_special_tokens",
    "add_special_tokens",
    "tokenize_conversation",
    "prepare_input_for_generation",
    "count_tokens",
    "truncate_to_token_limit",
    "get_tokenizer_info",
    "save_tokenizer",
    "load_tokenizer",
]