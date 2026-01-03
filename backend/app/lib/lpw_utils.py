import re
import torch
import logging

logger = logging.getLogger(__name__)

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

def get_prompts_with_weights(tokenizer, prompt: list[str], max_length: int):
    """
    Tokenize a list of prompts and return its tokens with weights of each token.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights

def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, chunk_length=77):
    """
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    Using simple padding without 'no_boseos_middle' complexity for now, or mimicking 'False' behavior.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    
    for i in range(len(tokens)):
        # Pad tokens: BOS + tokens + EOS + PAD...
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        
        # Pad weights: 1.0 + weights + 1.0 + 1.0...
        weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        
        # If we need chunking support later, we might need to adjust this padding logic
        # to insert BOS/EOS at chunk boundaries if we split.
        
    return tokens, weights

def chunk_tokens_and_weights(tokens, weights, chunk_length=77):
    """
    Split long tokens and weights into chunks of chunk_length.
    Assumes tokens are already padded to total_length.
    Returns list of chunks.
    """
    # Actually, the logic in LPW script handles splitting *before* passing to encoder or *inside* encoder helper.
    # Here we want to break them so we can pass them to a standard CLIP model.
    # tokens is list of lists
    
    chunked_tokens = []
    chunked_weights = []
    
    for seq_tokens, seq_weights in zip(tokens, weights):
        # We assume seq_tokens has BOS at 0 and EOS somewhere.
        # Use simple chunking with overlap? Or just hard split?
        # CLIP expects 77 tokens.
        # If we have 150 tokens.
        # Chunk 1: inputs[0:77]
        # Chunk 2: inputs[77:154]
        # BUT we need BOS/EOS in *each* chunk for standard CLIP?
        # LPW script does: <BOS> CHUNK <EOS> <PAD>...
        
        # For simplicity in this metric calculator:
        # We will take the full list of tokens (without BOS/EOS padding from previous step if possible, 
        # or we remove them), break into 75-token chunks, and wrap each with BOS/EOS.
        
        inner_tokens = seq_tokens[1:-1] # Remove BOS/EOS assuming they are there? 
        # Wait, get_prompts_with_weights returns tokens WITHOUT BOS/EOS.
        # pad_tokens_and_weights adds them.
        
        pass 
        # (I will implement the chunking logic inside metrics.py using get_prompts_with_weights directly)
    return
