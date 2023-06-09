import torch


def remap_input_to_cf_vocab(input_ids, tokenizer, cf_tokenizer):
    """
    Remap an input from the original tokenizer (e.g. bert tokenizer) to
    another tokenizer (e.g. t5 tokenizer). For example,
    for the word 'CURIOUS', we would get the following pieces:
        bert: ['C', '##UR', '##IO', '##US']
        t5  : ['▁C', 'URI', 'OUS']

    What we want is to map bert and t5 inputs s.t. len(bert_input) = len(t5_input).
    To achieve that, we tokenize each word individually and drop/repeat the
    position corresponding to the last piece. For the above example, we would
    get repeat_interleave counts such that:
        input:    ['C', '##UR', '##IO']
        cf_input: ['▁C', 'URI', 'OUS']

    If instead we originally got a tokenization like this:
        bert: ['C', '##URIOUS']
        t5  : ['▁C', 'URI', 'OUS']

    we would get repeat_interleave counts such that:
        input:    ['C', '##URIOUS', '##URIOUS']
        cf_input: ['▁C', 'URI', 'OUS']

    To achieve this, we just need the frequency that each token should be
    repeated in a interleaved manner (i.e. repeat_interleave counts).

    Args:
        input_ids: original input ids got from the factual tokenizer
        tokenizer: factual tokenizer
        cf_tokenizer: counterfactual tokenizer

    Returns:
        cf_input_counts: the frequency that each input_id will be repeated to
    """
    ff_special_tokens_vals = tokenizer.special_tokens_map.values()
    ff_special_tokens_keys = tokenizer.special_tokens_map.keys()
    ff_has_bos = 'cls_token' in ff_special_tokens_keys or 'bos_token' in ff_special_tokens_keys
    ff_has_eos = 'sep_token' in ff_special_tokens_keys or 'eos_token' in ff_special_tokens_keys
    ff_bos_token = tokenizer.cls_token if 'cls_token' in ff_special_tokens_keys else tokenizer.bos_token
    ff_eos_token = tokenizer.sep_token if 'sep_token' in ff_special_tokens_keys else tokenizer.eos_token
    cf_special_tokens_keys = cf_tokenizer.special_tokens_map.keys()
    cf_has_bos = 'cls_token' in cf_special_tokens_keys or 'bos_token' in cf_special_tokens_keys
    cf_has_eos = 'sep_token' in cf_special_tokens_keys or 'eos_token' in cf_special_tokens_keys
    cf_input_counts = []
    for x_s in tokenizer.batch_decode(input_ids):
        x_counts_inner = []
        for word in x_s.split():
            # handle special tokens (e.g., CLS, SEP, PAD, UNK)
            if word in ff_special_tokens_vals:
                p_f = [0]
                p_cf = [0]
            else:
                a = 1 if ff_has_bos else 0
                b = -1 if ff_has_eos else None
                p_f = tokenizer(word)['input_ids'][a:b]  # remove [cls] and [sep]
                a = 1 if cf_has_bos else 0
                b = -1 if cf_has_eos else None
                p_cf = cf_tokenizer(word)['input_ids'][a:b]  # remove <s> and </s>

            # set c so that we repeat last piece
            if len(p_f) < len(p_cf):
                c = [1] * (len(p_f) - 1) + [1 + len(p_cf) - len(p_f)]
            # set c so that we drop the last pieces
            elif len(p_f) > len(p_cf):
                c = [1] * len(p_cf) + [0]*(len(p_f) - len(p_cf))
            # do nothing, they match sizes
            else:
                if not cf_has_bos and word == ff_bos_token:
                    c = [0]  # drop [CLS] since some models dont have a bos token
                elif not cf_has_eos and word == ff_eos_token:
                    c = [0]  # drop [SEP] since some models dont have a eos token
                else:
                    c = [1] * len(p_f)
            x_counts_inner.extend(c)
        cf_input_counts.append(torch.as_tensor(x_counts_inner))
    return cf_input_counts


def remap_input_to_cf_vocab_brute_force(bert_tokenizer, t5_tokenizer, x, z, mask):
    """
    Do the remapping at the piece level instead of the word level.
    This method leads to inputs with more word pieces.
    """
    x_new = []
    x_counts = []
    for x_i in x:
        x_new_inner = []
        x_counts_inner = []
        for word in bert_tokenizer.convert_ids_to_tokens(x_i):
            word = word.replace('##', '')
            if word == bert_tokenizer.cls_token:
                p_bert = [bert_tokenizer.cls_token_id]
                p_t5 = [t5_tokenizer.vocab['X']]
            elif word == bert_tokenizer.sep_token:
                p_bert = [bert_tokenizer.sep_token_id]
                p_t5 = [t5_tokenizer.eos_token_id]
            elif word == bert_tokenizer.pad_token:
                p_bert = [bert_tokenizer.pad_token_id]
                p_t5 = [t5_tokenizer.pad_token_id]
            elif word == bert_tokenizer.unk_token:
                p_bert = [bert_tokenizer.unk_token_id]
                p_t5 = [t5_tokenizer.unk_token_id]
            else:
                p_bert = bert_tokenizer(word)['input_ids'][1:-1]  # remove [cls] and [sep]
                p_t5 = t5_tokenizer(word)['input_ids'][:-1]  # remove </s>
            if len(p_bert) < len(p_t5):
                c = [1] * (len(p_bert) - 1) + [1 + len(p_t5) - len(p_bert)]
            elif len(p_bert) > len(p_t5):
                c = [1] * len(p_t5) + [0]*(len(p_bert) - len(p_t5))
            else:
                c = [1] * len(p_bert)
            x_counts_inner.extend(c)
            x_new_inner.extend(p_t5)
        x_counts.append(torch.as_tensor(x_counts_inner))
        x_new.append(torch.as_tensor(x_new_inner))
    z_new = [z[i].repeat_interleave(x_counts[i], dim=-1) for i in range(len(x))]
    mask_new = [mask[i].repeat_interleave(x_counts[i], dim=-1) for i in range(len(x))]
    x_new_pt = torch.nn.utils.rnn.pad_sequence(x_new, batch_first=True, padding_value=t5_tokenizer.pad_token_id)
    z_new_pt = torch.nn.utils.rnn.pad_sequence(z_new, batch_first=True, padding_value=0)
    mask_new_pt = torch.nn.utils.rnn.pad_sequence(mask_new, batch_first=True, padding_value=0)
    return x_new_pt, z_new_pt, mask_new_pt.bool()


def concat_sequences(input_ids_1, input_ids_2):
    """
    Concatenates the input sequences.
    """
    # Each sequence is tokenized as:
    # <bos> <token> <token> ... <token> <eos>
    # So the concatenation will result in:
    # <bos> <mt> <eos> <bos> <src> <eos> <bos> <ref> <eos> ...
    # for some model, <bos> and <eos> might be None, so they are not concatenated.
    x1 = torch.as_tensor(input_ids_1)
    x2 = torch.as_tensor(input_ids_2)
    z1 = torch.zeros_like(x1)
    z2 = torch.ones_like(x2)
    input_ids = torch.cat([x1, x2], dim=-1)
    z1[-1] = 1  # set the last token to be part of the second sequence
    token_type_ids = torch.cat([z1, z2], dim=-1)
    return input_ids, token_type_ids


def get_lp_name(l):
    langs = {
        'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian',
        'bg': 'Bulgarian', 'bg-Latn': 'Bulgarian (Latin)', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano',
        'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
        'el-Latn': 'Greek (Latin)', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian',
        'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French',
        'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati',
        'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hi-Latn': 'Hindi (Latin script)', 'hmn': 'Hmong, Mong',
        'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic',
        'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'ja-Latn': 'Japanese (Latin)', 'jv': 'Javanese',
        'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish',
        'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian',
        'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi',
        'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian',
        'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian',
        'ru': 'Russian', 'ru-Latn': 'Russian (Latin)', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak',
        'sl': 'Slovenian', 'sm': 'San Marino', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian',
        'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu',
        'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu',
        'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese',
        'zh-Latn': 'Chinese (Latin)', 'zu': 'Zulu',
    }
    return langs.get(l, l)


def token_type_ids_from_input_ids(input_ids, sep_id=1):
    """
    Creates a tensor that encodes the token type ids given the input ids.

    Args:
        input_ids (torch.Tensor or list): the input ids.
        sep_id (int): the id of the separator token.

    Returns:
        torch.LongTensor: the token type ids.
    """
    m = torch.tensor(input_ids).roll(1, dims=0) == sep_id
    m[0] = False
    return torch.cumsum(m, dim=0).clamp(max=1)


def fix_saved_inputs_for_t5(input_ids, sep_id=1):
    """
    Fixes the saved inputs for T5, which may contain extras </s> tokens.

    Args:
        input_ids (torch.Tensor or list): the input ids.
        sep_id (int): the id of the separator token.

    Returns:
        torch.LongTensor: the fixed input ids.
    """
    x = torch.tensor(input_ids)
    is_eos = x == sep_id
    # is_fused = x == x.roll(-1, dims=-1)
    # valid_pos = ~(is_eos & is_fused)
    c = torch.cumsum(is_eos, dim=0)
    valid_pos = c <= 2
    return x[valid_pos]
