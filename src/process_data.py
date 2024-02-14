from transformers import AutoTokenizer

def preprocess(labels, tokenized_inputs, task, do_mask=False):
    result = {}

    old_labels = labels
    new_labels = []

    target_ids = [i for i, label in enumerate(old_labels) if label]
    if do_mask:
        tokens = tokenized_inputs.tokens
        for target_idx in target_ids:
            tokens[target_idx] = '[MASK]'
        result['masked_tokens'] = tokens

    current_word = None
    for word_id in tokenized_inputs.word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            new_labels.append(-100 if word_id is None else labels[word_id])
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def preprocess_batch(data_batch, tokenizer, task, do_mask=False):
    tokenized_inputs = tokenizer(data_batch['tokens'], truncation=True, is_split_into_words=True)

    all_labels = data_batch["frame_tags"] if task == 'frames' else data_batch["is_target"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        new_labels.append(preprocess(all_labels[i], tokenized_inputs[i], task))

    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs


def prepare_data(dataset, pretrained_model, task=None):
    if 'roberta' in pretrained_model:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized_datasets = dataset.map(
        preprocess_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, 'do_mask': False, 'task': task},
        remove_columns=dataset["train"].column_names
    )

    return tokenized_datasets
