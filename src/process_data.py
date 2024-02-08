from transformers import AutoTokenizer

def preprocess(labels, tokenized_inputs, task, do_mask=False):
    result = {}

    old_labels = labels
    new_labels = []
    target_ids = []
    is_target = []

    for i, label in enumerate(old_labels):
        if label:
            target_ids.append(i)

    if do_mask:
        tokens = tokenized_inputs.tokens
        for target_idx in target_ids:
            tokens[target_idx] = '[MASK]'
        result['masked_tokens'] = tokens

    previous_word_id = None
    for word_id in tokenized_inputs.word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != previous_word_id:
            new_labels.append(old_labels[word_id])
        else:
            new_labels.append(-100)

        #is_target.append(1) if word_id in target_ids else is_target.append(0) #-100??
        is_target.append(1) if word_id in target_ids else is_target.append(0 if word_id is not None else -100)
        previous_word_id = word_id


    #result['labels'] = is_target
    #tokenized_input['token_type_ids'] = is_target
    return is_target if task == 'targets' else new_labels

def preprocess_batch(data_batch, tokenizer, task, do_mask=False):
    tokenized_inputs = tokenizer(data_batch['tokens'], truncation=True, is_split_into_words=True)

    all_labels = data_batch["frame_tags"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        new_labels.append(preprocess(all_labels[i], tokenized_inputs[i], task))

    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs


def prepare_data(dataset, pretrained_model, task=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized_datasets = dataset.map(
        preprocess_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, 'do_mask': False, 'task': task},
        remove_columns=dataset["train"].column_names
    )

    return tokenized_datasets
