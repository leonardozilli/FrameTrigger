#%%

import torch
from nltk.corpus import framenet as fn
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from .evaluate import compute_metrics, compute_metrics_binary
from .fn17 import load_dataset_hf
from .process_data import prepare_data


#%%
def train(pretrained_model, task, dataset, epochs, batch_size, lr, model_output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    id2label = {i: frame['name'] for i, frame in enumerate(fn.frames())}
    id2label[-1] = "None"
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        evaluation_strategy='epoch',
        save_strategy='no',
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=True,
        output_dir=model_output_path,
        overwrite_output_dir=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model,
        id2label=id2label,
        label2id=label2id
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if task == 'frames' else compute_metrics_binary
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(model_output_path)


def test(pretrained_model, test_dataset, task):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
    model.eval()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if task == 'frames' else compute_metrics_binary
    )

    results = trainer.predict(test_dataset)

    return results[2]


#%%
if __name__ == '__main__':
    dataset = load_dataset_hf(flatten=True)

    OUT_DIR = './models/'
    CHECKPOINT = 'bert-base-cased'
    N_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    TASK = 'frames'

    tokenized_dataset = prepare_data(dataset, CHECKPOINT)

    train(pretrained_model=CHECKPOINT, task=TASK, dataset=tokenized_dataset,
        epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, model_output_path=OUT_DIR)
    
    test_results = test(OUT_DIR, tokenized_dataset['test'])

    print('\n==== Test Results ====')
    print(test_results)
