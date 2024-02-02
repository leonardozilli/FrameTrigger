#%%

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from evaluate import compute_metrics
from fn17 import load_dataset_hf


#%%
def train(pretrained_model, dataset, epochs, batch_size, lr, model_output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    id2label = {0: "Not a target", 1: "Target"}
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
        CHECKPOINT,
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
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()


#%%

if __name__ == '__main__':
    dataset = load_dataset_hf(flatten=True)

    OUT_DIR = './models/'
    CHECKPOINT = 'bert-base-cased'
    N_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5

    train(pretrained_model=CHECKPOINT, dataset=dataset,
        epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, model_output_path='models/')
