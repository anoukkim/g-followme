from transformers import Trainer, TrainingArguments
from utils import collate_fn

def train_model(model, processor, train_dataset, val_dataset, config):
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        evaluation_strategy="epoch",
        logging_dir=config["logging_dir"],
        num_train_epochs=config["num_train_epochs"],
        save_strategy="epoch",
        logging_steps=config["evaluation_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
