"""Fine-tune a compact text classifier locally or as an AML command job."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", default="data/raw/tickets.csv")
    parser.add_argument("--output-dir", default="outputs/model")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_data).dropna(subset=["text", "label"])
    labels = sorted(frame["label"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    frame["label"] = frame["label"].astype(str).map(label_to_id)

    dataset = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.autolog(log_models=False)
    training = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="no",
        report_to="mlflow",
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.model.merge_and_unload().save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "labels.json").write_text(json.dumps(label_to_id, indent=2))


if __name__ == "__main__":
    main()
