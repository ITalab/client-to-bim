import json
from pdb import run
import random
from re import split
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from transformers import (
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
)
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime




with open("db_full_augmented.json", "r", encoding="utf8") as f:
    db = json.load(f)

# flatten into list of (text,label) as before, then ignore labels for MLM
dataset = []
for lbl, texts in db.items():
    for txt in texts:
        dataset.append((txt, lbl))
# stratified sampling: take the k-th texts per label
k = 56
random.seed(42)
random.shuffle(dataset)
train_data = []
val_data = []

split_ratio = 0.8


for i in range(0, len(dataset), k):
    block = dataset[i : i + k]
    # take the first 80% of this block as test, rest as train
    n_test = int(split_ratio * len(block))
    val_data.extend(block[:n_test])
    train_data.extend(block[n_test:])


class MaskedTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        # prune unused columns, we only need the text for MLM
        self.texts = [text for text, _ in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # tokenize to fixed‐length, return raw input_ids + attention_mask
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# dynamic masking rate scheduling
class DynamicMaskingCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, init_prob, final_prob, total_steps):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=init_prob)
        self.init_prob = init_prob
        self.final_prob = final_prob
        self.total_steps = total_steps
        self.step = 0

    def torch_call(self, examples):
        # linear schedule of mlm_probability
        self.mlm_probability = self.init_prob + (
            self.final_prob - self.init_prob
        ) * min(self.step / self.total_steps, 1.0)
        batch = super().torch_call(examples)
        self.step += 1
        return batch


# span masking
class SpanMaskingCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, mean_span_length=3.0):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.mean_span_length = mean_span_length

    def mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        batch_size, seq_len = labels.size()
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        num_to_mask = max(1, int(self.mlm_probability * seq_len))
        for i in range(batch_size):
            while mask[i].sum().item() < num_to_mask:
                span_len = max(1, np.random.poisson(self.mean_span_length))
                start = random.randint(0, seq_len - span_len)
                mask[i, start : start + span_len] = True
        inputs[mask] = tokenizer.mask_token_id
        labels[~mask] = -100
        return inputs, labels


# informative token masking
class InforMaskingCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, pmi_scores, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.pmi_scores = pmi_scores

    def mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        batch_size, seq_len = labels.size()
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for i in range(batch_size):
            scores = self.pmi_scores[i][:seq_len]
            topk = int(self.mlm_probability * seq_len)
            mask_positions = torch.topk(scores, topk).indices
            mask[i, mask_positions] = True
        inputs[mask] = tokenizer.mask_token_id
        labels[~mask] = -100
        return inputs, labels


def get_writer(masking_prob):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"./train/mlm-pretrain-{run_id}-{masking_prob}")
    return writer


#  prepare tokenizer, data collator, dataloaders

MODEL_NAME = "bert-base-uncased"
TOKENIZER_NAME = MODEL_NAME
if os.path.exists("./train/bert-mlm-checkpoint_1"):
    MODEL_NAME = "./train/bert-mlm-checkpoint_1"
else:
    print("Warning: No checkpoint found, using original model.")

tokenizer = BertTokenizerFast.from_pretrained(f"./{TOKENIZER_NAME}-extended")

masking_prob = 0.15
for masking_prob in [0.15]:
    print(f"Masking probability: {masking_prob}")
    writer = get_writer(masking_prob)
    writer.add_text("Model name", MODEL_NAME)
    writer.add_text("Split ratio", str(split_ratio))
    writer.add_text("Number of texts per label", str(k))

    # This will handle random masking of 15% of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=masking_prob,
    )

    wwm_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=masking_prob
    )

    span_collator = SpanMaskingCollator(tokenizer=tokenizer)

    dynamic_collator = DynamicMaskingCollator(
        tokenizer=tokenizer,
        init_prob=masking_prob,
        final_prob=masking_prob,
        total_steps=10000,
    )

    # data_collator = dynamic_collator
    writer.add_text("Masking probability", str(masking_prob))

    train_ds = MaskedTextDataset(train_data, tokenizer)
    val_ds = MaskedTextDataset(val_data, tokenizer)

    BATCH_SIZE = 32
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator
    )

    writer.add_text("Batch size", str(BATCH_SIZE))
    writer.add_text("Train size", str(len(train_loader)))
    writer.add_text("Validation size", str(len(val_loader)))

    # Build the MLM model, optimizer, scheduler 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlm_model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    # resize embeddings to match your extended vocab
    mlm_model.resize_token_embeddings(len(tokenizer))
    writer.add_text("Vocab size", str(len(tokenizer)))

    mlm_model.to(device)

    optimizer = AdamW(mlm_model.parameters(), lr=5e-5, weight_decay=0.01)

    # 11 epochs is all you need for MLM pretraining (hyperparam study)
    EPOCHS = 40
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    writer.add_text("Epochs", str(EPOCHS))

    # Training loop 
    mlm_model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in train_loader:
            # batch already has input_ids, attention_mask, and labels=masked_inputs
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = mlm_model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} — Avg MLM train loss: {avg_train_loss:.4f}")
        writer.add_scalar("Avg MLM train loss", avg_train_loss, epoch)

        #  run one pass on val set
        mlm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = mlm_model(**batch).loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} — Avg MLM  val loss: {avg_val_loss:.4f}")
        writer.add_scalar("Avg MLM val loss", avg_val_loss, epoch)
        mlm_model.train()

        # Save MLM‐pretrained checkpoint 
        mlm_model.save_pretrained(f"./train/bert-mlm-checkpoint_1")
        tokenizer.save_pretrained("./train/bert-mlm-checkpoint")
        writer.flush()

import winsound

winsound.Beep(1000, 200)
