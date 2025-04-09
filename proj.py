from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
from rotary_embedding_torch import RotaryEmbedding
from datasets import load_dataset
import math

# https://huggingface.co/docs/transformers/en/tasks/language_modeling

# https://huggingface.co/docs/transformers/perf_train_gpu_one

# input_encoded = tokenizer(input_str, return_tensors='pt')
# print(type(input_encoded), input_encoded)

# input_utf8 = input_str.encode()
# np_input_utf8 = np.frombuffer(input_utf8, dtype=np.uint8)
# numpy_input = torch.randn(1, 8, 1024, 64)
# input_encoded = RotaryEmbedding(dim=32).rotate_queries_or_keys(numpy_input)
# print(type(input_encoded), input_encoded)

def get_and_process_dataset(tokenizer):
    eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
    eli5 = eli5.train_test_split(test_size=0.2).flatten()

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    def group_texts(examples, block_size=128):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return lm_dataset, data_collator

def build_trainer(model, tokenizer, lm_dataset, data_collator):
    training_args = TrainingArguments(
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        output_dir="my_awesome_eli5_clm-model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,
        optim="adamw_bnb_8bit",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        torch_empty_cache_steps=4,
        torch_compile=True,
        torch_compile_backend="inductor"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    return trainer

if __name__ == '__main__':
    huggingface_url = 'EleutherAI/gpt-neo-125m'
    model = AutoModelForCausalLM.from_pretrained(huggingface_url)
    tokenizer = AutoTokenizer.from_pretrained(huggingface_url)
    lm_dataset, data_collator = get_and_process_dataset(tokenizer)

    trainer = build_trainer(model, tokenizer, lm_dataset, data_collator)
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
