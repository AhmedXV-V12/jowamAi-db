# jowamAi-db

# 1. What's jowamAi-db
It's Database for model _jowamAi_ to train the model on this data 

# 2. What's the License
The license is specific to (```AhmedXV-V12```).
You may use the data under the following clear conditions:
1. You must state that this data is from ```AhmedXV-V12```
2. It may be used for commercial purposes.
3. We are not responsible if the data you have taken is modified. You bear full responsibility for this.

# 3. how to use this ```database```
you have 2 methods 
*1-*
use the [_raw url_](https://raw.githubusercontent.com/AhmedXV-V12/jowamAi-db/main/database.json) in your code :
```python
url = "https://raw.githubusercontent.com/AhmedXV-V12/jowamAi-db/main/database.json"
```
*2-*
1. install : [database](./database.json)
2. you can make code to train your model for _**Example**_ :
```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfiguration,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Configuration and Constants ---
# model_id: The base pre-trained LLM from Hugging Face Hub.
# dataset_path: The local path to your JSON data.
model_id = "meta-llama/Llama-2-7b-hf"
dataset_path = "./database.json"

# 1. Load Dataset
# Loads JSON data into a Hugging Face Dataset object. 
# Ensure your JSON has a consistent key (e.g., "text") for each entry.
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 2. BitsAndBytes Configuration (4-bit Quantization)
# This reduces the VRAM usage significantly by loading the model in 4-bit precision.
# NF4 (NormalFloat 4) is used for better weight distribution in quantized models.
bnb_config = BitsAndBytesConfiguration(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 3. Load Pre-trained Model
# device_map="auto" automatically distributes the model layers across available GPUs.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 4. Prepare Model for K-Bit Training
# Handles gradient checkpointing and freezing non-trainable weights to save memory.
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# 5. Tokenizer Initialization
# Padding token is set to EOS (End Of Sentence) to handle batch sequences.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 6. LoRA (Low-Rank Adaptation) Configuration
# r: The rank of the update matrices. Higher means more parameters but more capacity.
# target_modules: Specific layers in the Transformer (Attention heads) to apply LoRA.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA adapters to the base model.
model = get_peft_model(model, lora_config)

# 7. Training Arguments
# gradient_accumulation_steps: Simulates a larger batch size without increasing VRAM.
# optim="paged_adamw_8bit": Optimized 8-bit optimizer for memory efficiency.
training_args = TrainingArguments(
    output_dir="./lora_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit",
    report_to="none" # Set to "wandb" or "tensorboard" for tracking
)

# 8. SFTTrainer (Supervised Fine-Tuning)
# dataset_text_field: Must match the key in your database.json file.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# 9. Execute Training Process
trainer.train()

# 10. Save LoRA Adapters
# This saves only the small adapter weights, not the full 7B model.
model.save_pretrained("./final_lora_model")
tokenizer.save_pretrained("./final_lora_model")
```
to run this example run in Terminal : 
```bash
pip install torch transformers peft datasets bitsandbytes accelerate trl
```
if the *OS* is **Debian/Ubuntu**:
```bash
pip install torch transformers peft datasets bitsandbytes accelerate trl --break-system-packages
```
Note: This example requires a GPU with at least 12GB VRAM (e.g., RTX 3060 12GB or higher)

```by: AhmedXV-V12```
