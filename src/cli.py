"""Command-line interface for AI fine-tuning project."""

import os
import sys
import subprocess
import click
from typing import Optional


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AI Fine-tuning Project CLI."""
    pass


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for trained model")
@click.option("--epochs", "-e", type=int, default=3, help="Number of training epochs")
@click.option("--lr", "--learning-rate", type=float, default=5e-5, help="Learning rate")
def train(
    config: Optional[str],
    output_dir: Optional[str],
    epochs: int,
    lr: float,
):
    """Train an AI model."""
    click.echo(f"Training model with {epochs} epochs and learning rate {lr}")
    if config:
        click.echo(f"Using config file: {config}")
    if output_dir:
        click.echo(f"Output directory: {output_dir}")
    
    # TODO: Implement actual training logic
    click.echo("Training not implemented yet. Add your training logic here.")


@cli.command()
@click.option(
    "--model", "-m",
    type=str,
    default="mistralai/Mistral-7B-Instruct-v0.3",
    help="HuggingFace model name or path",
    show_default=True
)
@click.option(
    "--data", "-d",
    type=str,
    default="./training_dataset/*.jsonl",
    help="Data path (local glob pattern or HuggingFace dataset ID like 'tatsu-lab/alpaca')",
    show_default=True
)
@click.option(
    "--data-split",
    type=str,
    default="train",
    help="Dataset split to use when loading from HuggingFace (e.g., 'train', 'test')",
    show_default=True
)
@click.option(
    "--data-config",
    type=str,
    help="Dataset configuration/subset name for HuggingFace datasets (e.g., 'default', 'en')"
)
@click.option(
    "--output", "-o",
    type=str,
    default="./qlora_mistral7b_finetuned",
    help="Output directory for trained model",
    show_default=True
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="./configs/qlora.yml",
    help="Configuration file path (for other hyperparameters)",
    show_default=True
)
@click.option(
    "--max-length",
    type=int,
    help="Maximum sequence length (overrides config)"
)
@click.option(
    "--batch-size",
    type=int,
    help="Training batch size (overrides config)"
)
@click.option(
    "--epochs",
    type=int,
    help="Number of training epochs (overrides config)"
)
@click.option(
    "--learning-rate",
    type=float,
    help="Learning rate (overrides config)"
)
def qlora(
    model: str,
    data: str,
    data_split: str,
    data_config: Optional[str],
    output: str,
    config: str,
    max_length: Optional[int],
    batch_size: Optional[int],
    epochs: Optional[int],
    learning_rate: Optional[float],
):
    """Run QLoRA fine-tuning with custom model, data, and output paths.
    
    Supports both local files and HuggingFace datasets:
    - Local: ./data/*.jsonl
    - HuggingFace: tatsu-lab/alpaca, timdettmers/openassistant-guanaco, etc.
    
    Examples:
        # Local files
        ai-train qlora -d ./data/*.jsonl -o ./my_model
        
        # HuggingFace dataset
        ai-train qlora -d tatsu-lab/alpaca -o ./alpaca_model
        
        # HuggingFace with config
        ai-train qlora -d HuggingFaceH4/ultrachat_200k --data-config sft --data-split train_sft
    """
    click.echo("="*80)
    click.echo("QLoRA Fine-tuning")
    click.echo("="*80)
    click.echo(f"Model: {model}")
    click.echo(f"Data: {data}")
    if not data.startswith("./") and "*" not in data and "/" in data:
        click.echo(f"  Type: HuggingFace Dataset")
        click.echo(f"  Split: {data_split}")
        if data_config:
            click.echo(f"  Config: {data_config}")
    else:
        click.echo(f"  Type: Local Files")
    click.echo(f"Output: {output}")
    click.echo(f"Config: {config}")
    click.echo("="*80)
    
    # Prepare environment variables for the qlora script
    env = os.environ.copy()
    env["QLORA_CONFIG"] = config
    env["QLORA_MODEL"] = model
    env["QLORA_DATA"] = data
    env["QLORA_DATA_SPLIT"] = data_split
    if data_config:
        env["QLORA_DATA_CONFIG"] = data_config
    env["QLORA_OUTPUT"] = output
    
    # Add optional overrides
    if max_length is not None:
        env["QLORA_MAX_LENGTH"] = str(max_length)
    if batch_size is not None:
        env["QLORA_BATCH_SIZE"] = str(batch_size)
    if epochs is not None:
        env["QLORA_EPOCHS"] = str(epochs)
    if learning_rate is not None:
        env["QLORA_LEARNING_RATE"] = str(learning_rate)
    
    # Run the qlora script
    try:
        script_path = os.path.join(os.path.dirname(__file__), "fintuning", "qlora.py")
        subprocess.run([sys.executable, script_path], env=env, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Training failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(f"Error: Could not find qlora.py script", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--adapter-path", "-a",
    type=str,
    help="Path to the saved adapter checkpoint"
)
@click.option(
    "--model-name", "-m",
    type=str,
    help="Base model name or path (defaults to model from adapter config)"
)
@click.option(
    "--prompt", "-p",
    type=str,
    help="Single prompt to generate from (if not provided, enters interactive mode)"
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=512,
    help="Maximum number of tokens to generate",
    show_default=True
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for sampling",
    show_default=True
)
@click.option(
    "--top-p",
    type=float,
    default=0.9,
    help="Top-p (nucleus) sampling parameter",
    show_default=True
)
@click.option(
    "--top-k",
    type=int,
    default=50,
    help="Top-k sampling parameter",
    show_default=True
)
@click.option(
    "--repetition-penalty",
    type=float,
    default=1.1,
    help="Repetition penalty",
    show_default=True
)
@click.option(
    "--no-quantization",
    is_flag=True,
    help="Load model without 4-bit quantization (auto-disabled on CPU)"
)
def infer(
    adapter_path: Optional[str],
    model_name: Optional[str],
    prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_quantization: bool,
):
    """Run inference with a QLoRA adapted model.
    
    Supports both interactive and single-prompt modes.
    
    Examples:
        # Interactive mode with default adapter
        ai-infer
        
        # Interactive mode with custom adapter
        ai-infer -a ./my_adapter
        
        # Single prompt
        ai-infer -p "What is quantum computing?"
        
        # With custom generation parameters
        ai-infer -p "Explain AI" --temperature 0.9 --max-new-tokens 1024
    """
    click.echo("="*80)
    click.echo("QLoRA Inference")
    click.echo("="*80)
    
    # Prepare environment variables
    env = os.environ.copy()
    if adapter_path:
        env["ADAPTER_PATH"] = adapter_path
        click.echo(f"Adapter: {adapter_path}")
    else:
        click.echo(f"Adapter: (from config or default)")
    
    if model_name:
        env["MODEL_NAME"] = model_name
        click.echo(f"Model: {model_name}")
    
    if prompt:
        click.echo(f"Mode: Single prompt")
    else:
        click.echo(f"Mode: Interactive")
    
    click.echo("="*80 + "\n")
    
    # Build command arguments
    args = [sys.executable]
    script_path = os.path.join(os.path.dirname(__file__), "run", "run-with-adapter.py")
    args.append(script_path)
    
    if adapter_path:
        args.extend(["--adapter-path", adapter_path])
    if model_name:
        args.extend(["--model-name", model_name])
    if prompt:
        args.extend(["--prompt", prompt])
    args.extend(["--max-new-tokens", str(max_new_tokens)])
    args.extend(["--temperature", str(temperature)])
    args.extend(["--top-p", str(top_p)])
    args.extend(["--top-k", str(top_k)])
    args.extend(["--repetition-penalty", str(repetition_penalty)])
    if no_quantization:
        args.append("--no-quantization")
    
    # Run the inference script
    try:
        subprocess.run(args, env=env, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Inference failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(f"Error: Could not find run-with-adapter.py script", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model-path", "-m", type=click.Path(exists=True), required=True, help="Path to trained model")
@click.option("--data-path", "-d", type=click.Path(exists=True), required=True, help="Path to evaluation data")
@click.option("--output", "-o", type=click.Path(), help="Output file for evaluation results")
def eval(
    model_path: str,
    data_path: str,
    output: Optional[str],
):
    """Evaluate a trained model."""
    click.echo(f"Evaluating model at: {model_path}")
    click.echo(f"Using data from: {data_path}")
    if output:
        click.echo(f"Results will be saved to: {output}")
    
    # TODO: Implement actual evaluation logic
    click.echo("Evaluation not implemented yet. Add your evaluation logic here.")


def train_command():
    """Entry point for ai-train command."""
    # Create a standalone group for training
    @click.group()
    @click.version_option(version="0.1.0")
    def train_cli():
        """AI Model Training CLI."""
        pass
    
    train_cli.add_command(train, "train")
    train_cli.add_command(qlora, "qlora")
    train_cli()


def eval_command():
    """Entry point for ai-eval command."""
    # Invoke eval directly
    eval.main(standalone_mode=True)


def infer_command():
    """Entry point for ai-infer command."""
    # Invoke infer directly
    infer.main(standalone_mode=True)


if __name__ == "__main__":
    cli()
