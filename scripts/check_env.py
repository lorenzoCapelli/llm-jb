"""Quick environment check: torch, CUDA, visible GPUs."""

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Visible GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            print(f"  [{i}] {props.name} ({total_gb:.1f} GB)")
    else:
        print("No GPU visible, running on CPU.")


if __name__ == "__main__":
    main()
