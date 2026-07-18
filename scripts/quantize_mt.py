import os
import shutil
from pathlib import Path
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def main():
    dirs = [Path("models/mt/en-vi"), Path("models/mt/vi-en")]

    for save_dir in dirs:
        if not save_dir.exists():
            continue
        print(f"\n==================================================")
        print(f"Quantizing translation model components in {save_dir}...")
        print(f"==================================================")

        # 1. Define ARM64 optimized dynamic quantization config
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

        # 2. Quantize each model component
        components = ["encoder_model", "decoder_model_merged"]

        for component in components:
            onnx_path = save_dir / f"{component}.onnx"
            print(f"\nProcessing {component}.onnx...")

            quantizer = ORTQuantizer.from_pretrained(save_dir, file_name=f"{component}.onnx")

            # This will save the quantized model to the directory with the default suffix '_quantized'
            quantizer.quantize(
                save_dir=save_dir,
                quantization_config=qconfig,
                file_suffix="quantized"
            )

            # Verify that the quantized model was created
            quantized_path = save_dir / f"{component}_quantized.onnx"
            if not quantized_path.exists():
                raise FileNotFoundError(f"Quantization failed, {quantized_path} not found.")

            # Print original vs quantized sizes
            original_size = onnx_path.stat().st_size / (1024 * 1024)
            quantized_size = quantized_path.stat().st_size / (1024 * 1024)
            print(f"  Original:  {original_size:.1f} MB")
            print(f"  Quantized: {quantized_size:.1f} MB (Reduction: {(1.0 - quantized_size/original_size)*100:.1f}%)")

        # 3. Swap the files
        print("\nReplacing original models with quantized variants...")
        for component in components:
            onnx_path = save_dir / f"{component}.onnx"
            quantized_path = save_dir / f"{component}_quantized.onnx"

            # Backup the original model just in case
            backup_path = save_dir / f"{component}_original.onnx"
            shutil.move(onnx_path, backup_path)

            # Rename quantized to replace the original
            shutil.move(quantized_path, onnx_path)
            print(f"  Replaced {component}.onnx with quantized version.")

    print("\nQuantization successfully completed for all directories!")

if __name__ == "__main__":
    main()
