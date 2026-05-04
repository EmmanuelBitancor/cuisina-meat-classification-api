from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

# Ensure custom objects from KerasHub (e.g., MobileNetV5Backbone) can be resolved.
try:  # pragma: no cover
    import keras_hub  # noqa: F401
except Exception:
    keras_hub = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .keras model to a .tflite model."
    )
    parser.add_argument(
        "--keras-model",
        type=Path,
        default=Path("meat_classification_trained_mobilenetv5_model.keras"),
        help="Path to the .keras model file.",
    )
    parser.add_argument(
        "--tflite-out",
        type=Path,
        default=Path("meat_model.tflite"),
        help="Output path for the .tflite file.",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Enable float16 weight quantization (smaller, usually similar accuracy).",
    )
    parser.add_argument(
        "--disable-select-tf-ops",
        action="store_true",
        help="Use only TFLite built-in ops (may fail for some complex models).",
    )
    return parser.parse_args()


def _apply_common_converter_options(
    converter: tf.lite.TFLiteConverter,
    *,
    use_float16: bool,
    allow_select_tf_ops: bool,
) -> None:
    # Enable default optimizations (quantization) to reduce model size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if allow_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

    if use_float16:
        converter.target_spec.supported_types = [tf.float16]


def _convert_with_from_keras_model(
    model: tf.keras.Model,
    *,
    use_float16: bool,
    allow_select_tf_ops: bool,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    _apply_common_converter_options(
        converter,
        use_float16=use_float16,
        allow_select_tf_ops=allow_select_tf_ops,
    )
    return converter.convert()


def _convert_with_concrete_function(
    model: tf.keras.Model,
    *,
    use_float16: bool,
    allow_select_tf_ops: bool,
) -> bytes:
    input_shape = list(model.inputs[0].shape)
    input_dtype = model.inputs[0].dtype

    # Force a concrete batch size for tracing.
    if input_shape and input_shape[0] is None:
        input_shape[0] = 1

    input_signature = [tf.TensorSpec(shape=input_shape, dtype=input_dtype, name="input")]

    @tf.function(input_signature=input_signature)
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
    _apply_common_converter_options(
        converter,
        use_float16=use_float16,
        allow_select_tf_ops=allow_select_tf_ops,
    )
    return converter.convert()


def main() -> None:
    args = parse_args()
    keras_path = args.keras_model.resolve()
    tflite_path = args.tflite_out.resolve()

    if not keras_path.exists():
        raise SystemExit(f"Keras model not found: {keras_path}")

    # NOTE: This model depends on keras_hub (MobileNetV5Backbone). Make sure it's installed
    # in the conversion environment. The produced .tflite does NOT require keras_hub.
    model = tf.keras.models.load_model(keras_path, compile=False)

    allow_select_tf_ops = not args.disable_select_tf_ops

    errors: list[str] = []

    try:
        tflite_model = _convert_with_from_keras_model(
            model,
            use_float16=args.float16,
            allow_select_tf_ops=allow_select_tf_ops,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"from_keras_model failed: {type(exc).__name__}: {exc}")
        tflite_model = _convert_with_concrete_function(
            model,
            use_float16=args.float16,
            allow_select_tf_ops=allow_select_tf_ops,
        )

    tflite_path.write_bytes(tflite_model)

    if errors:
        print("Fallback conversion used.")
        for err in errors:
            print(err)

    print("Wrote:", tflite_path)


if __name__ == "__main__":
    main()
