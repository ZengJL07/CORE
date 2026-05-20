import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def _to_json_safe(value: Any) -> Any:
    """Recursively convert Python objects into JSON-serializable values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    # Best-effort fallback for custom objects
    if hasattr(value, "__dict__"):
        return {"__class__": value.__class__.__name__, "attrs": _to_json_safe(vars(value))}

    return str(value)


def convert_cache_to_json(input_dir: Path, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    """Convert all .pkl files under input_dir into .json files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for pkl_path in sorted(input_dir.glob("*.pkl")):
        json_path = output_dir / f"{pkl_path.stem}.json"
        if json_path.exists() and not overwrite:
            skipped += 1
            continue

        with pkl_path.open("rb") as f:
            payload = pickle.load(f)

        json_payload = {
            "source_file": str(pkl_path),
            "data": _to_json_safe(payload),
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

        converted += 1
        print(f"[AIME] Converted: {pkl_path} -> {json_path}")

    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GEPA fitness cache .pkl files to JSON.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/aime_math/fitness_cache"),
        help="Directory containing .pkl cache files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aime_math/fitness_cache_json"),
        help="Directory to write converted .json files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files if they already exist.",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    converted, skipped = convert_cache_to_json(args.input_dir, args.output_dir, args.overwrite)
    print(
        f"[AIME] Conversion finished. Converted={converted}, Skipped={skipped}, "
        f"OutputDir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
