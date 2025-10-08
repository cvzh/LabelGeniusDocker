import argparse, json, os, sys
from pathlib import Path

# import your functions from the file you provided
from labelgenius.LabelGenius import (
    classification_CLIP_0_shot,
    classification_CLIP_finetuned,
    finetune_CLIP,
    classification_GPT,
    generate_GPT_finetune_jsonl,
    finetune_GPT,
    auto_verification,
)

def main():
    p = argparse.ArgumentParser(prog="labelgenius", description="LabelGenius CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- zero-shot CLIP ------------------------------------------------------
    zs = sub.add_parser("clip-zero", help="Zero-shot CLIP classification")
    zs.add_argument("--text_path", type=Path, required=True)
    zs.add_argument("--img_dir", type=Path)
    zs.add_argument("--mode", choices=["text","image","both"], required=True)
    zs.add_argument("--labels", type=str, nargs="*", help="Label names (defaults inside code)")
    zs.add_argument("--text_column", nargs="*", default=["headline"])
    zs.add_argument("--predict_column", default="label")
    zs.add_argument("--text_backend", choices=["clip","llama"], default="clip")
    zs.add_argument("--llama_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    zs.add_argument("--llama_max_new_tokens", type=int, default=8)
    zs.add_argument("--save_csv", type=Path, help="If set, save results to this CSV")

    # ---- fine-tuned CLIP prediction -----------------------------------------
    ftp = sub.add_parser("clip-predict", help="Predict with fine-tuned CLIP checkpoint")
    ftp.add_argument("--mode", choices=["text","image","both"], required=True)
    ftp.add_argument("--text_path", type=Path, required=True)
    ftp.add_argument("--text_column", nargs="*", default=["headline"])
    ftp.add_argument("--img_dir", type=Path)
    ftp.add_argument("--prompt", type=str)
    ftp.add_argument("--model_name", type=Path, default=Path("best_clip_model.pth"))
    ftp.add_argument("--batch_size", type=int, default=8)
    ftp.add_argument("--num_classes", type=int, default=24)
    ftp.add_argument("--predict_column", default="label")
    ftp.add_argument("--true_label", default=None)
    ftp.add_argument("--save_csv", type=Path, help="If set, save results to this CSV")

    # ---- fine-tune CLIP ------------------------------------------------------
    ftn = sub.add_parser("clip-finetune", help="Fine-tune CLIP")
    ftn.add_argument("--mode", choices=["text","image","both"], default="both")
    ftn.add_argument("--text_path", type=Path, default=Path("Demo_data/D1_1.csv"))
    ftn.add_argument("--text_column", nargs="*", default=["headline"])
    ftn.add_argument("--img_dir", type=Path, default=Path("Demo_data/D1_imgs_1"))
    ftn.add_argument("--true_label", default="section_numeric")
    ftn.add_argument("--prompt", type=str)
    ftn.add_argument("--model_name", type=Path, default=Path("best_clip_model.pth"))
    ftn.add_argument("--num_epochs", type=int, default=20)
    ftn.add_argument("--batch_size", type=int, default=8)
    ftn.add_argument("--learning_rate", type=float, default=1e-5)

    # ---- GPT classification ---------------------------------------------------
    gpt = sub.add_parser("gpt-classify", help="GPT-based labeling")
    gpt.add_argument("--text_path", type=Path, required=True)
    gpt.add_argument("--category", nargs="*", required=True, help="Valid labels (as strings)")
    gpt.add_argument("--image_dir", type=Path)
    gpt.add_argument("--prompt", type=str, nargs="*")
    gpt.add_argument("--columns", nargs="*", default=["headline"])
    gpt.add_argument("--model", default="gpt-4o-mini")
    gpt.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    gpt.add_argument("--reasoning_effort", choices=["minimal","low","medium","high"], default="minimal")
    gpt.add_argument("--mode", choices=["text","image","both"], default="both")
    gpt.add_argument("--output_column_name", default="label")
    gpt.add_argument("--num_themes", type=int, default=1)
    gpt.add_argument("--num_votes", type=int, default=1)
    gpt.add_argument("--batch_size", type=int, default=5)
    gpt.add_argument("--wait_time", type=float, default=1.2)
    gpt.add_argument("--save_csv", type=Path, help="If set, save results to this CSV")

    # ---- export jsonl for GPT fine-tuning ------------------------------------
    ej = sub.add_parser("gpt-make-jsonl", help="Create JSONL for GPT fine-tuning")
    ej.add_argument("--input_csv", type=Path, required=True)
    ej.add_argument("--output_jsonl", type=Path, default=Path("classification_result.jsonl"))
    ej.add_argument("--label_col", nargs="*", default=["true_class"])
    ej.add_argument("--system_prompt", nargs="*", default=None)
    ej.add_argument("--input_col", nargs="*", default=["text_content"])

    # ---- fine-tune GPT --------------------------------------------------------
    fgj = sub.add_parser("gpt-finetune", help="Launch GPT fine-tuning job")
    fgj.add_argument("--training_file_path", type=Path, required=True)
    fgj.add_argument("--model", default="gpt-4o-mini")
    fgj.add_argument("--method_type", default="supervised")
    fgj.add_argument("--hyperparameters", type=str, default="{}",
                     help='JSON dict, e.g. {"n_epochs":2}')
    fgj.add_argument("--poll_interval", type=int, default=15)
    fgj.add_argument("--max_wait_time", type=int, default=3600)
    fgj.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))

    # ---- verification ---------------------------------------------------------
    ver = sub.add_parser("verify", help="Compute accuracy/F1, etc.")
    ver.add_argument("--csv", type=Path, required=True)
    ver.add_argument("--predicted_cols", nargs="+", required=True)
    ver.add_argument("--true_cols", nargs="+", required=True)
    ver.add_argument("--category", nargs="*")
    ver.add_argument("--sample_size", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "clip-zero":
        df = classification_CLIP_0_shot(
            text_path=str(args.text_path),
            img_dir=str(args.img_dir) if args.img_dir else None,
            mode=args.mode,
            prompt=args.labels,
            text_column=args.text_column,
            predict_column=args.predict_column,
            text_backend=args.text_backend,
            llama_model=args.llama_model,
            llama_max_new_tokens=args.llama_max_new_tokens,
        )
        _maybe_save(df, args.save_csv)

    elif args.cmd == "clip-predict":
        df = classification_CLIP_finetuned(
            mode=args.mode,
            text_path=str(args.text_path),
            text_column=args.text_column,
            img_dir=str(args.img_dir) if args.img_dir else None,
            prompt=args.prompt,
            model_name=str(args.model_name),
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            predict_column=args.predict_column,
            true_label=args.true_label,
        )
        _maybe_save(df, args.save_csv)

    elif args.cmd == "clip-finetune":
        best = finetune_CLIP(
            mode=args.mode,
            text_path=str(args.text_path),
            text_column=args.text_column,
            img_dir=str(args.img_dir) if args.img_dir else None,
            true_label=args.true_label,
            prompt=args.prompt,
            model_name=str(args.model_name),
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(json.dumps({"best_val_acc": best}))

    elif args.cmd == "gpt-classify":
        df = classification_GPT(
            text_path=str(args.text_path),
            category=args.category,
            image_dir=str(args.image_dir) if args.image_dir else None,
            prompt=args.prompt,
            column_4_labeling=args.columns,
            model=args.model,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
            mode=args.mode,
            output_column_name=args.output_column_name,
            num_themes=args.num_themes,
            num_votes=args.num_votes,
            batch_size=args.batch_size,
            wait_time=args.wait_time,
        )
        _maybe_save(df, args.save_csv)

    elif args.cmd == "gpt-make-jsonl":
        import pandas as pd
        df = pd.read_csv(args.input_csv) if args.input_csv.suffix==".csv" else pd.read_json(args.input_csv, lines=True)
        generate_GPT_finetune_jsonl(
            df=df,
            output_path=str(args.output_jsonl),
            label_col=args.label_col if len(args.label_col)>1 else args.label_col[0],
            system_prompt=args.system_prompt,
            input_col=args.input_col if len(args.input_col)>1 else args.input_col[0],
        )
        print(f"Wrote {args.output_jsonl}")

    elif args.cmd == "gpt-finetune":
        import json as _json
        hypers = _json.loads(args.hyperparameters)
        model = finetune_GPT(
            training_file_path=str(args.training_file_path),
            model=args.model,
            method_type=args.method_type,
            hyperparameters=hypers,
            poll_interval=args.poll_interval,
            max_wait_time=args.max_wait_time,
            api_key=args.api_key,
        )
        print(json.dumps({"fine_tuned_model": model}))

    elif args.cmd == "verify":
        import pandas as pd
        df = pd.read_csv(args.csv) if args.csv.suffix==".csv" else pd.read_json(args.csv, lines=True)
        auto_verification(
            df=df,
            predicted_cols=args.predicted_cols,
            true_cols=args.true_cols,
            category=args.category,
            sample_size=args.sample_size,
        )

def _maybe_save(df, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith(".jsonl"):
            with open(path, "w", encoding="utf-8") as f:
                for rec in df.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            df.to_csv(path, index=False)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
