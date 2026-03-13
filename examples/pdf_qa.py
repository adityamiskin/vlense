import argparse
import asyncio
from pathlib import Path

from vlense import Vlense


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index PDFs/images with ColFlor and ask grounded questions.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more PDF/image paths, or a directory containing supported files.",
    )
    parser.add_argument(
        "--question",
        help="Single question to ask after indexing. If omitted, an interactive prompt starts.",
    )
    parser.add_argument(
        "--collection",
        default="demo-docs",
        help="Collection name stored under the index directory.",
    )
    parser.add_argument(
        "--index-dir",
        default=".vlense",
        help="Directory where the ColFlor index is stored.",
    )
    parser.add_argument(
        "--retriever-model",
        default="ahmed-masry/ColFlor",
        help="ColFlor model name for retrieval.",
    )
    parser.add_argument(
        "--vision-model",
        default="openai/gpt-5-mini",
        help="Vision model used to answer from retrieved page images.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved pages to send to the vision model.",
    )
    parser.add_argument(
        "--temp-dir",
        help="Optional temporary working directory for indexing.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse an existing collection instead of rebuilding the index.",
    )
    return parser


async def run_once(args: argparse.Namespace) -> None:
    vlense = Vlense()

    if not args.skip_index:
        manifest_path = await vlense.index(
            data_dir=args.inputs,
            collection_name=args.collection,
            index_dir=args.index_dir,
            retriever_model=args.retriever_model,
            temp_dir=args.temp_dir,
        )
        print(f"Indexed collection: {manifest_path}")
    else:
        print(
            "Skipping indexing and reusing collection "
            f"'{args.collection}' from {Path(args.index_dir).resolve()}"
        )

    if args.question:
        answer = await vlense.ask(
            query=args.question,
            collection_name=args.collection,
            index_dir=args.index_dir,
            model=args.vision_model,
            top_k=args.top_k,
        )
        print("\nAnswer:\n")
        print(answer)
        return

    print("\nInteractive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = await vlense.ask(
            query=question,
            collection_name=args.collection,
            index_dir=args.index_dir,
            model=args.vision_model,
            top_k=args.top_k,
        )
        print("\nAnswer:\n")
        print(answer)
        print("")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_once(args))


if __name__ == "__main__":
    main()
