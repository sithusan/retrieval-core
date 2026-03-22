import argparse
from lib.multimodal_search import verify_image_embedding, image_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str)

    image_search_parser = subparsers.add_parser("image_search")
    image_search_parser.add_argument("image_path", type=str)

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            image_search(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
