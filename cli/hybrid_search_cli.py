import argparse
from lib.hybrid_search import normalize, weighted_search, rrf_search
from lib.constants import SEARCH_LIMIT, ALPHA, RRF_WEIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("scores", type=float, nargs="+")

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str, help="Search Query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Alpha to dynamically control the weighting between the scores",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=SEARCH_LIMIT, help="Search query Limit"
    )

    rrf_search_parser = subparsers.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str, help="Search Query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=RRF_WEIGHT,
        help="K to dynamically control the weighting between the scores",
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=SEARCH_LIMIT, help="Search query Limit"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
