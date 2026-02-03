#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
    chunk,
    semantic_chunk,
)
from lib.chunked_semantic_search import embed_chunks, search_chunked
from lib.constants import SEARCH_LIMIT, CHUNK_SIZE, OVERLAP, MAX_CHUNK_SIZE


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model to use the semantic search")
    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings to use the semantic search"
    )

    subparsers.add_parser("embed_chunks", help="Embed the chunks")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    embed_query_text_parser = subparsers.add_parser(
        "embedquery", help="Embed the query"
    )
    embed_query_text_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search Query")
    search_parser.add_argument("query", type=str, help="Query for search")
    search_parser.add_argument(
        "--limit", type=int, default=SEARCH_LIMIT, help="Search query Limit"
    )

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search Query")
    search_chunked_parser.add_argument("query", type=str, help="Query for search")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=SEARCH_LIMIT, help="Search query Limit"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk the text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE, help="Size of the chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=OVERLAP, help="Overlap of the chunk"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the text"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to semantic chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=MAX_CHUNK_SIZE,
        help="Max size of the semantic chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=OVERLAP, help="Overlap of the semantic chunk"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
