import os
import argparse
from lib.search_utils import get_path
import mimetypes
from dotenv import load_dotenv
from lib.prompts import get_image_query_rewrite_prompt
from google.genai import types
from google import genai


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument(
        "--query", type=str, help="A text query to be written based on the image"
    )

    args = parser.parse_args()

    image_path = get_path(args.image)
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as file:
        file_contents = file.read()

    response = result_from_llm_with_file(
        get_image_query_rewrite_prompt(), file_contents, mime, args.query
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


def result_from_llm_with_file(prompt: str, img: bytes, mime: str, query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    parts = [prompt, types.Part.from_bytes(data=img, mime_type=mime), query.strip()]

    return client.models.generate_content(model="gemini-2.5-flash", contents=parts)


if __name__ == "__main__":
    main()
