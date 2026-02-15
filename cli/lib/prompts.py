def get_spell_correcter_prompt(query: str) -> str:
    return f"""Fix any spelling errors in this movie search query.
        Only correct obvious typos. Don't change correctly spelled words.
        Query: "{query}"
        If no errors, return the original query.
        Corrected:"""


def get_query_rewriter_prompt(query: str) -> str:
    return f"""Rewrite this movie search query to be more specific and searchable.
        Original: "{query}"
        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic
        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:"""
