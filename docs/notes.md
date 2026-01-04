# RAG - Retrieval-Augmented Generation

## Keyword Search

The more technical and precise the terminology in the domain is, the more powerful keyword search is.

### When to Use Keyword Search vs Semantic Search

#### Medical Search - Precision Desired

- **Query:** "Covid 19"
- **Keyword Search:** Finds documents specifically about the Covid-19.
- **Semantic-only Search:** Finds less-related content about "respiratory viruses" or "pandemic diseases"

#### Movie Search - Generalization Desired

- **Query:** "Alien Movies"
- **Keyword Search:** Only finds "Alien"
- **Semantic-only Search:** Finds similar movies like "Invaders from Mars", "E.T", or "Alien"

---

## Text Processing Pipeline

For keyword search, it doesn't need to be exact. To improve search we need to do some text processing.

### Processing Steps

- Case Sensitivity
- Punctuation
- Tokenization
- Stop words
- Stemming

### Tokenization

It means splitting text into smaller pieces, called tokens. It's because unlike literate humans, computers do not understand the structure of natural language documents and cannot automatically recognize words and sentences. To a computer, a document is only a sequence of bytes. For example, a computer does not know that "a space" character separates the words. Instead, humans must program the computer to identify what constitutes individual or distinct words referred to as tokens. Such process is called tokenization.

### Stop Words

Stop words are words that do not have semantic meaning. We simply want to remove the stop words because they can cause misleading results.

**Example:**

User Query: `"the bear"`

- "Jungle Bear": matches "bear" ✓ (relevant match)
- "The Terminator": matches "the" ✗ (meaningless "the" matches, so misleading result)

### Stemming

    It is for help matching differentation of the keywords by reducing by it's root form.

**Example:**
User query: `"running"`

- A River Runs Through It: Not returned – no exact match for "running"
- Run Baby Run: Not returned – no exact match for "running"
