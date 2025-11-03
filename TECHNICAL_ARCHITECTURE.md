# Technical Architecture: ViViAnGPT

> Deep dive into the technical implementation, design patterns, and engineering decisions behind ViViAnGPT.

## Table of Contents
- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Indexing Pipeline](#indexing-pipeline)
- [RAG Pipeline](#rag-pipeline)
- [Advanced Features](#advanced-features)
- [Configuration System](#configuration-system)
- [Deployment](#deployment)

---

## System Overview

### Design Philosophy

ViViAnGPT follows three core principles:

1. **Modularity**: Every component is independently configurable and replaceable
2. **Offline-First**: Complete autonomy without external dependencies
3. **Resource-Aware**: Optimized for constrained hardware environments

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Ollama (Llama 3.2 3B) | Local inference engine |
| **Embeddings** | mxbai-embed-large | Vector representations |
| **Vector DB** | FAISS | Similarity search |
| **Framework** | LangChain | RAG orchestration |
| **UI** | Streamlit | Web interface |
| **Translation** | MarianMT | Multilingual support |
| **Reranking** | FlashRank | Document re-scoring |

### System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         Frontend Layer                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │    Streamlit UI (ui/main.py)                             │ │
│  │    - Configuration panel                                 │ │
│  │    - Chat interface with streaming                       │ │
│  │    - Source document display                             │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬───────────────────────────────┘
                                │
┌───────────────────────────────┴───────────────────────────────┐
│                      Application Layer                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  RAGPipeline (viviangpt/pipeline/rag_pipeline.py)        │ │
│  │  ┌────────────┬──────────────┬──────────────┬─────────┐  │ │
│  │  │   Query    │  Retrieval   │  Document    │ Answer  │  │ │
│  │  │ Processors │ Strategies   │ Processors   │Generator│  │ │
│  │  └────────────┴──────────────┴──────────────┴─────────┘  │ │
│  │                                                            │ │
│  │  Memory Manager • Translation Layer • Logging             │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬───────────────────────────────┘
                                │
┌───────────────────────────────┴───────────────────────────────┐
│                       Service Layer                            │
│  ┌──────────────┬──────────────┬────────────────────────────┐ │
│  │ LLM Service  │   Embedding  │    Retrieval Service       │ │
│  │  (Ollama)    │    Service   │   (FAISS + BM25)           │ │
│  └──────────────┴──────────────┴────────────────────────────┘ │
└───────────────────────────────┬───────────────────────────────┘
                                │
┌───────────────────────────────┴───────────────────────────────┐
│                        Data Layer                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Vector Store (FAISS) • BM25 Index • Conversation Logs   │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Builder Pattern Implementation

The system uses the Builder Pattern for flexible pipeline configuration:

```python
pipeline = (
    RAGPipelineBuilder()
    .with_llm(model_name="llama3.2:3b")
    .with_memory(window_size=10)
    .with_hybrid_retrieval(vector_weight=0.7, bm25_weight=0.3)
    .with_reranking(k=7)
    .with_compression()
    .build()
)
```

**Benefits**:
- Fluent, readable configuration
- Easy A/B testing of components
- Runtime component selection
- Type-safe configuration

### Component Hierarchy

```
Component (Abstract Base)
├── QueryProcessor
│   ├── QueryTransformer          # Improves query formulation
│   ├── QueryDecomposer           # Breaks complex queries
│   ├── SelfContainedQueryGenerator # Removes context dependency
│   └── QuestionRephraser         # Simplifies and normalizes
│
├── RetrievalStrategy
│   ├── VectorRetriever           # Semantic search (FAISS)
│   ├── BM25RetrievalStrategy     # Keyword search
│   ├── HybridRetriever           # Combines both
│   └── MultiQueryRetriever       # Multi-query orchestration
│
├── DocumentProcessor
│   ├── DocumentReranker          # FlashRank re-scoring
│   ├── KeywordFilterProcessor    # N-gram filtering
│   ├── HeaderBoostProcessor      # Title matching
│   └── DocumentCompressor        # Prompt compression
│
└── AnswerGenerator
    ├── RAGAnswerGenerator        # Standard RAG generation
    ├── HistoryAnswerGenerator    # Memory-based responses
    └── StreamingAnswerWrapper    # Adds streaming capability
```

### Component Factory

The `ComponentFactory` centralizes component creation:

```python
class ComponentFactory:
    @staticmethod
    def create_retrieval_strategy(config):
        if config.type == "hybrid":
            return HybridRetriever(
                vector_weight=config.vector_weight,
                bm25_weight=config.bm25_weight
            )
        # ... other strategies
```

**Advantages**:
- Single source of truth for configuration
- Consistent component initialization
- Easy mocking for tests
- Dependency injection

---

## Indexing Pipeline

### Pipeline Overview

```
PDF Documents
    ↓
marker-pdf Conversion (Optional: can skip if .md files exist)
    ↓
Markdown Preprocessing
    ↓
Header-Aware Chunking
    ↓
Keyword Extraction
    ↓
Vector Embedding
    ↓
FAISS Index Creation
```

### Stage 1: PDF to Markdown

**Tool**: marker-pdf library

**Process**:
1. Document-specific margin cropping
2. Layout and hierarchy preservation
3. Element recognition (headers, tables, lists)

**Configuration** (indexer/config.py):
```python
MARGIN_CONFIGS = {
    "document.pdf": {
        "crop_margins": {"left": 50, "right": 50, "top": 80, "bottom": 60},
        "skip_pages": [0, 1]  # Cover and index
    }
}
```

**Challenges Addressed**:
- Inconsistent header levels → Normalized in preprocessing
- Images misidentified as tables → Filtered based on size
- Header/footer noise → Auto-cropped with margins

### Stage 2: Markdown Preprocessing

**Operations**:
```python
def preprocess(markdown_text):
    text = remove_page_spans(text)      # Remove page references
    text = normalize_headers(text)      # Standardize H1-H6
    text = clean_html_tags(text)        # Remove residual HTML
    text = standardize_formatting(text) # Consistent spacing
    return text
```

**Header Normalization**:
- Detects header patterns from PDF conversion
- Corrects misidentified levels (h2 → h4)
- Maintains document hierarchy

### Stage 3: Intelligent Chunking

**Strategy**: Header-aware semantic splitting

**Parameters**:
```python
TARGET_CHUNK_SIZE = 800      # Target character count
MAX_CHUNK_SIZE = 1500        # Hard limit
MIN_CHUNK_SIZE = 300         # Minimum threshold
CHUNK_OVERLAP = 100          # Overlap for context
MERGE_THRESHOLD = 0.6        # Dynamic merging
```

**Algorithm**:
1. Split at header boundaries (h1 → h2 → h3)
2. Keep related content together (section coherence)
3. Merge small chunks if < MIN_CHUNK_SIZE
4. Split large chunks at paragraph boundaries
5. Add overlap for continuity

**Chunk Metadata**:
```python
{
    "source": "document_name.md",
    "page": 42,
    "markdown_content": "# Section\nContent...",
    "plain_text": "Section Content...",
    "keywords": ["keyword1", "keyword2", ...],
    "header_path": "Chapter 1 > Section 2 > Subsection 3"
}
```

### Stage 4: Vector Indexing

**Dual Content Strategy**:
- **Embedding**: Plain text (better semantic similarity)
- **Retrieval**: Markdown (better for generation with structure)

**Process**:
```python
# Generate embeddings from plain text
embeddings = embed_documents([chunk.plain_text for chunk in chunks])

# Store markdown in metadata for generation
for i, chunk in enumerate(chunks):
    vectorstore.add_documents(
        texts=[chunk.plain_text],
        embeddings=[embeddings[i]],
        metadatas=[{
            "markdown_content": chunk.markdown_text,
            "keywords": chunk.keywords,
            "source": chunk.source
        }]
    )
```

**Index Optimization**:
- FAISS IndexFlatIP (inner product) for cosine similarity
- Batch processing for memory efficiency
- Persistent storage for offline use

---

## RAG Pipeline

### Pipeline Execution Flow

#### Synchronous Mode (Traditional)

```python
result = pipeline.run("How to enable temperature compensation?")
# Returns complete result with answer, documents, metadata
```

**Flow**:
1. Query processing → transformed query
2. Retrieval → relevant documents
3. Document processing → filtered/reranked docs
4. Answer generation → complete response
5. Memory update → conversation state

#### Asynchronous Mode (Streaming)

```python
async for chunk in pipeline.async_run("query"):
    if "token" in chunk:
        display_token(chunk["token"])  # Real-time display
    elif "final_state" in chunk:
        process_results(chunk["final_state"])  # Final metadata
```

**Flow**:
1. **Pre-streaming**: All components up to AnswerGenerator
2. **Streaming**: Token-by-token generation
3. **Post-streaming**: Memory update, final logging

**Implementation**:
```python
class RAGPipeline:
    async def async_run(self, query):
        state = self._build_initial_state(query)
        
        # Execute pre-streaming components
        for component in self.pre_streaming_components:
            state = component.process(state)
        
        # Stream generation
        async for token in self.answer_generator.stream(state):
            yield {"token": token}
        
        # Post-processing
        state = self._finalize_state(state)
        yield {"final_state": state}
```

### Component Details

#### Pre-Retrieval: Query Processing

**QueryTransformer**:
```
Input: "How do I enable it?"
↓
LLM Transformation: "Reformulate for clarity"
↓
Output: "How to enable temperature compensation?"
```

**QueryDecomposer**:
```
Input: "How to export data and compare with DXF?"
↓
Decomposition: [
    "How to export measurement data?",
    "How to compare profile with DXF file?"
]
```

**SelfContainedQueryGenerator**:
```
Previous: "Where is Paris?"
Current: "What is its population?"
↓
Context Integration
↓
Output: "What is the population of Paris?"
```

#### Retrieval: Hybrid Search

**Vector Retrieval** (70% weight):
```python
# Generate query embedding
query_embedding = embedder.embed_query(query)

# Search vector store
docs = vectorstore.similarity_search(
    embedding=query_embedding,
    k=10,
    threshold=0.5
)
```

**BM25 Retrieval** (30% weight):
```python
# Tokenize and score
tokens = tokenize(query)
scores = bm25.get_scores(tokens)

# Retrieve top-k
docs = get_top_k_documents(scores, k=10)
```

**Fusion**:
```python
# Reciprocal Rank Fusion (RRF)
def fuse_results(vector_docs, bm25_docs, weights):
    combined = {}
    for doc, rank in vector_docs:
        combined[doc] = weights[0] / (rank + 60)
    for doc, rank in bm25_docs:
        combined[doc] += weights[1] / (rank + 60)
    return sorted(combined.items(), key=lambda x: -x[1])
```

#### Post-Retrieval: Document Processing

**Reranking** (FlashRank):
```python
# Re-score documents for relevance
reranker = FlashRank()
reranked_docs = reranker.rerank(
    query=query,
    documents=retrieved_docs,
    top_k=5
)
```

**Header Boosting**:
```python
# Boost documents with matching headers
for doc in documents:
    header_similarity = semantic_similarity(
        query, 
        doc.metadata["header_path"]
    )
    if header_similarity > 0.7:
        doc.score *= 1.5  # Boost score
```

**Compression**:
```python
# LLM-based prompt compression
compressor = PromptCompressor(llm=utility_llm)
compressed = compressor.compress(
    documents=documents,
    query=query,
    max_tokens=1000
)
```

#### Generation: Answer Generation

**Standard RAG**:
```python
prompt = f"""
Use the following context to answer the question.

Context:
{format_documents(documents)}

Question: {query}

Answer:
"""

answer = llm.invoke(prompt)
```

**Streaming RAG**:
```python
async for chunk in llm.astream(prompt):
    token = chunk.content
    yield token
    full_answer += token
```

---

## Advanced Features

### Memory Management

**Problem**: Limited context window on small LLMs

**Solution**: Intelligent query routing

```python
class MemoryManager:
    def classify_query(self, query, history):
        # Classify into: "history", "external", "both"
        classification = classifier.classify(query, history)
        
        if classification == "history":
            return history_generator.generate(query, history)
        elif classification == "external":
            return rag_generator.generate(query)
        else:  # "both"
            # Reformulate to self-contained query
            standalone = self_contained_generator.generate(query, history)
            return rag_generator.generate(standalone)
```

**Conversation Storage**:
```python
class SlidingWindowMemory:
    def __init__(self, window_size=10):
        self.messages = deque(maxlen=window_size)
    
    def add_exchange(self, question, answer):
        self.messages.append({"Q": question, "A": answer})
    
    def get_context(self):
        return "\n".join([
            f"Q: {m['Q']}\nA: {m['A']}" 
            for m in self.messages
        ])
```

### Multilingual Support

**Architecture**:

```
Input Query (IT/DE/Other)
    ↓
Query Translation → English
    ↓
RAG Pipeline (English)
    ↓
Answer Translation → Original Language
    ↓
Output (IT/DE/Other)
```

**Implementation**:
```python
class TranslationPipeline:
    def __init__(self, source_lang, target_lang):
        self.query_translator = MarianMTTranslator(
            model=f"Helsinki-NLP/opus-mt-{source_lang}-en"
        )
        self.answer_translator = MarianMTTranslator(
            model=f"Helsinki-NLP/opus-mt-en-{target_lang}"
        )
    
    def process(self, query):
        # Translate query to English
        en_query = self.query_translator.translate(query)
        
        # Run RAG in English
        en_answer = self.rag.run(en_query)
        
        # Translate answer back
        translated_answer = self.answer_translator.translate(en_answer)
        return translated_answer
```

**Model Caching**:
```python
class TranslationModelManager:
    def __init__(self):
        self.cache = {}
    
    def get_model(self, model_name):
        if model_name not in self.cache:
            self.cache[model_name] = download_and_load(model_name)
        return self.cache[model_name]
```

### Evaluation System

**Custom Metrics**:

```python
class MetricsEvaluator:
    def evaluate_sample(self, question, answer, ground_truth, contexts):
        return {
            "response_relevancy": self._response_relevancy(question, answer),
            "factual_correctness": self._factual_correctness(answer, ground_truth),
            "faithfulness": self._faithfulness(answer, contexts),
            "context_precision": self._context_precision(contexts, ground_truth),
            "context_recall": self._context_recall(contexts, ground_truth)
        }
    
    def _response_relevancy(self, question, answer):
        # Embedding-based semantic similarity
        q_embedding = self.embedder.embed(question)
        a_embedding = self.embedder.embed(answer)
        return cosine_similarity(q_embedding, a_embedding)
    
    def _factual_correctness(self, answer, ground_truth):
        # LLM-based claim verification
        claims = self._extract_claims(answer)
        correct = sum(
            self._verify_claim(claim, ground_truth)
            for claim in claims
        )
        return correct / len(claims) if claims else 0.0
```

**Batch Evaluation**:
```python
def evaluate_samples(evaluation_data, rag_type="pipeline"):
    rag = create_rag(rag_type)
    results = []
    
    for sample in evaluation_data:
        # Run RAG
        result = rag.run(sample["question"])
        
        # Evaluate
        metrics = evaluator.evaluate_sample(
            question=sample["question"],
            answer=result.answer,
            ground_truth=sample["answer"],
            contexts=sample["contexts"]
        )
        
        results.append({
            "sample": sample,
            "result": result,
            "metrics": metrics
        })
    
    return aggregate_results(results)
```

---

## Configuration System

### Three-Layer Configuration

**1. Global Configuration** (viviangpt/config.py):
```python
DEFAULT_CONFIG = {
    "llm": {
        "provider": "ollama",
        "model_name": "llama3.2:3b-instruct-q5_K_M",
        "temperature": 0.1,
        "num_ctx": 4096
    },
    "embeddings": {
        "model": "mxbai-embed-large",
        "provider": "ollama"
    },
    "retriever": {
        "k": 7,
        "threshold": 0.5
    }
}
```

**2. Builder Configuration**:
```python
pipeline = (
    RAGPipelineBuilder()
    .with_llm(model_name="llama3.2:8b")  # Override global
    .with_reranking(enabled=True, k=5)
    .build()
)
```

**3. Runtime Configuration**:
```python
import viviangpt

viviangpt.configure(
    llm={"temperature": 0.2},
    pipeline={"compression": True}
)
```

### Configuration Precedence

```
Runtime Config > Builder Config > Global Config > Defaults
```

---

## Deployment

### Offline Package Structure

```
ViViAnGPT_Package/
├── python_embedded/           # Portable Python 3.12.7
├── dependencies/              # Pre-downloaded .whl files
├── app/                       # Application code
│   ├── viviangpt/            # Core library
│   ├── ui/                   # Streamlit interface
│   ├── data/                 # Vector stores
│   └── models/               # Downloaded models
├── install.bat               # Installation script
├── launch_viviangpt.bat      # Launch script
└── README.txt                # Setup instructions
```

### Package Creation Process

**Step 1**: Download dependencies
```bash
pip download -r requirements.txt -d dependencies/
```

**Step 2**: Bundle Python
```bash
# Download embedded Python
wget https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip

# Extract
unzip python-3.12.7-embed-amd64.zip -d python_embedded/
```

**Step 3**: Copy application
```bash
cp -r viviangpt/ app/
cp -r ui/ app/
cp -r data/ app/
```

**Step 4**: Create installers
```batch
@echo off
cd /d "%~dp0"
python_embedded\python.exe -m pip install --no-index --find-links=dependencies -r requirements.txt
python_embedded\python.exe -m pip install -e app/
```

### Installation on Target Machine

```batch
1. Copy ViViAnGPT_Package to target
2. Run install.bat (5-15 minutes)
3. Launch via launch_viviangpt.bat
4. Access UI at http://localhost:8501
```

### System Requirements

**Minimum**:
- Windows 10+
- 8GB RAM
- 4GB disk space
- CPU: modern x64

**Recommended**:
- 16GB+ RAM
- 8GB+ disk space
- GPU: NVIDIA with CUDA 12.8+

---

## Performance Optimization

### CPU Optimization

**Quantization**: q5_K_M models for 3B LLMs
```
Original: 12GB VRAM
Quantized: 2GB RAM
Quality: ~95% of original
```

**Batch Processing**:
```python
# Embed multiple chunks at once
embeddings = embedder.embed_documents(chunks, batch_size=32)
```

**Caching**:
```python
@lru_cache(maxsize=128)
def get_embedding(text):
    return embedder.embed_query(text)
```

### GPU Acceleration

**Automatic Detection**:
```python
if torch.cuda.is_available():
    device = "cuda"
    # Ollama automatically uses GPU
else:
    device = "cpu"
```

**Performance Gain**:
- Embedding: 10-20x faster
- Generation: 15-30x faster
- Overall: ~20x speedup

---

## Code Quality

### Design Patterns

- **Builder Pattern**: Pipeline configuration
- **Factory Pattern**: Component creation
- **Strategy Pattern**: Retrieval strategies
- **Decorator Pattern**: Streaming wrapper
- **Observer Pattern**: Logging system

### Testing Strategy

**Unit Tests**: Component isolation
```python
def test_query_transformer():
    transformer = QueryTransformer(llm=mock_llm)
    result = transformer.process("vague query")
    assert len(result) > len("vague query")
```

**Integration Tests**: Pipeline flow
```python
def test_pipeline_flow():
    pipeline = RAGPipelineBuilder().build()
    result = pipeline.run("test query")
    assert result.answer is not None
    assert len(result.documents) > 0
```

**Evaluation Tests**: System performance
```python
def test_evaluation_metrics():
    results = evaluate_samples(test_dataset)
    assert results["avg_faithfulness"] > 0.8
```

---

## Future Enhancements

### Potential Improvements

1. **Multi-Modal**: Image understanding in diagrams
2. **Active Learning**: Feedback loop for improvement
3. **Fine-Tuning**: Domain-specific model adaptation
4. **Distributed**: Multi-machine deployment
5. **API**: REST/GraphQL interface

### Scalability Considerations

- **Horizontal**: Multiple vector stores for different domains
- **Vertical**: GPU cluster for high-throughput
- **Caching**: Redis for frequently asked questions
- **Load Balancing**: Multiple Ollama instances

---

**This architecture demonstrates production-level engineering with focus on modularity, performance, and maintainability.**
