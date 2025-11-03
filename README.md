# ViViAnGPT: Intelligent Documentation Assistant

> A production-ready Retrieval-Augmented Generation (RAG) system that transforms technical documentation into an intelligent Q&A assistant, designed for offline operation in resource-constrained environments.

## 🎯 Project Overview

ViViAnGPT is an AI-powered documentation assistant developed for Metrios, a professional photogrammetry software. The system addresses the challenge of navigating complex technical documentation (300+ pages across 8 PDFs) by providing instant, context-aware answers to user questions.

**Key Achievement**: Transformed static documentation into an intelligent assistant capable of understanding context, retrieving relevant information, and generating precise answers—all while operating completely offline on hardware with limited resources.

## 🚀 Technical Highlights

### Architecture & Design

**Modular RAG Pipeline**: Built on a flexible, component-based architecture using the Builder Pattern, allowing dynamic configuration and easy testing of different RAG strategies.

- **Pre-Retrieval**: Query transformation, decomposition, and rephrasing for optimal search
- **Retrieval**: Hybrid search combining vector similarity (FAISS) and keyword matching (BM25)
- **Post-Retrieval**: Document reranking, compression, and intelligent filtering
- **Generation**: Context-aware response generation with conversation memory

**Key Technologies**:

- **Ollama**: Local LLM inference (Llama 3.2)
- **LangChain**: RAG orchestration and component integration
- **FAISS**: High-performance vector similarity search
- **Streamlit**: Interactive web interface
- **Marker-PDF**: Advanced PDF-to-Markdown conversion

### Offline-First Design

Engineered for complete autonomy in air-gapped environments:

- ✅ No internet connection required
- ✅ Local LLM inference (3B parameter models)
- ✅ Embedded Python distribution
- ✅ Optimized for CPU-only operation
- ✅ Minimal RAM footprint (8GB minimum)

### Advanced Features

**Intelligent Document Processing**:

- Structure-preserving PDF conversion
- Header-aware semantic chunking (800-1500 characters)
- Automatic margin detection and noise removal
- Dynamic chunk merging and splitting

**Smart Retrieval**:

- Hybrid search (70% semantic + 30% lexical)
- FlashRank reranking for precision
- Header boosting for contextual relevance
- Keyword filtering using n-gram extraction

**Multilingual Support**:

- Native support for EN, IT, DE documentation
- Automatic translation for other languages (MarianMT)
- Bidirectional query/answer translation pipeline

**Conversation Management**:

- Sliding window memory for multi-turn dialogues
- Intelligent query classification (history/external/both)
- Context-free query reformulation for follow-ups

**Response Streaming**:

- Real-time token-by-token display
- Asynchronous pipeline execution
- Optimized user experience for CPU-based inference

## 📊 Performance & Evaluation

### Custom Evaluation Framework

Developed a lightweight evaluation system optimized for resource-constrained environments:

**Core Metrics**:

- **Response Relevancy**: Semantic alignment between question and answer
- **Factual Correctness**: Accuracy of information vs. ground truth
- **Faithfulness**: Response adherence to retrieved documents
- **Context Precision/Recall**: Quality and completeness of retrieval

**Results**: Comparative testing between baseline RAG and advanced pipeline configurations, with automated JSON/CSV reporting for performance tracking.

### System Performance

- **Indexing**: ~300 pages processed into searchable vector store
- **Retrieval**: Sub-second document search on CPU
- **Generation**: 2-5 seconds per response (CPU-only)
- **GPU Acceleration**: 10-20x speedup when available

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Streamlit Web App)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                    RAG Pipeline Engine                       │
├──────────────────────────────────────────────────────────────┤
│  Query Processing → Retrieval → Doc Processing → Generation  │
│                                                               │
│  • Query Transform      • Vector Search    • Reranking       │
│  • Decomposition        • BM25 Search      • Compression     │
│  • Rephrasing          • Hybrid Fusion    • Filtering        │
│                                                               │
│               Memory Manager • Translation Layer             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      Core Services                           │
├──────────────────────────────────────────────────────────────┤
│  LLM Service    • Embeddings     • Vector Store (FAISS)      │
│  (Ollama)      • (mxbai-embed)   • BM25 Index                │
└──────────────────────────────────────────────────────────────┘
```

### Indexing Pipeline

```
PDF Documents → marker-pdf Conversion → Markdown Preprocessing
                                                ↓
    FAISS Vector Store ← Embedding Generation ← Intelligent Chunking
```

**Innovative Features**:

- Document-specific margin configurations for noise removal
- Header hierarchy preservation for semantic coherence
- Dual content storage: plain text for embeddings, markdown for generation
- N-gram keyword extraction for fast filtering

## 💼 Business Impact

### Problem Solved

**Before**: Users struggled to find specific information across 300+ pages of inconsistently formatted PDF documentation, leading to inefficient workflows and support overhead.

**After**: Instant, accurate answers to technical questions with source attribution, dramatically reducing time spent searching documentation.

### Use Case Example

**User Query**: "How can I enable temperature compensation?"

**System Response**:

- Retrieves relevant sections from multiple documents
- Synthesizes step-by-step instructions
- Provides technical parameters and considerations
- Shows source documents for verification
- Handles follow-up questions with context awareness

## 🛠️ Technical Skills Demonstrated

### AI/ML Engineering

- RAG system design and optimization
- Vector embedding and similarity search
- LLM prompt engineering and fine-tuning
- Hybrid retrieval strategies
- Custom evaluation metrics development

### Backend Engineering

- Modular architecture (Builder Pattern, Factory Pattern)
- Service-oriented design
- Asynchronous processing and streaming
- Component abstraction and reusability
- Offline deployment and packaging

### Data Processing

- PDF extraction and preprocessing
- Document chunking strategies
- Vector database optimization
- Multilingual NLP pipeline
- OCR and vision-language models (VLM)

### Software Engineering Best Practices

- Clean architecture principles
- Comprehensive configuration management
- Automated testing and evaluation
- Documentation and code organization
- Git version control

## 🔧 Development Workflow

### Iterative Optimization

1. **Baseline Implementation**: Simple retrieval + generation
2. **Advanced Pipeline**: Modular components for testing strategies
3. **Evaluation Framework**: Automated performance measurement
4. **Configuration Testing**: A/B testing different techniques
5. **Production Optimization**: Resource usage and speed tuning

### Key Engineering Decisions

**Why Ollama?** Local inference with excellent performance/resource ratio

**Why FAISS?** CPU-optimized vector search with minimal dependencies

**Why Marker-PDF?** Superior structure preservation vs. alternatives

**Why Modular Design?** Enable rapid experimentation and configuration testing

**Why Custom Evaluation?** Standard frameworks (RAGAS, DeepEval) too resource-intensive

## 📈 Project Evolution

### Version History

- **v0.1**: Basic RAG with vector retrieval
- **v0.2**: Advanced pipeline with reranking and compression
- **v0.3**: Production-ready with streaming, multilingual support, and offline packaging

### Lessons Learned

**Technical Insights**:

- Hybrid search significantly outperforms pure vector or keyword search
- Header-aware chunking crucial for maintaining context
- Prompt compression enables larger context windows on small models
- Streaming improves perceived performance on CPU systems

**Engineering Insights**:

- Modular design essential for experimentation
- Custom evaluation faster than adapting existing frameworks
- Manual markdown curation superior to automated conversion for stable docs
- Offline deployment requires careful dependency management

## 🎓 Educational Value

This project demonstrates expertise in:

✅ **AI Engineering**: End-to-end RAG system development
✅ **Backend Development**: Scalable, modular architecture
✅ **Data Science**: Document processing and evaluation metrics
✅ **DevOps**: Offline deployment and packaging
✅ **Problem Solving**: Resource-constrained optimization
✅ **Production ML**: Real-world system design and deployment

## 🔒 Proprietary Considerations

**What's Public**: Architecture, methodology, technical decisions, and general approach.

**What's Private**: Actual Metrios documentation content, specific prompts, and business logic.

This documentation demonstrates the technical sophistication and engineering rigor of the project without revealing proprietary information.

---

## 📚 Additional Resources

For a detailed technical deep-dive, see companion documents:

- [TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md) — Detailed component breakdown and design patterns
- [RESULTS_AND_FINDINGS.md](./RESULTS_AND_FINDINGS.md) — Performance metrics, evaluation results, and insights

---

**Project Status**: Production-ready (v1.0.1-stable)
**Primary Language**: Python 3.9+
**License**: MIT (for architecture, not documentation content)
**Key Frameworks**: LangChain, FAISS, Ollama, Streamlit
