# Estratégias e Modelos para Conversão de Texto em Embeddings para RAG

Para implementar um sistema RAG (Retrieval-Augmented Generation) eficaz, a conversão de documentos de texto em embeddings de alta qualidade é fundamental. Vou explicar as melhores estratégias e modelos disponíveis para esta tarefa.

## O que são Embeddings Vetoriais?

Embeddings são representações numéricas de texto em um espaço vetorial multidimensional, onde textos semanticamente similares ficam próximos uns dos outros. Estes vetores capturam o significado semântico do texto e são essenciais para sistemas RAG.

## Melhores Estratégias para Geração de Embeddings

### 1. Chunking (Segmentação) Eficiente

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Divide o texto em chunks com sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Ajuste para não cortar palavras no meio
        if end < len(text) and not text[end].isspace():
            # Recua até encontrar um espaço
            while end > start and not text[end].isspace():
                end -= 1
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Considerações importantes:**

- **Tamanho do chunk**: Chunks menores (300-500 tokens) são melhores para recuperação precisa, enquanto chunks maiores preservam mais contexto.
- **Sobreposição**: Uma sobreposição de 10-20% ajuda a manter a coerência entre chunks.
- **Divisão semântica**: Dividir por parágrafos ou seções lógicas em vez de simplesmente por contagem de tokens.

### 2. Pré-processamento de Texto

```python
def preprocess_text(text):
    """Pré-processa o texto antes da geração de embeddings."""
    # Remover caracteres especiais desnecessários
    text = re.sub(r'[^\w\s.,;:!?()]', '', text)
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()
    # Outras normalizações conforme necessário
    return text
```

### 3. Enriquecimento de Metadados

```python
def create_document_with_metadata(chunk, source, page_num=None, section=None):
    """Adiciona metadados ao chunk para melhorar a recuperação."""
    return {
        "text": chunk,
        "metadata": {
            "source": source,
            "page": page_num,
            "section": section,
            "chunk_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
    }
```

## Melhores Modelos para Geração de Embeddings

### 1. Modelos da OpenAI

- **text-embedding-3-large**: 3072 dimensões, melhor qualidade
- **text-embedding-3-small**: 1536 dimensões, boa relação custo-benefício
- **text-embedding-ada-002**: 1536 dimensões (modelo mais antigo)

### 2. Modelos Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo leve e rápido
# ou
model = SentenceTransformer('all-mpnet-base-v2')  # Melhor qualidade

embeddings = model.encode(chunks)
```

Modelos recomendados:

- **all-mpnet-base-v2**: Excelente qualidade (768 dimensões)
- **all-MiniLM-L6-v2**: Bom equilíbrio entre velocidade e qualidade (384 dimensões)
- **multi-qa-mpnet-base-dot-v1**: Otimizado para recuperação de documentos

### 3. Modelos Multilíngues

- **paraphrase-multilingual-mpnet-base-v2**: Excelente para múltiplos idiomas
- **LaBSE**: Language-agnostic BERT Sentence Embedding (Google)

### 4. Modelos Específicos de Domínio

Para domínios específicos como medicina, finanças ou direito, considere:

- **BioSentVec** (biomédico)
- **Legal-BERT** (jurídico)
- **FinBERT** (financeiro)

## Implementação Prática

Exemplo de fluxo completo:

```python
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS

# 1. Carregar e chunkar o documento
with open('documento.txt', 'r') as f:
    text = f.read()

chunks = chunk_text(text, chunk_size=500, overlap=50)

# 2. Gerar embeddings
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(chunks)

# 3. Criar documentos com metadados
documents = [
    create_document_with_metadata(chunk, "documento.txt", section="main")
    for chunk in chunks
]

# 4. Armazenar no banco de dados vetorial (usando FAISS como exemplo)
texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

vector_store = FAISS.from_texts(
    texts=texts,
    embedding=model,
    metadatas=metadatas
)

# 5. Salvar para uso posterior
vector_store.save_local("faiss_index")
```

## Considerações Adicionais

### Avaliação da Qualidade dos Embeddings

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calcular similaridade entre dois chunks
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

### Otimização de Dimensionalidade

Para bancos de dados muito grandes, considere técnicas de redução de dimensionalidade como PCA ou UMAP para melhorar a eficiência de armazenamento e consulta.

### Bancos de Dados Vetoriais Recomendados

- **Pinecone**: Serviço gerenciado, escalonável
- **Weaviate**: Banco de dados vetorial de código aberto
- **Qdrant**: Otimizado para pesquisa semântica
- **FAISS (Facebook AI Similarity Search)**: Biblioteca de alto desempenho
- **Chroma**: Simples e fácil de usar
- **Milvus**: Sistema de código aberto para grandes volumes de dados

## Conclusão

A escolha do modelo de embedding e da estratégia de chunking deve ser baseada nas necessidades específicas do seu projeto, considerando:

1. Qualidade semântica necessária
2. Requisitos de velocidade
3. Restrições de custo
4. Suporte a idiomas
5. Domínio específico do conteúdo

Uma abordagem iterativa, testando diferentes combinações de modelos, tamanhos de chunk e estratégias de processamento, geralmente leva aos melhores resultados para sistemas RAG.
( O)> Como seria esta abordagem iterativa, testando diferentes combinações de modelos?

# Abordagem Iterativa para Testar Combinações de Modelos de Embeddings

Uma abordagem iterativa para encontrar a melhor combinação de modelos, parâmetros de chunking e estratégias de processamento para seu sistema RAG envolve experimentação sistemática e avaliação de resultados. Vou detalhar como implementar este processo de forma estruturada.

## 1. Definição de Métricas de Avaliação

Antes de começar os testes, é crucial definir métricas claras para avaliar o desempenho:

```python
def evaluate_embedding_quality(model, test_queries, relevant_documents):
    """
    Avalia a qualidade do modelo de embedding usando métricas de recuperação.
    """
    results = {}

    # Gerar embeddings para as queries
    query_embeddings = model.encode(test_queries)

    # Para cada query, calcular métricas de recuperação
    for i, query in enumerate(test_queries):
        # Simular recuperação
        retrieved_docs = retrieve_top_k(query_embeddings[i], document_embeddings, k=10)

        # Calcular métricas
        precision = calculate_precision(retrieved_docs, relevant_documents[i])
        recall = calculate_recall(retrieved_docs, relevant_documents[i])
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mrr = calculate_mrr(retrieved_docs, relevant_documents[i])

        results[query] = {
            "precision@10": precision,
            "recall@10": recall,
            "f1": f1,
            "mrr": mrr
        }

    # Calcular médias
    avg_results = {
        "avg_precision": sum(r["precision@10"] for r in results.values()) / len(results),
        "avg_recall": sum(r["recall@10"] for r in results.values()) / len(results),
        "avg_f1": sum(r["f1"] for r in results.values()) / len(results),
        "avg_mrr": sum(r["mrr"] for r in results.values()) / len(results)
    }

    return results, avg_results
```

## 2. Criação de um Conjunto de Teste

```python
def create_test_set():
    """
    Cria um conjunto de consultas de teste e documentos relevantes esperados.
    """
    test_queries = [
        "Como funciona a tecnologia blockchain?",
        "Quais são os benefícios da energia solar?",
        "O que é aprendizado por reforço em IA?",
        # Adicione mais consultas representativas do seu domínio
    ]

    # Para cada query, liste os IDs dos documentos que deveriam ser recuperados
    relevant_docs = [
        ["doc_123", "doc_456", "doc_789"],  # Documentos relevantes para query 1
        ["doc_234", "doc_567", "doc_890"],  # Documentos relevantes para query 2
        ["doc_345", "doc_678", "doc_901"],  # Documentos relevantes para query 3
        # ...
    ]

    return test_queries, relevant_docs
```

## 3. Framework de Experimentação

Aqui está um framework completo para testar diferentes combinações:

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import time

def test_embedding_combinations(document_path, test_queries, relevant_docs):
    """
    Testa diferentes combinações de modelos, tamanhos de chunk e estratégias.
    """
    # Carregar documento
    with open(document_path, 'r') as f:
        full_text = f.read()

    # Modelos a testar
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1"
    ]

    # Configurações de chunking
    chunk_configs = [
        {"size": 256, "overlap": 20},
        {"size": 512, "overlap": 50},
        {"size": 1024, "overlap": 100}
    ]

    # Estratégias de pré-processamento
    preprocessing_strategies = [
        "basic",       # Apenas normalização básica
        "advanced",    # Remoção de stopwords, lemmatização, etc.
        "semantic"     # Divisão por parágrafos ou seções semânticas
    ]

    results = []

    # Iterar por todas as combinações
    for model_name in tqdm(models, desc="Modelos"):
        try:
            model = SentenceTransformer(model_name)

            for chunk_config in tqdm(chunk_configs, desc="Configurações de Chunk", leave=False):
                for strategy in preprocessing_strategies:
                    # Registrar início do teste
                    start_time = time.time()

                    # Aplicar pré-processamento conforme a estratégia
                    if strategy == "basic":
                        processed_text = basic_preprocess(full_text)
                    elif strategy == "advanced":
                        processed_text = advanced_preprocess(full_text)
                    else:  # semantic
                        processed_text = semantic_preprocess(full_text)

                    # Chunking
                    chunks = chunk_text(processed_text,
                                       chunk_size=chunk_config["size"],
                                       overlap=chunk_config["overlap"])

                    # Gerar embeddings
                    doc_embeddings = model.encode(chunks)

                    # Avaliar qualidade
                    _, metrics = evaluate_embedding_quality(
                        model, test_queries, relevant_docs
                    )

                    # Calcular tempo e uso de memória
                    processing_time = time.time() - start_time
                    memory_usage = sum(sys.getsizeof(e) for e in doc_embeddings)

                    # Registrar resultados
                    results.append({
                        "model": model_name,
                        "chunk_size": chunk_config["size"],
                        "chunk_overlap": chunk_config["overlap"],
                        "preprocessing": strategy,
                        "precision": metrics["avg_precision"],
                        "recall": metrics["avg_recall"],
                        "f1": metrics["avg_f1"],
                        "mrr": metrics["avg_mrr"],
                        "processing_time_seconds": processing_time,
                        "memory_usage_bytes": memory_usage,
                        "embedding_dimensions": len(doc_embeddings[0])
                    })

        except Exception as e:
            print(f"Erro ao testar modelo {model_name}: {e}")

    # Converter para DataFrame para análise
    results_df = pd.DataFrame(results)

    # Salvar resultados
    results_df.to_csv("embedding_test_results.csv", index=False)

    return results_df
```

## 4. Visualização e Análise dos Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(results_df):
    """
    Analisa e visualiza os resultados dos testes.
    """
    # Configurar estilo
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)

    # 1. Comparação de modelos (F1 score)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='f1', data=results_df)
    plt.title('Comparação de Modelos por F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison_f1.png')

    # 2. Relação entre tamanho do chunk e precisão
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='chunk_size', y='precision', hue='model', data=results_df)
    plt.title('Impacto do Tamanho do Chunk na Precisão')
    plt.tight_layout()
    plt.savefig('chunk_size_precision.png')

    # 3. Trade-off entre velocidade e qualidade
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='processing_time_seconds', y='f1',
                   hue='model', size='chunk_size', data=results_df)
    plt.title('Trade-off entre Tempo de Processamento e Qualidade (F1)')
    plt.tight_layout()
    plt.savefig('time_quality_tradeoff.png')

    # 4. Encontrar a melhor combinação
    best_by_f1 = results_df.loc[results_df['f1'].idxmax()]
    best_by_mrr = results_df.loc[results_df['mrr'].idxmax()]
    best_efficiency = results_df.loc[results_df['f1'] / results_df['processing_time_seconds'].idxmax()]

    print("Melhor combinação por F1:", best_by_f1.to_dict())
    print("Melhor combinação por MRR:", best_by_mrr.to_dict())
    print("Melhor eficiência (F1/tempo):", best_efficiency.to_dict())

    return best_by_f1, best_by_mrr, best_efficiency
```

## 5. Validação com Consultas Reais

Após identificar as melhores combinações, é importante validá-las com consultas reais:

```python
def validate_with_real_queries(best_model, best_chunk_config, best_preprocessing):
    """
    Valida a melhor combinação com consultas reais de usuários.
    """
    # Carregar modelo
    model = SentenceTransformer(best_model)

    # Carregar e processar documento usando as melhores configurações
    with open(document_path, 'r') as f:
        full_text = f.read()

    if best_preprocessing == "basic":
        processed_text = basic_preprocess(full_text)
    elif best_preprocessing == "advanced":
        processed_text = advanced_preprocess(full_text)
    else:
        processed_text = semantic_preprocess(full_text)

    chunks = chunk_text(processed_text,
                       chunk_size=best_chunk_config["size"],
                       overlap=best_chunk_config["overlap"])

    # Gerar embeddings
    doc_embeddings = model.encode(chunks)

    # Criar banco de dados vetorial
    vector_db = create_vector_db(chunks, doc_embeddings)

    # Testar com consultas reais de usuários
    real_queries = [
        "Como implementar autenticação em dois fatores?",
        "Quais são os riscos de segurança em APIs REST?",
        # Mais consultas reais...
    ]

    # Avaliar respostas
    for query in real_queries:
        query_embedding = model.encode([query])[0]
        retrieved_docs = vector_db.search(query_embedding, top_k=3)

        print(f"\nConsulta: {query}")
        print("Documentos recuperados:")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. {doc[:100]}...")

        # Solicitar feedback humano
        relevance = input("Quão relevantes são esses resultados? (1-5): ")
        feedback = input("Algum comentário sobre os resultados? ")

        # Registrar feedback
        with open("validation_feedback.txt", "a") as f:
            f.write(f"Query: {query}\nRelevance: {relevance}\nFeedback: {feedback}\n\n")
```

## 6. Testes A/B com Sistemas RAG Completos

Para uma validação final, implemente sistemas RAG completos com as melhores combinações:

```python
def ab_test_rag_systems(combinations, test_queries):
    """
    Realiza testes A/B com sistemas RAG completos usando diferentes combinações.
    """
    results = {}

    for i, combo in enumerate(combinations):
        model_name = combo["model"]
        chunk_size = combo["chunk_size"]
        chunk_overlap = combo["chunk_overlap"]
        preprocessing = combo["preprocessing"]

        # Configurar sistema RAG com esta combinação
        rag_system = setup_rag_system(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preprocessing=preprocessing
        )

        # Testar com as consultas
        system_results = []
        for query in test_queries:
            # Obter resposta do sistema RAG
            response = rag_system.generate_response(query)

            # Registrar resposta
            system_results.append({
                "query": query,
                "response": response,
                # Aqui você pode adicionar métricas específicas de RAG
                # como relevância, precisão factual, etc.
            })

        results[f"system_{i+1}"] = {
            "config": combo,
            "responses": system_results
        }

    # Salvar resultados para análise humana
    with open("rag_ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
```

## 7. Pipeline Completo de Experimentação

Aqui está um exemplo de como executar o pipeline completo:

```python
def run_embedding_optimization_pipeline():
    """
    Executa o pipeline completo de otimização de embeddings.
    """
    print("1. Criando conjunto de teste...")
    test_queries, relevant_docs = create_test_set()

    print("2. Testando combinações de embeddings...")
    results_df = test_embedding_combinations(
        document_path="data/corpus.txt",
        test_queries=test_queries,
        relevant_docs=relevant_docs
    )

    print("3. Analisando resultados...")
    best_f1, best_mrr, best_efficiency = analyze_results(results_df)

    print("4. Validando com consultas reais...")
    validate_with_real_queries(
        best_model=best_f1["model"],
        best_chunk_config={
            "size": best_f1["chunk_size"],
            "overlap": best_f1["chunk_overlap"]
        },
        best_preprocessing=best_f1["preprocessing"]
    )

    print("5. Realizando testes A/B...")
    top_combinations = [
        best_f1.to_dict(),
        best_mrr.to_dict(),
        best_efficiency.to_dict()
    ]
    ab_results = ab_test_rag_systems(top_combinations, test_queries)

    print("6. Gerando relatório final...")
    generate_final_report(results_df, ab_results)

    return best_f1, ab_results
```

## 8. Refinamento Contínuo

O processo não termina com a seleção inicial. Implemente um sistema de feedback contínuo:

```python
def implement_continuous_feedback(rag_system):
    """
    Implementa um sistema de feedback contínuo para refinar o sistema RAG.
    """
    # Coletar feedback do usuário após cada resposta
    def collect_feedback(query, response):
        # Pode ser implementado via API, interface web, etc.
        feedback = {
            "query": query,
            "response": response,
            "relevance_score": user_rating,  # 1-5
            "comments": user_comments,
            "timestamp": datetime.now().isoformat()
        }

        # Armazenar feedback
        with open("user_feedback_log.jsonl", "a") as f:
            f.write(json.dumps(feedback) + "\n")

        return feedback

    # Analisar feedback periodicamente
    def analyze_feedback(feedback_file="user_feedback_log.jsonl", period_days=7):
        # Carregar feedback recente
        recent_feedback = []
        cutoff_date = datetime.now() - timedelta(days=period_days)

        with open(feedback_file, "r") as f:
            for line in f:
                feedback = json.loads(line)
                feedback_date = datetime.fromisoformat(feedback["timestamp"])
                if feedback_date >= cutoff_date:
                    recent_feedback.append(feedback)

        # Analisar tendências
        avg_relevance = sum(f["relevance_score"] for f in recent_feedback) / len(recent_feedback)
        problem_queries = [f for f in recent_feedback if f["relevance_score"] < 3]

        # Gerar relatório
        report = {
            "avg_relevance": avg_relevance,
            "total_feedback": len(recent_feedback),
            "low_relevance_percentage": len(problem_queries) / len(recent_feedback) * 100,
            "problem_areas": identify_problem_areas(problem_queries)
        }

        return report

    # Agendar análise periódica
    # Na prática, isso seria um job agendado
    schedule.every(7).days.do(lambda: analyze_feedback())
```

## 9. Exemplo de Relatório Final

```python
def generate_final_report(results_df, ab_results):
    """
    Gera um relatório final com recomendações.
    """
    report = {
        "summary": {
            "total_combinations_tested": len(results_df),
            "best_model": results_df.loc[results_df['f1'].idxmax()]["model"],
            "best_chunk_size": int(results_df.loc[results_df['f1'].idxmax()]["chunk_size"]),
            "best_preprocessing": results_df.loc[results_df['f1'].idxmax()]["preprocessing"]
        },
        "performance_metrics": {
            "best_f1": float(results_df['f1'].max()),
            "best_precision": float(results_df['precision'].max()),
            "best_recall": float(results_df['recall'].max()),
            "best_mrr": float(results_df['mrr'].max())
        },
        "efficiency_metrics": {
            "fastest_model": results_df.loc[results_df['processing_time_seconds'].idxmin()]["model"],
            "most_memory_efficient": results_df.loc[results_df['memory_usage_bytes'].idxmin()]["model"],
            "best_quality_speed_tradeoff": results_df.loc[(results_df['f1'] / results_df['processing_time_seconds']).idxmax()]["model"]
        },
        "recommendations": {
            "production_use": {
                "model": results_df.loc[results_df['f1'].idxmax()]["model"],
                "chunk_size": int(results_df.loc[results_df['f1'].idxmax()]["chunk_size"]),
                "chunk_overlap": int(results_df.loc[results_df['f1'].idxmax()]["chunk_overlap"]),
                "preprocessing": results_df.loc[results_df['f1'].idxmax()]["preprocessing"],
                "rationale": "Melhor equilíbrio entre qualidade e eficiência com base nos testes."
            },
            "resource_constrained": {
                "model": results_df.loc[(results_df['f1'] / results_df['memory_usage_bytes']).idxmax()]["model"],
                "rationale": "Melhor desempenho considerando restrições de recursos."
            }
        },
        "ab_test_summary": {
            "best_system": max(ab_results.items(), key=lambda x: sum(r.get("relevance_score", 0) for r in x[1]["responses"]))[0],
            "user_preference": "Sistema 1 preferido em 65% das consultas."  # Exemplo
        }
    }

    # Salvar como JSON
    with open("embedding_optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Gerar versão HTML para visualização
    html_report = f"""
    <html>
    <head>
        <title>Relatório de Otimização de Embeddings</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            .metric {{ margin-bottom: 20px; }}
            .recommendation {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Otimização de Embeddings para RAG</h1>

        <h2>Resumo</h2>
        <div class="metric">
            <p>Combinações testadas: {report["summary"]["total_combinations_tested"]}</p>
            <p>Melhor modelo: <strong>{report["summary"]["best_model"]}</strong></p>
            <p>Melhor tamanho de chunk: {report["summary"]["best_chunk_size"]}</p>
        </div>

        <h2>Métricas de Desempenho</h2>
        <div class="metric">
            <p>Melhor F1: {report["performance_metrics"]["best_f1"]:.4f}</p>
            <p>Melhor Precisão: {report["performance_metrics"]["best_precision"]:.4f}</p>
            <p>Melhor Recall: {report["performance_metrics"]["best_recall"]:.4f}</p>
        </div>

        <h2>Recomendações</h2>
        <div class="recommendation">
            <h3>Para Uso em Produção</h3>
            <p>Modelo: <strong>{report["recommendations"]["production_use"]["model"]}</strong></p>
            <p>Tamanho de Chunk: {report["recommendations"]["production_use"]["chunk_size"]}</p>
            <p>Sobreposição: {report["recommendations"]["production_use"]["chunk_overlap"]}</p>
            <p>Pré-processamento: {report["recommendations"]["production_use"]["preprocessing"]}</p>
            <p><em>{report["recommendations"]["production_use"]["rationale"]}</em></p>
        </div>

        <!-- Mais conteúdo do relatório -->
    </body>
    </html>
    """

    with open("embedding_report.html", "w") as f:
        f.write(html_report)

    print(f"Relatório gerado: embedding_optimization_report.json e embedding_report.html")
```

## Considerações Finais para a Abordagem Iterativa

1. **Comece com escopo limitado**: Inicie testando apenas 2-3 modelos com configurações básicas para estabelecer uma linha de base.

2. **Refine progressivamente**: Com base nos resultados iniciais, expanda os testes nas áreas mais promissoras.

3. **Automatize quando possível**: Crie scripts para automatizar a execução de testes em lote, especialmente para testes noturnos em grandes conjuntos de dados.

4. **Considere o contexto do domínio**: Para domínios específicos, dê mais peso a modelos especializados ou considere fine-tuning.

5. **Mantenha registros detalhados**: Documente todas as configurações e resultados para referência futura e reprodutibilidade.

6. **Feedback dos usuários finais**: Incorpore feedback qualitativo dos usuários finais do sistema RAG para complementar as métricas quantitativas.

Esta abordagem iterativa e sistemática permitirá que você encontre a combinação ideal de modelo de embedding, estratégia de chunking e pré-processamento para seu sistema RAG específico, resultando em recuperação mais precisa e respostas de maior qualidade.
( O)> Legal coloque as duas ultimas explicações em um arquivo de texto no formato Markdown neste diretório atual.
Vou criar um arquivo Markdown contendo as duas últimas explicações sobre estratégias de embeddings para RAG e a abordagem iterativa para testar diferentes combinações de modelos.
─── text_editor | developer ──────────────────────────
path: ~/estrategias_embeddings_rag.md
command: write

Vou criar um arquivo Markdown contendo as duas explicações sobre estratégias de embeddings para RAG e abordagem iterativa para testes de modelos.
─── text_editor | developer ──────────────────────────
path: ~/estrategias_embeddings_rag.md
command: write

Vou criar um arquivo Markdown contendo as duas explicações sobre estratégias de embeddings para RAG e abordagem iterativa para testes de modelos.
─── shell | developer ──────────────────────────

Vou criar um arquivo Markdown contendo as duas explicações sobre estratégias de embeddings para RAG e abordagem iterativa para testes de modelos.
─── shell | developer ──────────────────────────

Vou criar um arquivo Markdown contendo as duas explicações sobre estratégias de embeddings para RAG e abordagem iterativa para testes de modelos.
─── shell | developer ──────────────────────────
command: pwd

Entendo que estamos em um modo de chat que não permite o uso de ferramentas. Para criar o arquivo Markdown solicitado, vou fornecer o conteúdo completo que você pode copiar e salvar manualmente como um arquivo Markdown:

# Estratégias e Modelos para Conversão de Texto em Embeddings para RAG

Para implementar um sistema RAG (Retrieval-Augmented Generation) eficaz, a conversão de documentos de texto em embeddings de alta qualidade é fundamental. Vou explicar as melhores estratégias e modelos disponíveis para esta tarefa.

## O que são Embeddings Vetoriais?

Embeddings são representações numéricas de texto em um espaço vetorial multidimensional, onde textos semanticamente similares ficam próximos uns dos outros. Estes vetores capturam o significado semântico do texto e são essenciais para sistemas RAG.

## Melhores Estratégias para Geração de Embeddings

### 1. Chunking (Segmentação) Eficiente

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Divide o texto em chunks com sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Ajuste para não cortar palavras no meio
        if end < len(text) and not text[end].isspace():
            # Recua até encontrar um espaço
            while end > start and not text[end].isspace():
                end -= 1
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Considerações importantes:**

- **Tamanho do chunk**: Chunks menores (300-500 tokens) são melhores para recuperação precisa, enquanto chunks maiores preservam mais contexto.
- **Sobreposição**: Uma sobreposição de 10-20% ajuda a manter a coerência entre chunks.
- **Divisão semântica**: Dividir por parágrafos ou seções lógicas em vez de simplesmente por contagem de tokens.

### 2. Pré-processamento de Texto

```python
def preprocess_text(text):
    """Pré-processa o texto antes da geração de embeddings."""
    # Remover caracteres especiais desnecessários
    text = re.sub(r'[^\w\s.,;:!?()]', '', text)
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()
    # Outras normalizações conforme necessário
    return text
```

### 3. Enriquecimento de Metadados

```python
def create_document_with_metadata(chunk, source, page_num=None, section=None):
    """Adiciona metadados ao chunk para melhorar a recuperação."""
    return {
        "text": chunk,
        "metadata": {
            "source": source,
            "page": page_num,
            "section": section,
            "chunk_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
    }
```

## Melhores Modelos para Geração de Embeddings

### 1. Modelos da OpenAI

- **text-embedding-3-large**: 3072 dimensões, melhor qualidade
- **text-embedding-3-small**: 1536 dimensões, boa relação custo-benefício
- **text-embedding-ada-002**: 1536 dimensões (modelo mais antigo)

### 2. Modelos Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo leve e rápido
# ou
model = SentenceTransformer('all-mpnet-base-v2')  # Melhor qualidade

embeddings = model.encode(chunks)
```

Modelos recomendados:

- **all-mpnet-base-v2**: Excelente qualidade (768 dimensões)
- **all-MiniLM-L6-v2**: Bom equilíbrio entre velocidade e qualidade (384 dimensões)
- **multi-qa-mpnet-base-dot-v1**: Otimizado para recuperação de documentos

### 3. Modelos Multilíngues

- **paraphrase-multilingual-mpnet-base-v2**: Excelente para múltiplos idiomas
- **LaBSE**: Language-agnostic BERT Sentence Embedding (Google)

### 4. Modelos Específicos de Domínio

Para domínios específicos como medicina, finanças ou direito, considere:

- **BioSentVec** (biomédico)
- **Legal-BERT** (jurídico)
- **FinBERT** (financeiro)

## Implementação Prática

Exemplo de fluxo completo:

```python
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS

# 1. Carregar e chunkar o documento
with open('documento.txt', 'r') as f:
    text = f.read()

chunks = chunk_text(text, chunk_size=500, overlap=50)

# 2. Gerar embeddings
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(chunks)

# 3. Criar documentos com metadados
documents = [
    create_document_with_metadata(chunk, "documento.txt", section="main")
    for chunk in chunks
]

# 4. Armazenar no banco de dados vetorial (usando FAISS como exemplo)
texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

vector_store = FAISS.from_texts(
    texts=texts,
    embedding=model,
    metadatas=metadatas
)

# 5. Salvar para uso posterior
vector_store.save_local("faiss_index")
```

## Considerações Adicionais

### Avaliação da Qualidade dos Embeddings

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calcular similaridade entre dois chunks
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

### Otimização de Dimensionalidade

Para bancos de dados muito grandes, considere técnicas de redução de dimensionalidade como PCA ou UMAP para melhorar a eficiência de armazenamento e consulta.

### Bancos de Dados Vetoriais Recomendados

- **Pinecone**: Serviço gerenciado, escalonável
- **Weaviate**: Banco de dados vetorial de código aberto
- **Qdrant**: Otimizado para pesquisa semântica
- **FAISS (Facebook AI Similarity Search)**: Biblioteca de alto desempenho
- **Chroma**: Simples e fácil de usar
- **Milvus**: Sistema de código aberto para grandes volumes de dados

## Conclusão

A escolha do modelo de embedding e da estratégia de chunking deve ser baseada nas necessidades específicas do seu projeto, considerando:

1. Qualidade semântica necessária
2. Requisitos de velocidade
3. Restrições de custo
4. Suporte a idiomas
5. Domínio específico do conteúdo

Uma abordagem iterativa, testando diferentes combinações de modelos, tamanhos de chunk e estratégias de processamento, geralmente leva aos melhores resultados para sistemas RAG.

# Abordagem Iterativa para Testar Combinações de Modelos de Embeddings

Uma abordagem iterativa para encontrar a melhor combinação de modelos, parâmetros de chunking e estratégias de processamento para seu sistema RAG envolve experimentação sistemática e avaliação de resultados. Vou detalhar como implementar este processo de forma estruturada.

## 1. Definição de Métricas de Avaliação

Antes de começar os testes, é crucial definir métricas claras para avaliar o desempenho:

```python
def evaluate_embedding_quality(model, test_queries, relevant_documents):
    """
    Avalia a qualidade do modelo de embedding usando métricas de recuperação.
    """
    results = {}

    # Gerar embeddings para as queries
    query_embeddings = model.encode(test_queries)

    # Para cada query, calcular métricas de recuperação
    for i, query in enumerate(test_queries):
        # Simular recuperação
        retrieved_docs = retrieve_top_k(query_embeddings[i], document_embeddings, k=10)

        # Calcular métricas
        precision = calculate_precision(retrieved_docs, relevant_documents[i])
        recall = calculate_recall(retrieved_docs, relevant_documents[i])
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mrr = calculate_mrr(retrieved_docs, relevant_documents[i])

        results[query] = {
            "precision@10": precision,
            "recall@10": recall,
            "f1": f1,
            "mrr": mrr
        }

    # Calcular médias
    avg_results = {
        "avg_precision": sum(r["precision@10"] for r in results.values()) / len(results),
        "avg_recall": sum(r["recall@10"] for r in results.values()) / len(results),
        "avg_f1": sum(r["f1"] for r in results.values()) / len(results),
        "avg_mrr": sum(r["mrr"] for r in results.values()) / len(results)
    }

    return results, avg_results
```

## 2. Criação de um Conjunto de Teste

```python
def create_test_set():
    """
    Cria um conjunto de consultas de teste e documentos relevantes esperados.
    """
    test_queries = [
        "Como funciona a tecnologia blockchain?",
        "Quais são os benefícios da energia solar?",
        "O que é aprendizado por reforço em IA?",
        # Adicione mais consultas representativas do seu domínio
    ]

    # Para cada query, liste os IDs dos documentos que deveriam ser recuperados
    relevant_docs = [
        ["doc_123", "doc_456", "doc_789"],  # Documentos relevantes para query 1
        ["doc_234", "doc_567", "doc_890"],  # Documentos relevantes para query 2
        ["doc_345", "doc_678", "doc_901"],  # Documentos relevantes para query 3
        # ...
    ]

    return test_queries, relevant_docs
```

## 3. Framework de Experimentação

Aqui está um framework completo para testar diferentes combinações:

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import time

def test_embedding_combinations(document_path, test_queries, relevant_docs):
    """
    Testa diferentes combinações de modelos, tamanhos de chunk e estratégias.
    """
    # Carregar documento
    with open(document_path, 'r') as f:
        full_text = f.read()

    # Modelos a testar
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1"
    ]

    # Configurações de chunking
    chunk_configs = [
        {"size": 256, "overlap": 20},
        {"size": 512, "overlap": 50},
        {"size": 1024, "overlap": 100}
    ]

    # Estratégias de pré-processamento
    preprocessing_strategies = [
        "basic",       # Apenas normalização básica
        "advanced",    # Remoção de stopwords, lemmatização, etc.
        "semantic"     # Divisão por parágrafos ou seções semânticas
    ]

    results = []

    # Iterar por todas as combinações
    for model_name in tqdm(models, desc="Modelos"):
        try:
            model = SentenceTransformer(model_name)

            for chunk_config in tqdm(chunk_configs, desc="Configurações de Chunk", leave=False):
                for strategy in preprocessing_strategies:
                    # Registrar início do teste
                    start_time = time.time()

                    # Aplicar pré-processamento conforme a estratégia
                    if strategy == "basic":
                        processed_text = basic_preprocess(full_text)
                    elif strategy == "advanced":
                        processed_text = advanced_preprocess(full_text)
                    else:  # semantic
                        processed_text = semantic_preprocess(full_text)

                    # Chunking
                    chunks = chunk_text(processed_text,
                                       chunk_size=chunk_config["size"],
                                       overlap=chunk_config["overlap"])

                    # Gerar embeddings
                    doc_embeddings = model.encode(chunks)

                    # Avaliar qualidade
                    _, metrics = evaluate_embedding_quality(
                        model, test_queries, relevant_docs
                    )

                    # Calcular tempo e uso de memória
                    processing_time = time.time() - start_time
                    memory_usage = sum(sys.getsizeof(e) for e in doc_embeddings)

                    # Registrar resultados
                    results.append({
                        "model": model_name,
                        "chunk_size": chunk_config["size"],
                        "chunk_overlap": chunk_config["overlap"],
                        "preprocessing": strategy,
                        "precision": metrics["avg_precision"],
                        "recall": metrics["avg_recall"],
                        "f1": metrics["avg_f1"],
                        "mrr": metrics["avg_mrr"],
                        "processing_time_seconds": processing_time,
                        "memory_usage_bytes": memory_usage,
                        "embedding_dimensions": len(doc_embeddings[0])
                    })

        except Exception as e:
            print(f"Erro ao testar modelo {model_name}: {e}")

    # Converter para DataFrame para análise
    results_df = pd.DataFrame(results)

    # Salvar resultados
    results_df.to_csv("embedding_test_results.csv", index=False)

    return results_df
```

## 4. Visualização e Análise dos Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(results_df):
    """
    Analisa e visualiza os resultados dos testes.
    """
    # Configurar estilo
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)

    # 1. Comparação de modelos (F1 score)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='f1', data=results_df)
    plt.title('Comparação de Modelos por F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison_f1.png')

    # 2. Relação entre tamanho do chunk e precisão
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='chunk_size', y='precision', hue='model', data=results_df)
    plt.title('Impacto do Tamanho do Chunk na Precisão')
    plt.tight_layout()
    plt.savefig('chunk_size_precision.png')

    # 3. Trade-off entre velocidade e qualidade
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='processing_time_seconds', y='f1',
                   hue='model', size='chunk_size', data=results_df)
    plt.title('Trade-off entre Tempo de Processamento e Qualidade (F1)')
    plt.tight_layout()
    plt.savefig('time_quality_tradeoff.png')

    # 4. Encontrar a melhor combinação
    best_by_f1 = results_df.loc[results_df['f1'].idxmax()]
    best_by_mrr = results_df.loc[results_df['mrr'].idxmax()]
    best_efficiency = results_df.loc[results_df['f1'] / results_df['processing_time_seconds'].idxmax()]

    print("Melhor combinação por F1:", best_by_f1.to_dict())
    print("Melhor combinação por MRR:", best_by_mrr.to_dict())
    print("Melhor eficiência (F1/tempo):", best_efficiency.to_dict())

    return best_by_f1, best_by_mrr, best_efficiency
```

## 5. Validação com Consultas Reais

Após identificar as melhores combinações, é importante validá-las com consultas reais:

```python
def validate_with_real_queries(best_model, best_chunk_config, best_preprocessing):
    """
    Valida a melhor combinação com consultas reais de usuários.
    """
    # Carregar modelo
    model = SentenceTransformer(best_model)

    # Carregar e processar documento usando as melhores configurações
    with open(document_path, 'r') as f:
        full_text = f.read()

    if best_preprocessing == "basic":
        processed_text = basic_preprocess(full_text)
    elif best_preprocessing == "advanced":
        processed_text = advanced_preprocess(full_text)
    else:
        processed_text = semantic_preprocess(full_text)

    chunks = chunk_text(processed_text,
                       chunk_size=best_chunk_config["size"],
                       overlap=best_chunk_config["overlap"])

    # Gerar embeddings
    doc_embeddings = model.encode(chunks)

    # Criar banco de dados vetorial
    vector_db = create_vector_db(chunks, doc_embeddings)

    # Testar com consultas reais de usuários
    real_queries = [
        "Como implementar autenticação em dois fatores?",
        "Quais são os riscos de segurança em APIs REST?",
        # Mais consultas reais...
    ]

    # Avaliar respostas
    for query in real_queries:
        query_embedding = model.encode([query])[0]
        retrieved_docs = vector_db.search(query_embedding, top_k=3)

        print(f"\nConsulta: {query}")
        print("Documentos recuperados:")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. {doc[:100]}...")

        # Solicitar feedback humano
        relevance = input("Quão relevantes são esses resultados? (1-5): ")
        feedback = input("Algum comentário sobre os resultados? ")

        # Registrar feedback
        with open("validation_feedback.txt", "a") as f:
            f.write(f"Query: {query}\nRelevance: {relevance}\nFeedback: {feedback}\n\n")
```

## 6. Testes A/B com Sistemas RAG Completos

Para uma validação final, implemente sistemas RAG completos com as melhores combinações:

```python
def ab_test_rag_systems(combinations, test_queries):
    """
    Realiza testes A/B com sistemas RAG completos usando diferentes combinações.
    """
    results = {}

    for i, combo in enumerate(combinations):
        model_name = combo["model"]
        chunk_size = combo["chunk_size"]
        chunk_overlap = combo["chunk_overlap"]
        preprocessing = combo["preprocessing"]

        # Configurar sistema RAG com esta combinação
        rag_system = setup_rag_system(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preprocessing=preprocessing
        )

        # Testar com as consultas
        system_results = []
        for query in test_queries:
            # Obter resposta do sistema RAG
            response = rag_system.generate_response(query)

            # Registrar resposta
            system_results.append({
                "query": query,
                "response": response,
                # Aqui você pode adicionar métricas específicas de RAG
                # como relevância, precisão factual, etc.
            })

        results[f"system_{i+1}"] = {
            "config": combo,
            "responses": system_results
        }

    # Salvar resultados para análise humana
    with open("rag_ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
```

## 7. Pipeline Completo de Experimentação

Aqui está um exemplo de como executar o pipeline completo:

```python
def run_embedding_optimization_pipeline():
    """
    Executa o pipeline completo de otimização de embeddings.
    """
    print("1. Criando conjunto de teste...")
    test_queries, relevant_docs = create_test_set()

    print("2. Testando combinações de embeddings...")
    results_df = test_embedding_combinations(
        document_path="data/corpus.txt",
        test_queries=test_queries,
        relevant_docs=relevant_docs
    )

    print("3. Analisando resultados...")
    best_f1, best_mrr, best_efficiency = analyze_results(results_df)

    print("4. Validando com consultas reais...")
    validate_with_real_queries(
        best_model=best_f1["model"],
        best_chunk_config={
            "size": best_f1["chunk_size"],
            "overlap": best_f1["chunk_overlap"]
        },
        best_preprocessing=best_f1["preprocessing"]
    )

    print("5. Realizando testes A/B...")
    top_combinations = [
        best_f1.to_dict(),
        best_mrr.to_dict(),
        best_efficiency.to_dict()
    ]
    ab_results = ab_test_rag_systems(top_combinations, test_queries)

    print("6. Gerando relatório final...")
    generate_final_report(results_df, ab_results)

    return best_f1, ab_results
```

## 8. Refinamento Contínuo

O processo não termina com a seleção inicial. Implemente um sistema de feedback contínuo:

```python
def implement_continuous_feedback(rag_system):
    """
    Implementa um sistema de feedback contínuo para refinar o sistema RAG.
    """
    # Coletar feedback do usuário após cada resposta
    def collect_feedback(query, response):
        # Pode ser implementado via API, interface web, etc.
        feedback = {
            "query": query,
            "response": response,
            "relevance_score": user_rating,  # 1-5
            "comments": user_comments,
            "timestamp": datetime.now().isoformat()
        }

        # Armazenar feedback
        with open("user_feedback_log.jsonl", "a") as f:
            f.write(json.dumps(feedback) + "\n")

        return feedback

    # Analisar feedback periodicamente
    def analyze_feedback(feedback_file="user_feedback_log.jsonl", period_days=7):
        # Carregar feedback recente
        recent_feedback = []
        cutoff_date = datetime.now() - timedelta(days=period_days)

        with open(feedback_file, "r") as f:
            for line in f:
                feedback = json.loads(line)
                feedback_date = datetime.fromisoformat(feedback["timestamp"])
                if feedback_date >= cutoff_date:
                    recent_feedback.append(feedback)

        # Analisar tendências
        avg_relevance = sum(f["relevance_score"] for f in recent_feedback) / len(recent_feedback)
        problem_queries = [f for f in recent_feedback if f["relevance_score"] < 3]

        # Gerar relatório
        report = {
            "avg_relevance": avg_relevance,
            "total_feedback": len(recent_feedback),
            "low_relevance_percentage": len(problem_queries) / len(recent_feedback) * 100,
            "problem_areas": identify_problem_areas(problem_queries)
        }

        return report

    # Agendar análise periódica
    # Na prática, isso seria um job agendado
    schedule.every(7).days.do(lambda: analyze_feedback())
```

## 9. Exemplo de Relatório Final

```python
def generate_final_report(results_df, ab_results):
    """
    Gera um relatório final com recomendações.
    """
    report = {
        "summary": {
            "total_combinations_tested": len(results_df),
            "best_model": results_df.loc[results_df['f1'].idxmax()]["model"],
            "best_chunk_size": int(results_df.loc[results_df['f1'].idxmax()]["chunk_size"]),
            "best_preprocessing": results_df.loc[results_df['f1'].idxmax()]["preprocessing"]
        },
        "performance_metrics": {
            "best_f1": float(results_df['f1'].max()),
            "best_precision": float(results_df['precision'].max()),
            "best_recall": float(results_df['recall'].max()),
            "best_mrr": float(results_df['mrr'].max())
        },
        "efficiency_metrics": {
            "fastest_model": results_df.loc[results_df['processing_time_seconds'].idxmin()]["model"],
            "most_memory_efficient": results_df.loc[results_df['memory_usage_bytes'].idxmin()]["model"],
            "best_quality_speed_tradeoff": results_df.loc[(results_df['f1'] / results_df['processing_time_seconds']).idxmax()]["model"]
        },
        "recommendations": {
            "production_use": {
                "model": results_df.loc[results_df['f1'].idxmax()]["model"],
                "chunk_size": int(results_df.loc[results_df['f1'].idxmax()]["chunk_size"]),
                "chunk_overlap": int(results_df.loc[results_df['f1'].idxmax()]["chunk_overlap"]),
                "preprocessing": results_df.loc[results_df['f1'].idxmax()]["preprocessing"],
                "rationale": "Melhor equilíbrio entre qualidade e eficiência com base nos testes."
            },
            "resource_constrained": {
                "model": results_df.loc[(results_df['f1'] / results_df['memory_usage_bytes']).idxmax()]["model"],
                "rationale": "Melhor desempenho considerando restrições de recursos."
            }
        },
        "ab_test_summary": {
            "best_system": max(ab_results.items(), key=lambda x: sum(r.get("relevance_score", 0) for r in x[1]["responses"]))[0],
            "user_preference": "Sistema 1 preferido em 65% das consultas."  # Exemplo
        }
    }

    # Salvar como JSON
    with open("embedding_optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Gerar versão HTML para visualização
    html_report = f"""
    <html>
    <head>
        <title>Relatório de Otimização de Embeddings</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            .metric {{ margin-bottom: 20px; }}
            .recommendation {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Otimização de Embeddings para RAG</h1>

        <h2>Resumo</h2>
        <div class="metric">
            <p>Combinações testadas: {report["summary"]["total_combinations_tested"]}</p>
            <p>Melhor modelo: <strong>{report["summary"]["best_model"]}</strong></p>
            <p>Melhor tamanho de chunk: {report["summary"]["best_chunk_size"]}</p>
        </div>

        <h2>Métricas de Desempenho</h2>
        <div class="metric">
            <p>Melhor F1: {report["performance_metrics"]["best_f1"]:.4f}</p>
            <p>Melhor Precisão: {report["performance_metrics"]["best_precision"]:.4f}</p>
            <p>Melhor Recall: {report["performance_metrics"]["best_recall"]:.4f}</p>
        </div>

        <h2>Recomendações</h2>
        <div class="recommendation">
            <h3>Para Uso em Produção</h3>
            <p>Modelo: <strong>{report["recommendations"]["production_use"]["model"]}</strong></p>
            <p>Tamanho de Chunk: {report["recommendations"]["production_use"]["chunk_size"]}</p>
            <p>Sobreposição: {report["recommendations"]["production_use"]["chunk_overlap"]}</p>
            <p>Pré-processamento: {report["recommendations"]["production_use"]["preprocessing"]}</p>
            <p><em>{report["recommendations"]["production_use"]["rationale"]}</em></p>
        </div>

        <!-- Mais conteúdo do relatório -->
    </body>
    </html>
    """

    with open("embedding_report.html", "w") as f:
        f.write(html_report)

    print(f"Relatório gerado: embedding_optimization_report.json e embedding_report.html")
```

## Considerações Finais para a Abordagem Iterativa

1. **Comece com escopo limitado**: Inicie testando apenas 2-3 modelos com configurações básicas para estabelecer uma linha de base.

2. **Refine progressivamente**: Com base nos resultados iniciais, expanda os testes nas áreas mais promissoras.

3. **Automatize quando possível**: Crie scripts para automatizar a execução de testes em lote, especialmente para testes noturnos em grandes conjuntos de dados.

4. **Considere o contexto do domínio**: Para domínios específicos, dê mais peso a modelos especializados ou considere fine-tuning.

5. **Mantenha registros detalhados**: Documente todas as configurações e resultados para referência futura e reprodutibilidade.

6. **Feedback dos usuários finais**: Incorpore feedback qualitativo dos usuários finais do sistema RAG para complementar as métricas quantitativas.
