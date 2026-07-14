# Model Card: recomendador BPR-MF

Model Card do modelo neural do projeto, seguindo a exigência da Etapa 4. Cobre uso
pretendido, dados, hiperparâmetros, métricas, comparação com baselines e limitações.

## Visão geral

- **Nome**: `MovieLens_BPR_Reco` (registrado no MLflow Registry).
- **Tipo**: fatoração de matrizes com BPR (Bayesian Personalized Ranking), rede neural
  em PyTorch. Definição em `src/recsys/models/bpr.py`.
- **Arquitetura**: duas tabelas de embedding (`nn.Embedding`), uma para usuários e uma
  para itens, ambas com dimensão 128. O score de um par (usuário, item) é o dot product
  dos embeddings: `score(u, i) = <p_u, q_i>`. Não há termos de viés.
- **Por que sem viés**: em BPR pairwise, a diferença de scores `pos - neg` cancela a
  média global e o viés de usuário; um viés de item absorveria a popularidade global e
  colapsaria o modelo em "recomende o popular". Assim o ranking depende só da
  personalização `p_u . q_i`.
- **Entrada**: apenas IDs (índices contíguos de usuário e item). Não usa conteúdo ou
  metadados (gênero, tags, genome scores).
- **Saída**: score de relevância por par (usuário, item), usado para ranquear o catálogo
  de itens ainda não vistos e retornar o top-10.

## Uso pretendido

- **Caso de uso**: recomendação top-10 de filmes para um usuário conhecido, servida pela
  API FastAPI (`GET /recommend?user_id=...`).
- **Usuários-alvo**: avaliação acadêmica do challenge e demonstração de um pipeline de
  recomendação de ponta a ponta.
- **Fora de escopo**: usuários ou itens não vistos no treino (sem cold-start), predição
  calibrada de nota (o BPR produz scores de ranking, não notas na escala 0,5 a 5,0).

## Dados de treino

- **Fonte**: MovieLens 20M (`data/raw/rating.csv`), versionado por DVC (remote no DagsHub).
- **Pré-processamento** (stage `preprocess`): filtro de atividade (usuários e itens com
  pelo menos 20 interações), amostragem de 20.000 usuários (seed 42) e reindexação para
  índices contíguos.
- **Split** (stage `feature_eng`): `TemporalLeaveLastFraction(0.2)`, os 20% de interações
  mais recentes de cada usuário vão para teste, sem vazamento de futuro. Todo usuário de
  teste também aparece no treino (necessário porque o modelo usa embeddings aprendidos).
- **Resultado** (`data/processed/meta.json`): 20.000 usuários, 13.088 itens, 2.857.387
  interações.
- **Definição de positivo**: interações com nota maior ou igual a 4,0 (o threshold de like).

## Hiperparâmetros efetivos

Fonte única em `configs/config.yaml` (chave `models.bpr`); esses são os valores que
rodaram (os defaults no código são apenas fallback).

| Hiperparâmetro | Valor |
| -------------- | ----- |
| `emb_dim` | 128 |
| `lr` | 0,005 |
| `weight_decay` | 1e-5 |
| `n_neg` (negativos por positivo) | 10 |
| `batch_size` | 2048 |
| `epochs` | 40 |
| `patience` (early stopping) | 6 |
| `val_users` | 300 |
| `like_threshold` | 4,0 |
| `seed` | 42 |

- **Loss**: BPR pairwise, `-logsigmoid(score_pos - score_neg).mean()`.
- **Negative sampling**: `UniformNegativeSampler`, uniforme sobre o catálogo, filtrando
  itens já vistos pelo usuário.
- **Otimizador**: Adam com weight decay (L2) nos embeddings.
- **Early stopping**: monitora o NDCG@10 de catálogo completo em um subconjunto de
  validação (não a loss). O melhor checkpoint por NDCG é restaurado ao final.
- **Device**: seleção automática mps > cuda > cpu no treino; forçado para CPU no serving.

## Métricas e comparação com baselines

Todas as métricas vêm de `metrics.json`/`comparison.csv`, avaliadas no conjunto de teste
com protocolo de catálogo completo (para cada usuário, ranqueia todos os itens não vistos;
k = 10; positivos com nota maior ou igual a 4,0). São 10 métricas em três grupos:
regressão (RMSE, MAE, MSE, R2), ranking (Precision@10, Recall@10, NDCG@10) e diversidade
(coverage, novelty, Gini).

### Baselines (scikit-learn / NumPy)

- **GlobalMean**: prediz a média global das notas.
- **Bias**: `mu + b_user + b_item` regularizado (reg = 10,0).
- **Popularity**: ranqueia por `log1p(contagem de interações)`.
- **SVD**: `TruncatedSVD` do scikit-learn (`n_components=50`, seed 42) sobre resíduos
  centrados na média.

### Resultados

| Modelo | RMSE | MAE | R2 | P@10 | R@10 | NDCG@10 | Coverage | Novelty | Gini |
| ------ | ---- | --- | -- | ---- | ---- | ------- | -------- | ------- | ---- |
| GlobalMean | 1,047 | 0,829 | -0,005 | 0,004 | 0,003 | 0,009 | 0,023 | 2,73 | 0,995 |
| Bias | **0,870** | **0,665** | **0,306** | 0,027 | 0,026 | 0,039 | 0,003 | 7,44 | 0,999 |
| SVD | 0,991 | 0,778 | 0,101 | **0,080** | **0,080** | **0,102** | **0,029** | 8,13 | 0,988 |
| Popularity | 1,738 | 1,412 | -1,767 | 0,050 | 0,062 | 0,074 | 0,006 | 8,82 | 0,998 |
| **BPR (neural)** | 2,246 | 1,949 | -3,623 | 0,062 | 0,073 | 0,090 | 0,011 | 8,74 | 0,998 |

- **Melhor RMSE**: Bias (0,870). **Melhor NDCG@10**: SVD (0,102), com o BPR em segundo
  (0,090). A promoção a `production` no Registry usa o critério de NDCG@10.
- **Por que o BPR tem RMSE/MAE/R2 ruins**: ele é um modelo de ranking puro. Seus scores
  são dot products não calibrados, não notas na escala 0,5 a 5,0, então as métricas de
  regressão não fazem sentido para ele. O valor do BPR está no ranking (Precision, Recall,
  NDCG), onde supera todos os baselines exceto o SVD.

## Limitações e vieses

- **Sem cold-start**: usuários ou itens fora do índice de treino não podem ser servidos;
  `user_id` desconhecido retorna 404. A avaliação nunca testa cold-start real (todo
  usuário de teste está no treino).
- **Sem uso de conteúdo**: apesar do MovieLens trazer gêneros, tags e genome scores, os
  modelos usam só IDs de interação. Isso limita cold-start e interpretabilidade.
- **Viés de popularidade e concentração**: Gini em torno de 0,99 e coverage baixa
  (BPR aproximadamente 1,1%, SVD aproximadamente 2,9%) indicam que as recomendações se
  concentram em uma fração pequena dos 13.088 itens.
- **Viés de amostragem**: só 20.000 usuários (de aproximadamente 138 mil) e apenas
  usuários e itens com pelo menos 20 interações. O modelo é treinado sobre usuários ativos
  e itens populares, não sobre a distribuição completa.
- **Threshold de like**: positivos são notas maiores ou iguais a 4,0; a faixa 0,5 a 3,5 é
  tratada como não-positiva, descartando preferência graduada.
- **Avaliação subamostrada**: ranking avaliado em até 500 usuários e o NDCG de early
  stopping em até 300 usuários, o que adiciona ruído de amostra às métricas reportadas.

## Reprodutibilidade

- Seed 42 global (torch, RNG NumPy, TruncatedSVD, amostragem de usuários).
- Dados versionados com DVC; dependências fixadas em `uv.lock`.
- Pipeline reproduzível via `dvc repro` (ou `make pipeline`).

## Registro no MLflow

Registrado como `MovieLens_BPR_Reco`. Cada versão recebe o alias `staging`; o alias
`production` só migra quando o NDCG@10 supera o da produção atual. Detalhes do fluxo de
tracking e promoção estão em [ARCHITECTURE.md](ARCHITECTURE.md).
