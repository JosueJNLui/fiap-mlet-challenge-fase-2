# Model Card: recomendador BPR (NeuMF híbrido)

Model Card do modelo neural do projeto, seguindo a exigência da Etapa 4. Cobre uso
pretendido, dados, hiperparâmetros, métricas, comparação com baselines e limitações.

## Visão geral

- **Nome**: `MovieLens_BPR_Reco` (registrado no MLflow Registry, servido pela API).
- **Tipo**: rede neural de ranking treinada com perda BPR (Bayesian Personalized Ranking)
  em PyTorch. Definição em `src/recsys/models/bpr.py`.
- **Arquitetura — NeuMF híbrido** (inspirado em He et al. 2017, *Neural Collaborative
  Filtering*):
  - **GMF (Generalized MF):** embeddings usuário/item (dim 128) combinados por produto
    elemento a elemento — captura interações lineares.
  - **MLP:** embeddings usuário/item separados (dim 128) concatenados e passados por camadas
    densas `[256, 128, 64]` com ReLU + Dropout — captura interações não lineares.
  - **Termos de viés:** embeddings de viés de usuário e de item (`user_bias`, `item_bias`).
  - **Fusão:** concatena as saídas GMF e MLP → camada densa final → `score(u, i)`, somado ao
    viés. Usado para ranquear o catálogo de itens não vistos e retornar o top-10.
- **Por que viés + pop-sampling:** um BPR-MF puro (só produto interno, sem viés, negativos
  uniformes) satura em ~0,09 NDCG@10. Os **termos de viés** deram o principal ganho de
  ranking, e o **negative sampling ponderado por popularidade** (ver abaixo) força o modelo
  a distinguir itens populares de itens relevantes, elevando a diversidade sem perder ranking.
- **Entrada**: apenas IDs (índices contíguos de usuário e item). Não usa conteúdo ou
  metadados (gênero, tags, genome scores).
- **Saída**: score de relevância por par (usuário, item). Uma calibração z-score opcional
  (`predict`) mapeia o score para a escala de notas [0,5; 5,0] — cosmética, **não** um
  preditor de nota confiável (ver métricas de regressão).

## Uso pretendido

- **Caso de uso**: recomendação top-10 de filmes para um usuário conhecido, servida pela
  API FastAPI (`GET /recommend?user_id=...`).
- **Usuários-alvo**: avaliação acadêmica do challenge e demonstração de um pipeline de
  recomendação de ponta a ponta.
- **Fora de escopo**: usuários ou itens não vistos no treino (sem cold-start); predição
  calibrada de nota (o BPR otimiza ranking top-K, não erro de nota).

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
- **Definição de positivo**: interações com nota maior ou igual a **3,5** (o threshold de
  like, usado tanto para os positivos do BPR quanto para o alvo do ranking).

## Hiperparâmetros efetivos

Fonte única em `configs/config.yaml` (chave `models.bpr`), lida tanto pelo pipeline DVC
(`load_settings()`) quanto pelo notebook — os valores não divergem. Defaults no código são
apenas fallback.

| Hiperparâmetro | Valor |
| -------------- | ----- |
| `mf_emb_dim` / `mlp_emb_dim` | 128 / 128 |
| `mlp_layers` | [256, 128, 64] |
| `dropout` | 0,2 |
| `lr` | 0,001 |
| `weight_decay` | 1e-4 |
| `n_neg` (negativos por positivo) | 10 |
| `batch_size` | 2048 |
| `epochs` (máx.) | 100 |
| `patience` (early stopping) | 15 |
| `grad_clip_norm` | 1,0 |
| `pop_alpha` (peso de popularidade) | 0,75 |
| `val_users` | 1000 |
| `val_item_batch` | 4096 |
| `like_threshold` | 3,5 |
| `seed` | 42 |

- **Loss**: BPR pairwise, `-logsigmoid(score_pos - score_neg).mean()`.
- **Negative sampling**: `PopularityWeightedNegativeSampler` (`alpha=0.75`) — itens populares
  aparecem como negativos com mais frequência, filtrando os já vistos pelo usuário.
- **Otimizador**: AdamW (weight decay desacoplado) + gradient clipping (norma 1,0), com
  **taxa de aprendizado constante** — o early stopping faz o controle (o val NDCG estabiliza
  cedo, então um scheduler com warm restarts foi removido por não alterar as métricas).
- **Early stopping**: monitora o NDCG@10 de catálogo completo em um subconjunto de
  validação (não a loss). O melhor checkpoint por NDCG é restaurado ao final.
- **Device**: seleção automática mps > cuda > cpu no treino; forçado para CPU no serving.

## Métricas e comparação com baselines

Métricas da execução do pipeline (`dvc repro` → `comparison.csv`/`metrics.json`), com a
configuração única de `configs/config.yaml`. **Esses arquivos são a fonte canônica dos
números** — `notebooks/models.ipynb` roda os mesmos modelos pela mesma config, mas é uma
demonstração da orquestração: como o treino do BPR é não-determinístico, seus valores
oscilam dentro do IC e não substituem o `comparison.csv`.

Protocolo de catálogo completo: para cada usuário, ranqueia todos os itens não vistos;
k = 10; positivos com nota ≥ 3,5. São 11 colunas em três grupos: regressão (RMSE, MAE,
MSE, R2), ranking (Precision@10, Recall@10, NDCG@10 e o half-width do IC 95%) e
diversidade (coverage, novelty, Gini).

**Avaliação justa:** o ranking usa o **mesmo conjunto fixo de 500 usuários** para todos os
modelos (`get_ranking_users`); o NDCG@10 é reportado com **IC 95%** (± 1,96·EP).

### Baselines (scikit-learn / NumPy)

- **GlobalMean**: prediz a média global das notas.
- **Bias**: `mu + b_user + b_item` regularizado (reg = 10,0).
- **Popularity**: ranqueia por `log1p(contagem de interações)`.
- **SVD**: `TruncatedSVD` do scikit-learn (`n_components=64`, seed 42) sobre resíduos
  centrados na média.

### Resultados

| Modelo | RMSE | MAE | R2 | P@10 | R@10 | NDCG@10 | Coverage | Novelty | Gini |
| ------ | ---- | --- | -- | ---- | ---- | ------- | -------- | ------- | ---- |
| GlobalMean | 1,047 | 0,829 | -0,005 | 0,004 | 0,003 | 0,009 | 0,021 | 17,23 | 0,995 |
| Bias | **0,870** | **0,665** | **0,306** | 0,025 | 0,022 | 0,037 | 0,003 | 10,39 | 0,999 |
| SVD | 0,992 | 0,778 | 0,099 | **0,097** | 0,079 | **0,122** | 0,032 | 9,50 | 0,987 |
| Popularity | 1,738 | 1,412 | -1,767 | 0,053 | 0,041 | 0,069 | 0,006 | 8,41 | 0,998 |
| **BPR (neural)** | 1,342 | 1,086 | -0,650 | 0,087 | **0,088** | 0,114 | **0,092** | 10,46 | **0,957** |

- **Melhor RMSE**: Bias (0,870). **Melhor NDCG@10 (valor nominal)**: SVD (0,122). Como o
  ranking empata estatisticamente (próximo item), o desempate é a **diversidade**, e o
  modelo promovido a `production` é a rede neural (BPR) — que é também o modelo servido
  pela API.
- **Ranking: empate estatístico entre SVD e BPR.** SVD 0,122 ± 0,015 vs BPR 0,114 ± 0,014;
  a diferença (0,008) cai **bem dentro do IC 95% da diferença** (± 0,021) — não é significativa
  com 500 usuários. A rede NeuMF empata com o SVD no ranking, sem superá-lo. (O NDCG@10 do
  BPR varia entre execuções por não-determinismo de GPU/MPS, dentro do próprio IC — mais uma
  evidência do empate.)
- **Diversidade: o diferencial do BPR.** Coverage 9,2% vs 3,2% do SVD (2,9×) — a maior
  cobertura de catálogo entre todos os modelos —, Gini 0,957 vs 0,987 (menos concentrado) e
  recomendações mais de nicho (novelty 10,46 vs 9,50 do SVD; a `Popularity` é a menos
  novel, 8,41, como esperado). Efeito direto do pop-sampling.
- **Por que o BPR tem RMSE/MAE/R2 ruins**: é um modelo de ranking. A calibração z-score
  coloca o score na escala [0,5; 5,0], mas os valores não são notas confiáveis (R2 negativo,
  pior que a média global). O valor do BPR está no ranking e na diversidade, não na regressão.

## Limitações e vieses

- **Sem cold-start**: usuários ou itens fora do índice de treino não podem ser servidos;
  `user_id` desconhecido retorna 404. A avaliação nunca testa cold-start real (todo
  usuário de teste está no treino).
- **Sem uso de conteúdo**: apesar do MovieLens trazer gêneros, tags e genome scores, os
  modelos usam só IDs de interação. Isso limita cold-start e interpretabilidade.
- **Viés de popularidade e concentração**: Gini alto (0,96–0,99) e coverage baixa indicam
  que as recomendações se concentram numa fração dos 13.088 itens. O BPR mitiga isso
  (coverage 9,2%, Gini 0,957), mas não elimina.
- **Viés de amostragem**: só 20.000 usuários (de aproximadamente 138 mil) e apenas
  usuários e itens com pelo menos 20 interações. O modelo é treinado sobre usuários ativos
  e itens populares, não sobre a distribuição completa.
- **Threshold de like**: positivos são notas ≥ 3,5; a faixa 0,5 a 3,0 é tratada como
  não-positiva, descartando preferência graduada.
- **Avaliação subamostrada**: ranking avaliado em 500 usuários (IC 95% de ± ~0,015 no
  NDCG@10) e o NDCG de early stopping em até 1000 usuários, o que adiciona ruído de amostra
  às métricas — motivo pelo qual o ranking SVD vs BPR fica em empate estatístico.

## Reprodutibilidade

- Seed 42 global (torch, RNG NumPy, TruncatedSVD, amostragem de usuários).
- Configuração única em `configs/config.yaml`, lida por notebook e pipeline.
- Dados versionados com DVC; dependências fixadas em `uv.lock`.
- Pipeline reproduzível via `dvc repro` (ou `make pipeline`).

## Registro no MLflow

Registrado como `MovieLens_BPR_Reco` — o modelo servido pela API e promovido a
`production`. Cada versão recebe o alias `staging`; o alias `production` só migra quando o
NDCG@10 supera o da produção atual do mesmo modelo. Os baselines também são registrados
(`MovieLens_<Label>_Reco`: `GlobalMean`, `Bias`, `SVD`, `Popularity`) para comparação, mas
não recebem `production`. **Só o pipeline (`train`/`evaluate`) escreve no Registry** — o
notebook apenas loga runs, para não criar versões com outro formato de artefato.

O modelo servido é configurável por `SERVED_MODEL` no `.env`: apontar para
`MovieLens_SVD_Reco` permite A/B testar o melhor baseline sem deploy de código — exige que a
versão alvo já tenha o alias `production` (só a rede neural tem, por padrão). Detalhes do
fluxo de tracking e promoção estão em [ARCHITECTURE.md](ARCHITECTURE.md).

### Modelo em produção vs. última execução

A tabela de **Resultados** acima reporta a última execução do pipeline (`comparison.csv`,
regenerado por `dvc repro`). Ela **não** é necessariamente a versão em `production`: o
treino do BPR é não-determinístico (kernels de GPU/MPS), então cada `dvc repro` produz uma
versão com métricas ligeiramente diferentes, e o gate de promoção só migra o alias se a nova
versão superar o NDCG@10 da produção atual.

Estado atual do Registry:

| Alias | Versão | NDCG@10 | Coverage |
| ----- | ------ | ------- | -------- |
| `production` (servido pela API) | v11 | **0,1165** | 5,5% |
| `staging` (última execução, tabela acima) | v13 | 0,1141 | 9,2% |

A v13 ficou 0,0024 abaixo em NDCG@10 — dentro do próprio IC (± 0,014), ou seja, ruído — e o
gate barrou a promoção, mantendo a v11. É o comportamento pretendido: o alias `production`
não regride por variação de execução. Note que a v13 é melhor em diversidade (coverage
9,2% vs 5,5%); promovê-la exigiria um critério multi-métrica, não um override manual.

**Fallback local**: se o Registry estiver inacessível, a API cai para `models/bpr.pkl`
(ver [ARCHITECTURE.md](ARCHITECTURE.md)). Esse pickle é o da **última execução local**, não
necessariamente a versão em `production` — o fallback garante disponibilidade, não
reprodutibilidade das recomendações.
