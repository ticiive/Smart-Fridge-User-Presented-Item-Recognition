# Customizações — O que foi desenvolvido além do uso padrão das bibliotecas

Este documento descreve as soluções não triviais implementadas no projeto, indicando o problema que motivou cada uma e onde o código correspondente está.

---

## 1. Segmentação por passagem

**Problema observado:** Rodar o pipeline de identificação quadro a quadro e registrar um evento a cada inferência positiva gerava duplicatas (o mesmo produto identificado várias vezes enquanto atravessava o campo de visão) e tornava impossível distinguir a entrada de um produto novo da permanência de um já reconhecido.

**O que foi feito:** O sistema agrupa os quadros em que um mesmo produto é detectado em uma unidade chamada "passagem". A passagem abre na primeira detecção válida e fecha quando passam 11 quadros consecutivos sem detecção (`PASSAGEM_TIMEOUT`), ou quando o produto permanece visível por mais de 60 quadros (`PASSAGEM_MAX_FRAMES`) para evitar acúmulo indefinido. Uma única decisão de reconhecimento é tomada ao fim de cada passagem. Passagens com menos de 5 quadros são descartadas para eliminar detecções espúrias. Se a passagem for encerrada por excesso de frames e ainda houver detecção no quadro corrente, uma nova passagem abre imediatamente com aquele quadro.

**Onde está:** `demo_ao_vivo.py`, variáveis `passagem_ativa`, `passagem_frames_sem_det`, `passagem_frames_total` e o bloco de lógica de passagem a partir da linha `if not direcao_ativa:` na função `main()`. Constantes: `PASSAGEM_TIMEOUT`, `PASSAGEM_MAX_FRAMES`, `PASSAGEM_MIN_FRAMES`.

---

## 2. Seleção de até 6 recortes por nitidez

**Problema observado:** Salvar todos os quadros de uma passagem desperdiçava disco e aumentava o tempo de OCR. Selecionar apenas o primeiro ou o último recorte não garantia que o rótulo estivesse nítido e legível.

**O que foi feito:** Durante cada passagem, todos os quadros com detecção válida têm sua nitidez calculada pela variância do Laplaciano (`cv2.Laplacian(gray, cv2.CV_64F).var()`). Ao encerrar a passagem, os quadros são ordenados por nitidez decrescente e os 6 mais nítidos são salvos em `capturas/passagem_NNN/`. O OCR e o VLM recebem esses recortes nessa ordem, do mais ao menos nítido.

**Onde está:** `demo_ao_vivo.py`, constante `OCR_MAX_RECORTES = 6`. O cálculo de nitidez está no bloco de acumulação de `passagem_frames_dados` e a seleção final em `_sorted_fds = sorted(passagem_frames_dados, key=lambda fd: fd["sharp"], reverse=True)[:OCR_MAX_RECORTES]`.

---

## 3. Máscara de novidade contra fundo de referência

**Problema observado:** O YOLO-World detecta qualquer embalagem no campo de visão, incluindo itens já presentes na geladeira antes da sessão começar. Sem filtro, um produto estático acionaria o pipeline a cada quadro.

**O que foi feito:** Ao iniciar a demo (automaticamente no 15.º quadro, ou manualmente com a tecla `b`), o sistema captura um quadro de referência da cena vazia. A cada detecção do YOLO, uma máscara de diferença absoluta (com threshold de Otsu) é calculada entre o quadro atual e o fundo. Se menos de 15% dos pixels dentro da caixa detectada são "novos" em relação ao fundo (`NOVIDADE_MIN = 0.15`), a detecção é descartada. Quando há várias detecções concorrentes, a que apresenta maior fração de pixels novos tem prioridade.

**Onde está:** `demo_ao_vivo.py`, função `_mascara_novidade()` e uso dentro de `detectar_produto_yolo()`. Captura do fundo: função `capturar_fundo()`.

---

## 4. Descarte de pessoa, mão, braço, rosto e roupa

**Problema observado:** A câmera no topo da geladeira enquadra o braço e a mão da pessoa ao pegar ou devolver um produto. Sem filtragem, partes do corpo eram detectadas como embalagens e enviadas ao pipeline de identificação.

**O que foi feito:** O vocabulário do YOLO-World inclui categorias humanas explícitas (`"person"`, `"human face"`, `"hand"`, `"arm"`, `"clothing"`, `"shirt sleeve"`, `"fabric"`, `"human skin"`). As caixas que recaem nessas categorias são separadas das embalagens e usadas como filtro: qualquer caixa de embalagem cuja sobreposição de IoU com uma caixa humana supere 0,50 é descartada. Caixas com área acima de 25% do quadro também são descartadas para eliminar troncos ou pessoas inteiras que passem na frente da câmera.

**Onde está:** `demo_ao_vivo.py`, constantes `YOLO_DESCARTAR`, `YOLO_IOU_PESSOA = 0.50` e `YOLO_MAX_AREA_FRAC = 0.25`. Lógica de filtragem na função `detectar_produto_yolo()`, função auxiliar `_iou()`.

---

## 5. Índice de palavras-chave com remoção automática de palavras compartilhadas

**Problema observado:** Palavras genéricas como "zero", "açúcar", "sem" aparecem em múltiplos produtos (ex.: Coca-Cola Zero e Monster Ultra). Incluí-las no índice provocava falsos positivos: um produto era identificado pela presença de uma palavra que não o distinguia.

**O que foi feito:** `construir_indice_palavras()` lê `catalogo_palavras.yaml` e constrói um dicionário `{palavra_normalizada: nome_produto}` apenas para palavras que aparecem em exatamente um produto. Palavras em dois ou mais produtos são descartadas automaticamente e impressas no terminal ao carregar. A função `match_palavras()` adiciona uma guarda extra: só decide se o produto vencedor obteve pelo menos uma palavra longa (≥ 5 caracteres) ou pelo menos dois votos distintos — isso evita que palavras curtas e comuns causem identificações espúrias mesmo que tenham passado pelo filtro de unicidade.

**Onde está:** `ocr_rotulo.py`, funções `construir_indice_palavras()` e `match_palavras()`.

---

## 6. OCR com pré-processamento (2×, CLAHE, unsharp) e teste das quatro rotações

**Problema observado:** Recortes pequenos capturados por câmera de topo têm baixa resolução e contraste irregular por causa da iluminação interna da geladeira. O produto nem sempre entra com o rótulo na orientação correta.

**O que foi feito:** Antes de enviar para o Apple Vision, cada recorte passa por `_preprocessar_para_ocr()`: ampliação 2× com `INTER_CUBIC`, conversão para escala de cinza, equalização adaptativa de histograma com CLAHE (clipLimit 2,0, grade 8×8) e unsharp mask com desvio padrão 3. Em seguida, `ocr_imagem()` executa o OCR nas quatro rotações do recorte (0°, 90°, 180°, 270°) e escolhe a orientação que produziu o maior número de caracteres alfanuméricos — isso resolve rótulos girados sem exigir que o usuário apresente o produto em orientação específica.

**Onde está:** `ocr_rotulo.py`, funções `_preprocessar_para_ocr()` e `ocr_imagem()`.

---

## 7. Cascata de decisão com limiares e validação cruzada da resposta do VLM

**Problema observado:** Usar apenas um método de identificação gerava erros sistemáticos: o CLIP sozinho sofria de efeito de atrator (recortes escuros convergiam para o mesmo produto); o OCR sozinho falhava quando o rótulo estava oculto ou girado; o VLM sozinho era lento e às vezes respondia um produto sem conseguir ler o rótulo.

**O que foi feito:** Os três métodos operam em cascata com limiares distintos:

1. O OCR tem prioridade se encontrar uma correspondência no índice de palavras exclusivas.
2. O CLIP decide se o candidato mais provável supera similaridade 0,66 com margem acima de 0,05 em relação ao segundo colocado (`CLIP_LIMIAR_ALTO`, `CLIP_MARGEM_ALTA`). Similaridade abaixo de 0,55 descarta a passagem; entre 0,55 e 0,66 encaminha ao VLM.
3. O VLM é acionado quando nem o OCR nem o CLIP conseguem decidir, ou quando ambos têm opinião mas discordam (cada um apontando para um produto diferente — neste caso o VLM atua como árbitro).

A validação cruzada da resposta do VLM consiste em dois passos: (a) verificar se o identificador no campo `PRODUTO:` pertence ao catálogo; (b) se `PRODUTO:` retornou "nenhum" mas o campo `TEXTO:` tem conteúdo, casar o texto lido com o índice de palavras-chave como fallback.

**Onde está:** `demo_ao_vivo.py`, bloco de cascata a partir do comentário `# --- Cascade: cascata de decisao ---` na função `main()`. Constantes: `CLIP_LIMIAR_ALTO`, `CLIP_LIMIAR_MEDIO`, `CLIP_MARGEM_ALTA`. Validação cruzada do VLM na função `_chamar_ollama_vlm()` e no bloco de processamento das respostas `vlm_fila_respostas`.

---

## 8. Fila assíncrona do VLM com uma resposta por passagem e no máximo uma chamada

**Problema observado:** O VLM demora dezenas de segundos por imagem. Bloquear o loop principal para aguardar a resposta travava a interface e impedia que novas passagens fossem processadas enquanto o modelo pensava.

**O que foi feito:** O VLM é atendido por um único worker em thread daemon (`_worker_vlm()`), que consome pedidos de `vlm_fila_pedidos` e deposita respostas em `vlm_fila_respostas`. O loop principal verifica a fila de respostas em modo não bloqueante (`get_nowait()`) a cada quadro. Cada passagem pode originar no máximo um pedido ativo; se o primeiro recorte enviado não produzir identificação, o worker tenta o segundo recorte da mesma passagem antes de desistir. Passagens já decididas por outro método têm a resposta do VLM ignorada (`passagem_estado[num] == "decidida"`). Ao pressionar `q` com pedidos pendentes, o sistema aguarda até 120 segundos pelas respostas antes de encerrar.

**Onde está:** `demo_ao_vivo.py`, função `_worker_vlm()`, `vlm_fila_pedidos` e `vlm_fila_respostas` (instâncias de `queue.Queue`), e lógica de timeout no tratamento da tecla `q` na função `main()`.

---

## 9. Relatórios em texto e processamento offline das capturas para avaliação

**Problema observado:** Avaliar a acurácia do sistema exigia rever manualmente os recortes salvos e comparar com as decisões tomadas em tempo real. Repetir o experimento em modo "frio" (apenas catálogo original) ou "enriquecido" (com referências auto-adicionadas) precisava de uma forma de reprocessar os mesmos recortes.

**O que foi feito:** Ao encerrar a demo, um relatório em texto é gerado em `relatorios/` com o detalhamento de cada passagem: duração em quadros, número de inferências, resultado do OCR (texto bruto e produto casado), resultado do CLIP (produto vencedor e similaridade), resultado do VLM (produto e tempo de resposta) e qual método tomou a decisão final. O script `processar_capturas.py` reprocessa os recortes salvos em `capturas/` e `logs_demo/` de forma independente da demo, produzindo um CSV em `resultados/` com três colunas de opinião (CLIP, OCR, VLM) e uma coluna `verdadeiro` para preenchimento manual. Para zerar o índice enriquecido e repetir o experimento, basta esvaziar `catalogo_enriquecido/` e rodar `processar_capturas.py` sem a flag `--com-enriquecido`.

**Onde está:** `demo_ao_vivo.py`, bloco de escrita do relatório no final de `main()` (variáveis `relatorio_passagens`, `relatorio_descartadas`). `processar_capturas.py`, funções `coletar_imagens()`, `inferir_clip()`, `inferir_vlm()` e `main()`.
