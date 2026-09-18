# Contexto do projeto

Sistema de reconhecimento de produtos de geladeira para um trabalho acadêmico
de Engenharia de Computação (IBMEC). Existe um artigo em andamento, e as
decisões técnicas precisam bater com a premissa do artigo. Apresentação com
demonstração ao vivo na próxima semana.

## Ambiente

- MacBook Air 13", macOS 26.3.1, provavelmente Apple Silicon.
- Python do sistema é 3.9.6. Essa versão é antiga e alguns pacotes de visão já
  não publicam builds para ela. Verificar compatibilidade antes de instalar; se
  necessário, instalar Python 3.11 ou 3.12 pelo Homebrew.
- Nunca instalar no Python do sistema. Criar ambiente virtual antes de qualquer
  instalação.
- Se for Apple Silicon, verificar se o PyTorch consegue usar MPS, que acelera
  bastante o cálculo dos embeddings. Se não der, CPU resolve.

## Premissa (não pode ser alterada)

- O catálogo de referência tem UMA imagem por produto, como num catálogo de
  e-commerce. Não existe dataset com dezenas de fotos por item.
- Nenhum modelo é treinado aqui. Modelos pré-treinados são usados como estão.
- A câmera fica FIXA no topo da geladeira. O produto aparece pequeno no quadro,
  em movimento e em ângulo arbitrário.
- A iluminação é a lâmpada interna da geladeira, que acende quando a porta
  abre. A condição de luz é constante, mas é um ponto de luz no alto, então há
  risco de reflexo em embalagem brilhante e de sombra do braço da pessoa.
- Objetivo: a pessoa pega e guarda os produtos normalmente, sem apresentar o
  item à câmera nem mirar o rótulo. O professor já confirmou que apresentar o
  item também é aceitável para esta entrega, então isso serve como plano B, mas
  o alvo é funcionar sem exigir esforço do usuário.

## Pipeline (os três estágios rodam em sequência, num programa só)

1. Detecção. Usar detector de vocabulário aberto, que aceita as categorias como
   texto na hora da execução e não exige treino: YOLO-World ou YOLOE, ambos
   disponíveis na biblioteca Ultralytics. As categorias são descrições
   genéricas de embalagem ("pote de iogurte", "caixa de leite", "garrafa",
   "pacote"), não marcas. Se a detecção falhar em parte dos produtos, testar
   duas alternativas: aproveitar as caixas propostas ignorando a classe
   atribuída, ou subtração de fundo comparando o quadro com a geladeira vazia.

2. Identificação. Calcular o embedding do recorte com um modelo pré-treinado
   (CLIP) e comparar com o catálogo por similaridade de cosseno. Como o produto
   atravessa o campo de visão, capturar uma sequência de quadros e decidir pelo
   de maior similaridade. Abaixo de um limiar configurável, responder
   "não reconhecido" em vez de chutar o mais parecido.

3. Leitura de rótulo (OCR). Rodar OCR no recorte e comparar o texto obtido com
   os NOMES dos produtos do catálogo, por similaridade de string. Combinar essa
   pontuação com a do embedding. Quando o OCR não devolver texto legível, o
   sistema decide só pelo embedding. O OCR é voto extra, nunca etapa que trava
   o pipeline.

## Auto-enriquecimento do catálogo

Quando o reconhecimento for muito confiante, guardar o embedding daquele
recorte como referência ADICIONAL do produto, para o catálogo ganhar imagens
nas condições reais de uso. Quatro travas obrigatórias:

1. limiar de adição bem mais alto que o limiar de reconhecimento;
2. margem mínima entre o primeiro e o segundo colocado, para não adicionar
   quando dois produtos estão quase empatados;
3. número máximo de referências por produto;
4. a imagem original do catálogo é âncora permanente, nunca removida nem
   substituída.

Todos esses parâmetros num arquivo de configuração, não espalhados no código.

## Saída da demonstração

Além do nome do produto na tela, manter um painel de lista de compras ao lado
da imagem: quando um produto sai da geladeira e não volta, ele entra na lista.
Cada item da lista vira um link que abre a busca daquele produto num site de
mercado, montado a partir do nome. Sem API de delivery, sem cadastro, sem
automação de compra.

## Avaliação

Dois modos: frio (só as imagens de catálogo) e enriquecido (com as referências
auto-adicionadas). Precisa ser possível zerar o índice enriquecido e repetir o
experimento, porque os resultados vão para o artigo e precisam ser
reproduzíveis.

Registrar em CSV cada reconhecimento: data e hora, produto previsto,
similaridade, se o OCR contribuiu, e se o recorte foi auto-adicionado.

## Restrições de implementação

- Manter a captura separada do reconhecimento, em arquivos distintos. Depois a
  captura migra para um Raspberry Pi enviando as imagens pela rede, então o
  código de reconhecimento não pode assumir que a câmera é local.
- O repositório é público e citado no artigo, então precisa de README e
  organização.

## Fora de escopo por enquanto

- Comunicação em rede, MQTT ou broker.
- Integração com app de delivery ou pedido automático de compra.
- Treinar ou ajustar qualquer modelo.

## Como trabalhar

Passo a passo, testando cada estágio antes de seguir para o próximo, nesta
ordem: estágio 2 primeiro (testando com recortes de fotos do celular), depois o
estágio 1, depois o estágio 3. Commit a cada estágio que funcionar.

