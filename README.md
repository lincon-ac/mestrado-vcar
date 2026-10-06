# Pipeline de análise de imagem com OpenCV

## Repositório do projeto

- [lincon-ac/mestrado-vcar](https://github.com/lincon-ac/mestrado-vcar)

## Arquivos

- `pipeline_analise_imagem.ipynb`: notebook Jupyter executado, com as etapas 1–6, resultados e respostas do item 7 em Markdown.
- `pipeline_analise_imagem.py`: script com o pipeline dos itens 1–6; gera figuras na pasta `resultados/` e imprime as métricas.
- `imagem_mercado.jpg`: imagem de entrada (necessária para executar código/notebook sem trocar o caminho).
- `analise_final.pdf`: relatório com as respostas do item 7, métricas, figuras, fonte e declaração de IA.
- `resultados/`: figuras salvas pelo script.

## Requisitos e execução

Instale Python 3, `opencv-contrib-python` (para SIFT), `numpy` e `matplotlib`:

```bash
python -m pip install opencv-contrib-python numpy matplotlib
```

Execute o pipeline na pasta do projeto:

```bash
python pipeline_analise_imagem.py
```

Para indicar outro arquivo de imagem ou pasta de saída:

```bash
python pipeline_analise_imagem.py --image outra_imagem.jpg --output-dir resultados
```

Abra o notebook pelo Jupyter e execute as células na ordem; mantenha `imagem_mercado.jpg` na mesma pasta. A fotografia usada é de Francisco Anzola, _Fruit Stand_, licenciada sob [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/); a atribuição e a fonte estão registradas no notebook e no PDF.
