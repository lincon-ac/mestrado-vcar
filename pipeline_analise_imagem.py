#!/usr/bin/env python3
"""Pipeline didático de processamento de imagem com OpenCV (itens 1–6).

Uso:
    python pipeline_analise_imagem.py [--image caminho/imagem.jpg] [--output-dir resultados]

Dependências: opencv-contrib-python (SIFT), numpy e matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def executar_pipeline(caminho_imagem: Path, pasta_saida: Path) -> dict:
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # OpenCV lê em BGR; a conversão explícita permite exibição correta em matplotlib.
    imagem_bgr = cv2.imread(str(caminho_imagem), cv2.IMREAD_COLOR)
    if imagem_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    altura, largura, canais = imagem_bgr.shape
    print("1. CARACTERIZAÇÃO")
    print(f"Arquivo: {caminho_imagem}")
    print(f"Dimensões (altura × largura × canais): {imagem_bgr.shape}")
    print(f"Largura × altura: {largura} × {altura} pixels")
    print(f"Número de canais: {canais}; dtype: {imagem_bgr.dtype}; intervalo: {imagem_bgr.min()}–{imagem_bgr.max()}")

    # 2. Grayscale e HSV. Os planos HSV são exibidos individualmente em tons de cinza.
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    canal_h, canal_s, canal_v = cv2.split(hsv)
    fig, eixos = plt.subplots(2, 3, figsize=(14, 8))
    paineis = [
        (imagem_rgb, "Original (RGB)", None),
        (cinza, "Escala de cinza", "gray"),
        (canal_h, "H — matiz", "gray"),
        (canal_s, "S — saturação", "gray"),
        (canal_v, "V — valor/brilho", "gray"),
    ]
    for ax, (dados, titulo, cmap) in zip(eixos.flat, paineis):
        ax.imshow(dados, cmap=cmap)
        ax.set_title(titulo)
        ax.axis("off")
    eixos.flat[-1].axis("off")
    fig.suptitle("Espaços de cor e planos HSV", fontsize=15)
    fig.tight_layout()
    fig.savefig(pasta_saida / "fig_01_espacos_cor.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. Redimensionamento: dimensões calculadas a partir de image.shape.
    nova_largura = max(1, int(largura * 0.5))
    nova_altura = max(1, int(altura * 0.5))
    redimensionada = cv2.resize(imagem_rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    print("\n3. REDIMENSIONAMENTO")
    print(f"Dimensões resultantes: {nova_largura} × {nova_altura} pixels (50% em cada eixo)")
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    eixos[0].imshow(imagem_rgb); eixos[0].set_title(f"Original — {largura} × {altura}")
    eixos[1].imshow(redimensionada); eixos[1].set_title(f"50% — {nova_largura} × {nova_altura}")
    for ax in eixos: ax.axis("off")
    fig.tight_layout()
    fig.savefig(pasta_saida / "fig_02_redimensionamento.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4. Afim com pontos definidos proporcionalmente à largura/altura.
    pontos_origem = np.float32([
        [0.12 * largura, 0.12 * altura],
        [0.88 * largura, 0.12 * altura],
        [0.12 * largura, 0.88 * altura],
    ])
    pontos_destino = np.float32([
        [0.16 * largura, 0.16 * altura],
        [0.86 * largura, 0.22 * altura],
        [0.10 * largura, 0.84 * altura],
    ])
    matriz_afim = cv2.getAffineTransform(pontos_origem, pontos_destino)
    imagem_afim_bgr = cv2.warpAffine(
        imagem_bgr, matriz_afim, (largura, altura),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    imagem_afim_rgb = cv2.cvtColor(imagem_afim_bgr, cv2.COLOR_BGR2RGB)
    cinza_afim = cv2.cvtColor(imagem_afim_bgr, cv2.COLOR_BGR2GRAY)
    print("\n4. TRANSFORMAÇÃO AFIM")
    print("Pontos de controle (origem → destino), definidos por proporções da imagem:")
    for origem, destino in zip(pontos_origem, pontos_destino):
        print(f"  {origem.round(1).tolist()} → {destino.round(1).tolist()}")
    print("Matriz afim:\n", np.array2string(matriz_afim, precision=4))
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    eixos[0].imshow(imagem_rgb); eixos[0].set_title("Original")
    eixos[1].imshow(imagem_afim_rgb); eixos[1].set_title("Transformação afim")
    for ax in eixos: ax.axis("off")
    fig.tight_layout()
    fig.savefig(pasta_saida / "fig_03_transformacao_afim.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 5. Canny em grayscale com três pares ordenados de limiares.
    limiares = [(30, 60), (80, 160), (150, 250)]
    contagens_bordas: dict[tuple[int, int], int] = {}
    resultados_canny = []
    for t1, t2 in limiares:
        bordas = cv2.Canny(cinza, t1, t2)
        resultados_canny.append(bordas)
        contagens_bordas[(t1, t2)] = int(np.count_nonzero(bordas))
    print("\n5. CANNY — quantidade de pixels de borda (valor 255)")
    for par, quantidade in contagens_bordas.items():
        percentual = 100 * quantidade / (altura * largura)
        print(f"  thresholds={par}: {quantidade:,} pixels ({percentual:.2f}% da imagem)")
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5))
    for ax, bordas, (t1, t2) in zip(eixos, resultados_canny, limiares):
        ax.imshow(bordas, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Canny ({t1}, {t2})\n{contagens_bordas[(t1, t2)]:,} pixels de borda")
        ax.axis("off")
    fig.suptitle("Comparação de thresholds do Canny", fontsize=15)
    fig.tight_layout()
    fig.savefig(pasta_saida / "fig_04_canny.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 6. SIFT: detecção separada na imagem original e na transformada.
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("Esta instalação do OpenCV não oferece cv2.SIFT_create(); instale opencv-contrib-python.")
    sift = cv2.SIFT_create()
    keypoints_original, descritores_original = sift.detectAndCompute(cinza, None)
    keypoints_afim, descritores_afim = sift.detectAndCompute(cinza_afim, None)
    visual_original = cv2.drawKeypoints(
        imagem_rgb, keypoints_original, None, color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    visual_afim = cv2.drawKeypoints(
        imagem_afim_rgb, keypoints_afim, None, color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    print("\n6. SIFT")
    print(f"Keypoints na imagem original: {len(keypoints_original)}")
    print(f"Keypoints após transformação afim: {len(keypoints_afim)}")
    print(f"Variação absoluta: {len(keypoints_afim) - len(keypoints_original):+d}")
    fig, eixos = plt.subplots(1, 2, figsize=(15, 6))
    eixos[0].imshow(visual_original)
    eixos[0].set_title(f"Original — {len(keypoints_original)} keypoints")
    eixos[1].imshow(visual_afim)
    eixos[1].set_title(f"Afim — {len(keypoints_afim)} keypoints")
    for ax in eixos: ax.axis("off")
    fig.suptitle("Pontos de interesse SIFT", fontsize=15)
    fig.tight_layout()
    fig.savefig(pasta_saida / "fig_05_sift.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "shape": tuple(imagem_bgr.shape),
        "dtype": str(imagem_bgr.dtype),
        "resized_shape": (nova_altura, nova_largura, canais),
        "canny_counts": contagens_bordas,
        "sift_original": len(keypoints_original),
        "sift_affine": len(keypoints_afim),
        "affine_matrix": matriz_afim,
        "output_dir": pasta_saida,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    raiz = Path(__file__).resolve().parent
    parser.add_argument("--image", type=Path, default=raiz / "imagem_mercado.jpg", help="Imagem colorida de entrada")
    parser.add_argument("--output-dir", type=Path, default=raiz / "resultados", help="Pasta de saída das figuras")
    args = parser.parse_args()
    executar_pipeline(args.image, args.output_dir)


if __name__ == "__main__":
    main()
