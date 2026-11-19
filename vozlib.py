"""
vozlib — biblioteca do notebook "Raio-X da Voz"
================================================
Mede o perfil de uma voz a partir de um áudio e desenha os gráficos do raio-x.

Como usar no Colab:
    1. Envie este arquivo (vozlib.py) para o painel de arquivos, ao lado da pasta "audios".
    2. No notebook:   import vozlib as vz
    3. Use as funções com o prefixo "vz.":   r = vz.raio_x(vz.carregar("audios/A07_frase1.ogg"))

Se editar este arquivo com o notebook aberto, rode  importlib.reload(vz)  para recarregá-lo.
"""

# ============ BIBLIOTECAS QUE A vozlib USA ============
import os, subprocess
import numpy as np, pandas as pd, matplotlib.pyplot as plt, librosa
from scipy import fft as sfft
from IPython.display import Audio, display, HTML

# ============ CONSTANTES ============
SR, FRAME, HOP = 16000, 640, 160          # 16 kHz; quadros de 40 ms a cada 10 ms
EXTS = (".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm", ".aac")   # formatos de áudio aceitos


# ============ LER ============
def carregar(caminho, max_s=30):
    """Converte qualquer áudio para 16 kHz mono (via ffmpeg) e normaliza o volume de pico."""
    wav = "/tmp/_rx_" + os.path.basename(caminho) + ".wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", caminho, "-ac", "1", "-ar", str(SR), "-t", str(max_s), wav], check=True)
    y, _ = librosa.load(wav, sr=SR)
    return y / (np.abs(y).max() + 1e-9) * 0.9


# ============ MEDIR ============
NOTAS = ["Dó", "Dó#", "Ré", "Ré#", "Mi", "Fá", "Fá#", "Sol", "Sol#", "Lá", "Lá#", "Si"]

def midi(f):
    """Hz → número MIDI (escala em semitons; Lá4 = 440 Hz = 69)."""
    return 69 + 12*np.log2(np.asarray(f)/440)

def hz(m):
    """Número MIDI → Hz (o caminho inverso)."""
    return 440*2**((np.asarray(m) - 69)/12)

def nota(f):
    """Nome da nota musical mais próxima de uma frequência em Hz (ex.: 220 → "Lá3")."""
    if not f or np.isnan(f):
        return "—"
    m = int(round(float(midi(f))))
    return f"{NOTAS[m % 12]}{m//12 - 1}"

def quadros(y):
    """Fatia o áudio em quadros de 40 ms que começam a cada 10 ms (uma linha por quadro)."""
    return np.lib.stride_tricks.sliding_window_view(y, FRAME)[::HOP]

def estimar_f0(Q, energia, fmin=65, fmax=450, limiar_r=0.5, pct_energia=30):
    """F0 por AUTOCORRELAÇÃO: em cada quadro, o atraso k de maior correlação do sinal com ele mesmo é o período."""
    kmin, kmax = SR//fmax, SR//fmin
    X = Q - Q.mean(axis=1, keepdims=True)
    R = sfft.irfft(np.abs(sfft.rfft(X, n=2048))**2, n=2048)[:, :kmax+2]     # autocorrelação de todos os quadros (via FFT)
    R = R * FRAME/(FRAME - np.arange(kmax + 2))                              # estimador NÃO VIESADO: divide por (n − k) termos somados
    R = R / np.maximum(R[:, :1], 1e-12)                                      # normaliza: r(0) = 1, como um coeficiente de correlação
    f0 = np.full(len(Q), np.nan)
    for i in np.where(energia > np.percentile(energia, pct_energia))[0]:    # só quadros com energia acima do P30
        r = R[i]
        trecho = r[kmin:kmax]
        k = kmin + np.argmax(trecho >= 0.97*trecho.max())                   # 1º pico com ≥ 97% do máximo (evita erro de oitava)
        while k + 1 < kmax and r[k+1] > r[k]:
            k += 1
        if r[k] > limiar_r:                                                  # correlação forte = é voz
            a, b, c = r[k-1], r[k], r[k+1]
            den = a - 2*b + c
            if den:
                f0[i] = SR/(k + 0.5*(a - c)/den)                             # interpolação parabólica
            else:
                f0[i] = SR/k
    return f0

def cv_local(a):
    """Variação média entre valores vizinhos ÷ média, em % (usada no jitter e no shimmer)."""
    if len(a) < 2:
        return np.nan
    return 100*np.mean(np.abs(np.diff(a)))/np.mean(a)

MEL = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=26, fmin=80, fmax=7600)
def raio_x(y):
    """Transforma um áudio em um perfil vocal (dicionário de números)."""
    Q = quadros(y)
    E = Q.std(axis=1)                                                        # energia = desvio padrão de cada quadro
    f0 = estimar_f0(Q, E)
    iv = np.where(~np.isnan(f0))[0]
    voz = f0[iv]
    st = midi(voz)
    q1, q3 = np.percentile(st, [25, 75])
    iqr = q3 - q1
    limpo = st[(st >= q1 - 1.5*iqr) & (st <= q3 + 1.5*iqr)]                  # regra do boxplot remove outliers
    p10, p30, p90 = np.percentile(E, [10, 30, 90])
    suave = pd.Series(E).rolling(9, center=True, min_periods=1).mean().values   # média móvel de 90 ms
    picos = np.sum((suave[1:-1] > suave[:-2]) & (suave[1:-1] >= suave[2:]) & (suave[1:-1] > p30))
    ativos = Q[E > p30] * np.hanning(FRAME)
    P = np.abs(np.fft.rfft(ativos, n=1024))**2                               # espectro de potência dos quadros ativos
    freqs = np.fft.rfftfreq(1024, 1/SR)
    centroide = (P*freqs).sum(1)/(P.sum(1) + 1e-12)
    mfcc = sfft.dct(np.log(P @ MEL.T + 1e-10), type=2, norm="ortho", axis=1)[:, 1:13]   # 12 MFCCs por quadro
    return dict(duracao=len(y)/SR, f0=f0, energia=E, voz=voz, esp_medio=P.mean(0), freqs=freqs,
                f0_mediana=float(hz(np.median(limpo))),
                tess_lo=float(hz(np.percentile(limpo, 5))), tess_hi=float(hz(np.percentile(limpo, 95))),
                tessitura=float(np.percentile(limpo, 95) - np.percentile(limpo, 5)),
                expressividade=float(np.std(limpo, ddof=1)),
                ritmo=picos/(len(y)/SR),
                pausas=100*np.mean(E < p90*10**(-25/20)),
                dinamica=20*np.log10(p90/max(p10, 1e-9)),
                instabilidade=cv_local(1/voz), shimmer=cv_local(E[iv]),
                brilho=float(centroide.mean()), voz_pct=100*len(voz)/len(f0),
                impressao=np.concatenate([mfcc.mean(0), mfcc.std(0)]))       # 24 números = impressão digital do timbre

MEDIDORES = [  # (chave, nome, unidade, mín, máx, escala log?, rótulo esquerda, rótulo direita, comparação)
    ("f0_mediana", "Tom médio", "Hz", 80, 300, True, "grave", "agudo", "mais aguda"),
    ("tessitura", "Tessitura usada", "semitons", 0, 14, False, "estreita", "ampla", "mais ampla"),
    ("expressividade", "Expressividade", "st (DP)", 0, 5, False, "monótona", "melódica", "mais melódica"),
    ("ritmo", "Ritmo", "sílabas/s", 1.5, 7, False, "calmo", "acelerado", "mais acelerada"),
    ("pausas", "Pausas", "% do tempo", 0, 40, False, "corrida", "pausada", "mais pausada"),
    ("dinamica", "Dinâmica", "dB", 10, 45, False, "contida", "enfática", "mais enfática"),
    ("instabilidade", "Instabilidade", "% (jitter)", 0, 8, False, "firme", "trêmula", "mais trêmula"),
    ("brilho", "Brilho", "Hz", 300, 2500, True, "escura", "brilhante", "mais brilhante")]

def posicao(v, lo, hi, log):
    """Posição do valor v na régua do medidor, de 0 (esquerda) a 1 (direita)."""
    if log:
        t = (np.log(v) - np.log(lo))/(np.log(hi) - np.log(lo))
    else:
        t = (v - lo)/(hi - lo)
    return float(np.clip(t, 0, 1))

def faixa(r, chave):
    """Em que terço da régua o valor cai: 0 (início), 1 (meio) ou 2 (fim)."""
    for m in MEDIDORES:                       # procura o medidor dessa chave
        if m[0] == chave:
            lo, hi, log = m[3], m[4], m[5]
            break
    return min(2, int(posicao(r[chave], lo, hi, log)*3))

def resumo(r):
    """Duas frases em português que descrevem o perfil vocal r."""
    tom = ["grave", "média", "aguda"][faixa(r, "f0_mediana")]
    exp = ["monótona", "com entonação moderada", "bem melódica"][faixa(r, "expressividade")]
    rit = ["ritmo calmo", "ritmo médio", "ritmo acelerado"][faixa(r, "ritmo")]
    if r["instabilidade"] > 4:
        emissao = "com um leve tremor"
    else:
        emissao = "com emissão firme"
    return (f"Voz {tom}, {exp}, em {rit}",
            f"Vive entre {nota(r['tess_lo'])} e {nota(r['tess_hi'])} ({r['tessitura']:.1f} semitons), "
            f"{emissao} e pausas em {r['pausas']:.0f}% do tempo.")


# ============ DESENHAR ============
COR, QUENTE, CINZA = "#0b8f8a", "#e2513c", "#9aa2aa"
def desenhar_medidores(r, turma=None):
    """Desenha as 8 réguas do perfil; turma (lista de perfis) vira os pontos cinza."""
    fig, axs = plt.subplots(2, 4, figsize=(15, 4.2))
    paineis = axs.flatten()                   # os 8 painéis em uma lista simples
    for i in range(len(MEDIDORES)):
        ax = paineis[i]
        k, nome, un, lo, hi, log, esq, dir_, mais = MEDIDORES[i]

        # régua cinza de fundo
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-1, 1.6)
        ax.axis("off")
        ax.plot([0, 1], [0, 0], lw=5, color="#dcdfe2", solid_capstyle="round")

        # um ponto cinza para cada colega
        if turma:
            xs = []
            for t in turma:
                xs.append(posicao(t[k], lo, hi, log))
            ax.scatter(xs, [0]*len(xs), s=28, color=CINZA, alpha=.6, zorder=2)

        # o traço vermelho é a pessoa analisada
        x = posicao(r[k], lo, hi, log)
        ax.plot([x, x], [-0.45, 0.45], lw=5, color=QUENTE, solid_capstyle="round", zorder=3)
        ax.text(0, 1.05, nome, fontsize=11, weight="bold")
        ax.text(1, 1.05, f"{r[k]:.1f} {un}", fontsize=10, ha="right", family="monospace")
        ax.text(0, -0.85, esq, fontsize=9, color="gray")
        ax.text(1, -0.85, dir_, fontsize=9, color="gray", ha="right")

        # percentil: quantos colegas têm valor menor que o da pessoa
        if turma and len(turma) >= 3:
            abaixo = 0
            for t in turma:
                if t[k] < r[k]:
                    abaixo += 1
            p = 100*abaixo/len(turma)
            ax.text(0.5, 0.55, f"{mais} que {p:.0f}% da turma", fontsize=9, color=COR, ha="center")
    plt.tight_layout()
    plt.show()

def desenhar_melodia_e_timbre(r):
    """Desenha a melodia da fala (F0 ao longo do tempo) e o espectro médio (timbre)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.3))
    st = midi(r["voz"])
    lo = int(np.floor(min(st.min(), midi(r["tess_lo"])) - 2))
    hi = int(np.ceil(max(st.max(), midi(r["tess_hi"])) + 2))

    # fundo de "teclado": faixas cinza nas teclas pretas; as brancas viram marcas do eixo
    marcas = []
    for m in range(lo, hi + 1):
        if m % 12 in (1, 3, 6, 8, 10):
            a1.axhspan(m - .5, m + .5, color="#eeeeee", zorder=0)
        else:
            marcas.append(m)
    rotulos = []
    for m in marcas:
        rotulos.append(nota(hz(m)))

    a1.axhspan(midi(r["tess_lo"]), midi(r["tess_hi"]), color=COR, alpha=.08, zorder=0, label="tessitura (P5–P95)")
    t = np.arange(len(r["f0"]))*HOP/SR
    a1.scatter(t, midi(r["f0"]), s=6, color=COR, zorder=2)
    a1.axhline(midi(r["f0_mediana"]), color=QUENTE, ls="--", lw=1.2, label=f"mediana: {nota(r['f0_mediana'])}")
    a1.set_yticks(marcas, rotulos, fontsize=8)
    a1.set_ylim(lo, hi)
    a1.set(title="A melodia da fala (cada ponto = 10 ms)", xlabel="tempo (s)")
    a1.legend(loc="upper right", fontsize=8)

    lim = r["freqs"] <= 5000
    db = 10*np.log10(r["esp_medio"][lim] + 1e-12)
    db -= db.max()
    a2.fill_between(r["freqs"][lim], db, -60, color=COR, alpha=.18)
    a2.plot(r["freqs"][lim], db, color=COR)
    a2.axvline(r["brilho"], color=QUENTE, ls="--", lw=1.2, label=f"brilho (centroide) {r['brilho']:.0f} Hz")
    a2.set(title="O timbre: espectro médio (picos = formantes)", xlabel="frequência (Hz)", ylabel="dB", ylim=(-60, 3))
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def mostrar_raio_x(y, titulo, turma=None):
    """Calcula o raio-x de um áudio e mostra cartão, player e gráficos. Devolve o perfil."""
    r = raio_x(y)
    t1, t2 = resumo(r)
    display(HTML(f"""<div style="font-family:sans-serif;background:#e6f3f2;border-radius:12px;padding:14px 18px;display:flex;justify-content:space-between;gap:16px;align-items:center;color:#17191c">
      <div><div style="font:600 12px monospace;color:{COR};text-transform:uppercase">{titulo}</div>
      <div style="font:800 26px sans-serif;margin:4px 0">{t1}</div><div style="color:#5e646d">{t2}</div></div>
      <div style="text-align:center"><div style="font:800 46px sans-serif;color:{COR}">{nota(r['f0_mediana'])}</div>
      <div style="font:13px monospace;color:#5e646d">{r['f0_mediana']:.0f} Hz</div></div></div>"""))
    display(Audio(y, rate=SR))
    desenhar_medidores(r, turma)
    desenhar_melodia_e_timbre(r)
    return r


# ============ TESTE RÁPIDO ============
# Este bloco só roda quando o arquivo é executado direto (!python vozlib.py),
# e NÃO quando ele é importado com "import vozlib".
if __name__ == "__main__":
    t = np.arange(2*SR)/SR
    tom = np.zeros_like(t)
    for h in range(1, 10):                     # um "Lá3" (220 Hz) com 9 harmônicos
        tom += np.sin(2*np.pi*220*h*t)/h
    r = raio_x(tom)
    print("Teste da vozlib → nota esperada: Lá3 | nota medida:", nota(r["f0_mediana"]))
