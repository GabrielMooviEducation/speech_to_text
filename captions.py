"""Legenda WebVTT a partir dos trechos que o Whisper devolve.

POR QUE AQUI: o Whisper já entrega cada trecho com início e fim, e até agora isso
era jogado fora (pro resumo, o texto corrido basta). Com os tempos guardados, o
vídeo da aula pode ir pro player com faixa de legenda (o Cloudflare Stream aceita
um .vtt por idioma) e a tradução pra outros idiomas mantém a sincronia.

SEM TEMPO POR PALAVRA: um trecho do Whisper costuma ter de 2 a 10 segundos. O que
passa do tamanho de uma legenda é dividido aqui, distribuindo o tempo do trecho
pelos caracteres. Dentro de um trecho a fala tem ritmo quase constante, então o
erro fica em fração de segundo, sem o custo de CPU do `word_timestamps` no motor
local e sem perder a pontuação (as palavras da OpenAI vêm sem ela).
"""

import math

# (início em segundos, fim em segundos, texto)
Segment = tuple[float, float, str]

# Padrão de legenda de streaming: até 2 linhas de 42 caracteres e até 7 segundos
# na tela. Passou disso, o aluno não termina de ler antes da próxima.
LINE_CHARS = 42
CUE_CHARS = LINE_CHARS * 2
CUE_SECONDS = 7.0


def _nearest_space(text: str, target: float, after: int) -> int | None:
    best = None
    for i, c in enumerate(text):
        if c == " " and i > after and (best is None or abs(i - target) < abs(best - target)):
            best = i
    return best


def _split(start: float, end: float, text: str) -> list[Segment]:
    """Quebra um trecho em legendas que cabem na tela, cortando em espaço."""
    text = " ".join(text.split())
    if not text:
        return []
    # Trecho sem duração (o Whisper às vezes devolve início == fim): ganha 1s, e
    # o ajuste de sobreposição em `build_vtt` apara se invadir o próximo.
    end = max(end, start + 1.0)
    n = max(math.ceil(len(text) / CUE_CHARS), math.ceil((end - start) / CUE_SECONDS))
    cuts = [0]
    for k in range(1, n):
        cut = _nearest_space(text, len(text) * k / n, cuts[-1])
        if cut is None:
            break
        cuts.append(cut)
    cuts.append(len(text))

    per_char = (end - start) / len(text)
    return [
        (start + a * per_char, start + b * per_char, text[a:b].strip())
        for a, b in zip(cuts, cuts[1:])
    ]


def _wrap(text: str) -> str:
    """Até 2 linhas, cortando no espaço mais perto do meio."""
    if len(text) <= LINE_CHARS:
        return text
    cut = _nearest_space(text, len(text) / 2, -1)
    return text if cut is None else f"{text[:cut]}\n{text[cut + 1:]}"


def _escape(text: str) -> str:
    # No WebVTT, `<` abre tag e `&` abre entidade; escapar `>` também impede que
    # uma fala com "-->" seja lida como linha de tempo.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _timestamp(seconds: float) -> str:
    ms = round(max(seconds, 0.0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_vtt(segments: list[Segment]) -> str | None:
    """Monta o .vtt. None = nenhum trecho com texto (áudio mudo)."""
    cues = [cue for seg in segments for cue in _split(*seg)]
    if not cues:
        return None
    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(cues):
        # Duas legendas na tela ao mesmo tempo empilham no player: o fim de uma
        # não pode passar do início da seguinte.
        if i + 1 < len(cues) and cues[i + 1][0] > start:
            end = min(end, cues[i + 1][0])
        lines += [f"{_timestamp(start)} --> {_timestamp(end)}", _escape(_wrap(text)), ""]
    return "\n".join(lines)
