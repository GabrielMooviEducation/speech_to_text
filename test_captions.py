"""Testes da legenda WebVTT montada a partir dos trechos do Whisper.

    python -m unittest test_captions -v
"""

import re
import unittest

import captions

TIME_LINE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})$")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _cues(vtt: str) -> list[tuple[float, float, str]]:
    """Lê o .vtt de volta: (início, fim, texto com as quebras de linha)."""
    blocks = vtt.split("\n\n")
    assert blocks[0] == "WEBVTT"
    out = []
    for block in blocks[1:]:
        if not block.strip():
            continue
        time_line, *text = block.strip("\n").split("\n")
        g = TIME_LINE.match(time_line).groups()
        out.append((_seconds(*g[:4]), _seconds(*g[4:]), "\n".join(text)))
    return out


LONG = (
    "Hoje a gente vai ver como funciona uma BAPI de criação de pedido de compra, "
    "começando pela transação SE37, depois passando pelos parâmetros de entrada "
    "e terminando com o tratamento da tabela de retorno, que é onde mora o erro."
)


class BuildVttTest(unittest.TestCase):
    def test_trecho_curto_vira_uma_legenda(self):
        cues = _cues(captions.build_vtt([(1.0, 3.5, " Bom dia, pessoal. ")]))
        self.assertEqual(cues, [(1.0, 3.5, "Bom dia, pessoal.")])

    def test_sem_texto_nao_gera_legenda(self):
        self.assertIsNone(captions.build_vtt([]))
        self.assertIsNone(captions.build_vtt([(0.0, 2.0, "   ")]))

    def test_trecho_longo_e_dividido_sem_perder_texto_nem_tempo(self):
        cues = _cues(captions.build_vtt([(10.0, 22.0, LONG)]))
        self.assertGreater(len(cues), 1)
        self.assertEqual(cues[0][0], 10.0)
        self.assertEqual(cues[-1][1], 22.0)
        for (_, end, _), (start, _, _) in zip(cues, cues[1:]):
            self.assertAlmostEqual(end, start, places=3)
        joined = " ".join(text.replace("\n", " ") for _, _, text in cues)
        self.assertEqual(joined, LONG)
        for _, _, text in cues:
            self.assertLessEqual(len(text.replace("\n", " ")), captions.CUE_CHARS)

    def test_trecho_demorado_e_dividido_mesmo_com_pouco_texto(self):
        # O corte cai no espaço mais perto, então cada legenda fica PERTO dos 7s,
        # não exatamente abaixo: o que importa é não sobrar uma de 20s na tela.
        cues = _cues(captions.build_vtt([(0.0, 20.0, "então aqui a gente espera o job terminar de rodar")]))
        self.assertEqual(len(cues), 3)
        self.assertEqual((cues[0][0], cues[-1][1]), (0.0, 20.0))

    def test_quebra_em_duas_linhas_de_ate_42(self):
        cues = _cues(captions.build_vtt([(0.0, 5.0, "a tabela EKKO guarda o cabeçalho e a EKPO guarda os itens")]))
        self.assertEqual(len(cues), 1)
        lines = cues[0][2].split("\n")
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertLessEqual(len(line), captions.LINE_CHARS)

    def test_escapa_marcacao_do_vtt(self):
        vtt = captions.build_vtt([(0.0, 2.0, "se a < b && c --> d")])
        self.assertIn("se a &lt; b &amp;&amp; c --&gt; d", vtt)

    def test_legenda_nao_invade_a_seguinte(self):
        cues = _cues(captions.build_vtt([(0.0, 3.0, "primeira fala"), (2.0, 4.0, "segunda fala")]))
        self.assertEqual(cues[0][1], 2.0)

    def test_trecho_sem_duracao_ganha_tempo_na_tela(self):
        cues = _cues(captions.build_vtt([(5.0, 5.0, "ok")]))
        self.assertEqual(cues, [(5.0, 6.0, "ok")])

    def test_tempo_acima_de_uma_hora(self):
        cues = _cues(captions.build_vtt([(3725.25, 3727.0, "já passou de uma hora")]))
        self.assertEqual(cues[0][0], 3725.25)
        self.assertIn("01:02:05.250 --> 01:02:07.000", captions.build_vtt([(3725.25, 3727.0, "x")]))


if __name__ == "__main__":
    unittest.main()
