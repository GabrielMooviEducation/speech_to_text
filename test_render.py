"""Testes do montador de comandos do render.

Não exercitam ffmpeg (isso é o teste de paridade visual, que precisa das fontes
reais): exercitam a ÁLGEBRA do grafo, que é onde os erros são silenciosos — um
rótulo de filtro trocado, um índice de entrada fora de ordem ou um `atempo` fora
da faixa aceita não quebram o import, quebram o vídeo do professor.

    python -m unittest test_render -v
"""

import os
import unittest

os.environ.setdefault("RENDER_ENCODER", "libx264")  # não sonda GPU no teste

import render  # noqa: E402


def _plan(**over) -> render.RenderPlan:
    base = dict(
        recording_id="rec1",
        spec_hash="h1",
        upload_url="https://minio.example.com/export.mp4",
        sources={
            "screen": "https://minio.example.com/screen-fixed.mp4",
            "camera": "https://minio.example.com/camera-fixed.mp4",
        },
        duration=10.0,
        segments=[
            render.Segment(
                dur=10.0,
                under_url="https://minio.example.com/under.png",
                over_url="https://minio.example.com/over.png",
                layers=[
                    render.VideoLayer(
                        source="screen",
                        start=2.0,
                        rect=render.Rect(x=100, y=50, w=1720, h=968),
                        crop=render.Rect(x=0, y=0.05, w=1, h=0.9),
                        mask_url="https://minio.example.com/mask-screen.png",
                    ),
                    render.VideoLayer(
                        source="camera",
                        start=2.0,
                        rect=render.Rect(x=1500, y=750, w=384, h=216),
                        fit="cover",
                        mirror=True,
                        brightness=1.2,
                        contrast=1.1,
                    ),
                ],
                fade_in=0.35,
                fade_out=0.35,
            )
        ],
        audio_clips=[
            render.AudioClip(source="camera", start=2.0, dur=10.0, at=0.0, gain=1.0)
        ],
    )
    base.update(over)
    return render.RenderPlan(**base)


def _local(plan: render.RenderPlan) -> dict[str, str]:
    """Mapa URL → caminho local, como o `_run_plan` monta depois do download."""
    return {u: f"/tmp/{i}.bin" for i, u in enumerate(plan.urls())} | {
        u: f"/tmp/src{i}.mp4" for i, u in enumerate(plan.sources.values())
    }


class SegmentCmd(unittest.TestCase):
    def test_ordem_das_entradas_bate_com_os_rotulos(self):
        plan = _plan()
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        # 0=under, 1=tela, 2=máscara da tela, 3=câmera, 4=over (sem máscara)
        self.assertIn("[0:v]format=rgba", fc)
        self.assertIn("[1:v]fps=30", fc)
        self.assertIn("[2:v]scale=1720:968,format=gray[m0]", fc)
        self.assertIn("[3:v]fps=30", fc)
        self.assertIn("[4:v]format=rgba[ov]", fc)
        # Cada `-i` do comando é uma entrada; o maior índice citado no grafo não
        # pode passar disso, senão o ffmpeg morre com "Invalid file index".
        n_inputs = cmd.count("-i")
        self.assertEqual(n_inputs, 5)

    def test_corte_na_entrada_usa_tempo_de_fonte_e_duracao_com_velocidade(self):
        plan = _plan()
        plan.segments[0].speed = 2.0
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        # 10s de timeline a 2× consomem 20s de fonte, a partir do segundo 2.
        self.assertIn("2.000000", cmd)
        self.assertIn("20.000000", cmd)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("setpts=(PTS-STARTPTS)/2.000000", fc)

    def test_cover_recorta_depois_de_escalar(self):
        plan = _plan()
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("scale=384:216:force_original_aspect_ratio=increase,crop=384:216", fc)

    def test_brilho_multiplicativo_nao_vai_pro_eq(self):
        plan = _plan()
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("colorchannelmixer=rr=1.2000", fc)
        self.assertIn("eq=contrast=1.1000", fc)
        self.assertNotIn("eq=brightness", fc)

    def test_camada_sem_mascara_nao_gera_alphamerge(self):
        plan = _plan()
        plan.segments[0].layers[0].mask_url = None
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertEqual(fc.count("alphamerge"), 0)
        self.assertEqual(cmd.count("-i"), 4)

    def test_fade_de_saida_comeca_no_fim_do_segmento(self):
        plan = _plan()
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("fade=t=in:st=0:d=0.3500", fc)
        self.assertIn("fade=t=out:st=9.6500:d=0.3500", fc)
        self.assertTrue(fc.rstrip().endswith("[vout]"))

    def test_saida_e_muda_e_limitada_a_duracao(self):
        plan = _plan()
        cmd = render._segment_cmd(plan, plan.segments[0], _local(plan), "/tmp/out.mp4")
        self.assertIn("-an", cmd)
        self.assertEqual(cmd[cmd.index("-t") + 1], "10.000000")


class AudioCmd(unittest.TestCase):
    def test_sem_audio_nenhum_devolve_none(self):
        plan = _plan(audio_clips=[])
        self.assertIsNone(render._audio_cmd(plan, _local(plan), "/tmp/a.m4a"))

    def test_clip_mudo_nao_vira_entrada(self):
        plan = _plan()
        plan.audio_clips[0].gain = 0.0
        self.assertIsNone(render._audio_cmd(plan, _local(plan), "/tmp/a.m4a"))

    def test_amix_nao_normaliza(self):
        plan = _plan(
            audio_blocks=[
                render.AudioBlock(
                    source="camera", at=1.0, dur=5.0, gain=0.15,
                    fade_in=1.0, fade_out=2.0, loop=True,
                )
            ]
        )
        cmd = render._audio_cmd(plan, _local(plan), "/tmp/a.m4a")
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("amix=inputs=2:normalize=0", fc)
        self.assertIn("alimiter=", fc)

    def test_bloco_em_loop_usa_stream_loop_antes_da_entrada(self):
        plan = _plan(
            audio_blocks=[render.AudioBlock(source="camera", at=0, dur=5, loop=True)]
        )
        cmd = render._audio_cmd(plan, _local(plan), "/tmp/a.m4a")
        self.assertIn("-stream_loop", cmd)
        self.assertLess(cmd.index("-stream_loop"), len(cmd) - 1)

    def test_fade_do_bloco_vem_antes_do_adelay(self):
        plan = _plan(
            audio_blocks=[
                render.AudioBlock(source="camera", at=3.0, dur=5.0, fade_in=1.0)
            ]
        )
        cmd = render._audio_cmd(plan, _local(plan), "/tmp/a.m4a")
        fc = cmd[cmd.index("-filter_complex") + 1]
        chain = [c for c in fc.split(";") if "[ab0]" in c][0]
        self.assertLess(chain.index("afade=t=in"), chain.index("adelay=3000"))


class Atempo(unittest.TestCase):
    def test_velocidade_1_nao_gera_filtro(self):
        self.assertEqual(render._atempo_chain(1.0), [])

    def test_acima_de_2_encadeia(self):
        self.assertEqual(render._atempo_chain(4.0), ["atempo=2.0", "atempo=2.000000"])

    def test_abaixo_de_meio_encadeia(self):
        # O editor permite 0.5 como mínimo, mas a cadeia tem que aguentar menos.
        self.assertEqual(render._atempo_chain(0.25), ["atempo=0.5", "atempo=0.500000"])

    def test_dentro_da_faixa_e_um_filtro_so(self):
        self.assertEqual(render._atempo_chain(1.5), ["atempo=1.500000"])


class Fila(unittest.TestCase):
    def test_mesma_gravacao_nao_entra_duas_vezes(self):
        plan = _plan()
        self.assertEqual(render.enqueue(plan), "queued")
        self.assertEqual(render.enqueue(plan), "duplicate")
        render._queue.get_nowait()
        render._active.discard(plan.recording_id)


if __name__ == "__main__":
    unittest.main()
