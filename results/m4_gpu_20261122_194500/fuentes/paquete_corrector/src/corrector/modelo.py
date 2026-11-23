"""Corrector semántico por hora sobre E + C+R (red numérica congelada; solo se ajusta el corrector).

Entrada del día objetivo d:
  num   calendario del día d y residuos relativos por hora de E + C+R ya observables: r_{d-2,h} y la media de
        r_{d-8..d-2,h} (24 + 24), estandarizados con el tramo de ajuste
  texto hasta M candidatos (fragmentos E5 congelados de 768 dimensiones) con sus rasgos y su fuente
Arquitectura:
  e_j = LN(W_e x_j + W_f f_j + emb(fuente_j))                       representación de cada candidato
  q_h = MLP(num) + emb(h),  h = 0..23                                una consulta por hora del horizonte
  c_h = atención multicabeza de q_h sobre {e_j} ∪ {nulo}             el vector nulo permite no usar texto
  Δ_h = b_h · σ(g_h) · tanh(v_h),  [v_h, g_h] = MLP([c_h, q_h])      corrección acotada con compuerta (sesgo −2)
  ŷ_h = ŷ^{C+R}_h (1 + Δ_h)
b_h es el percentil 99 de |r_h| en el tramo de ajuste: la corrección nunca sale del rango de residuos observado.
Pérdida: MAE de y frente a ŷ escalado por q168 (el MASE del contrato). Supervisión exclusiva del residuo: la red
numérica no se reentrena, así que el texto solo puede reducir la pérdida explicando lo que E + C+R no explica.

Variantes (máscara de candidatos): num_horario (solo el nulo: corrector numérico por hora), oficial_reciente,
oficial (reciente + alineados oficiales), oficial_prensa (todo); controles: metadatos (sin vectores ni reserva
alineada; solo canal, antigüedad y fuente) y desalineado_s (los candidatos del día d - s).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

VARIANTES = ("num_horario", "oficial_reciente", "oficial", "oficial_prensa", "metadatos")


class Corrector(nn.Module):
    def __init__(self, n_num: int, n_feat: int, n_fuentes: int, d: int, cabezas: int, dropout: float, cota: np.ndarray, dim_texto: int = 768):
        super().__init__()
        self.proy = nn.Linear(dim_texto, d)
        self.rasgo = nn.Linear(n_feat, d)
        self.fuente = nn.Embedding(n_fuentes + 1, d)
        self.norma = nn.LayerNorm(d)
        self.consulta = nn.Sequential(nn.Linear(n_num, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.hora = nn.Embedding(24, d)
        self.nulo = nn.Parameter(torch.zeros(1, 1, d))
        self.atencion = nn.MultiheadAttention(d, cabezas, dropout=dropout, batch_first=True)
        self.salida = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 2))
        nn.init.zeros_(self.salida[-1].weight)
        with torch.no_grad():
            self.salida[-1].bias.copy_(torch.tensor([0.0, -2.0]))
        self.register_buffer("cota", torch.as_tensor(cota, dtype=torch.float32))

    def forward(self, num, emb, feat, fte, mascara):
        """num (B, n_num); emb (B, M, 768); feat (B, M, F); fte (B, M); mascara (B, M) True = candidato válido."""
        B = num.shape[0]
        e = self.norma(self.proy(emb) + self.rasgo(feat) + self.fuente(fte.long()))
        e = torch.cat([self.nulo.expand(B, 1, -1), e], dim=1)
        m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=num.device), mascara], dim=1)
        q = self.consulta(num)[:, None, :] + self.hora.weight[None]
        c, _ = self.atencion(q, e, e, key_padding_mask=~m, need_weights=False)
        vg = self.salida(torch.cat([c, q], dim=-1))
        return self.cota * torch.sigmoid(vg[..., 1]) * torch.tanh(vg[..., 0])


def mascara_variante(variante: str, reservas: dict, cand: np.ndarray, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (máscara de candidatos válidos, rasgos modificados) para una variante."""
    M = cand.shape[1]
    sel = np.zeros(M, dtype=bool)
    al, of, pr = (slice(*reservas[k]) for k in ("alineados", "oficial", "prensa"))
    if variante == "num_horario":
        pass
    elif variante == "oficial_reciente":
        sel[of] = True
    elif variante == "oficial":
        sel[of] = True
        sel[al] = True
    elif variante in ("oficial_prensa", "metadatos") or variante.startswith("desalineado"):
        sel[:] = True
        if variante == "metadatos":
            sel[al] = False                                    # la reserva alineada depende del contenido (fechas del texto)
    else:
        raise ValueError(variante)
    masc = (cand >= 0) & sel[None, :]
    if variante.startswith("oficial") and variante != "oficial_prensa":
        masc &= feat[..., 0] > 0.5                              # en las variantes oficiales, solo documentos oficiales
    f = feat.copy()
    if variante == "metadatos":
        f[..., 5:] = 0.0
    return masc, f


class Datos:
    """Arreglos alineados por día para una semilla de la base: num, objetivo y candidatos de texto."""

    def __init__(self, cr: dict, C, cand: dict, q168: float, desplazamiento: int = 0):
        dias = np.asarray(cr["dia"]).astype("datetime64[D]")
        self.dias, self.anio = dias, np.asarray(cr["anio"])
        self.y, self.ycr = cr["y"].astype("float32"), cr["yhat_cr"].astype("float32")
        r = (self.y - self.ycr) / self.ycr
        serie = {d: r[i] for i, d in enumerate(dias)}
        un = np.timedelta64(1, "D")
        lag2 = np.stack([serie.get(d - 2 * un, np.zeros(24, dtype="float32")) for d in dias])
        m7 = np.stack([np.mean([serie[d - k * un] for k in range(2, 9) if d - k * un in serie] or [np.zeros(24, dtype="float32")], axis=0)
                       for d in dias])
        cal = C.reindex(dias.astype("datetime64[ns]")).fillna(0.0).to_numpy(dtype="float32")
        self.num = np.hstack([cal, lag2, m7]).astype("float32")
        self.r = r.astype("float32")
        cd = np.asarray(cand["dias"]).astype("datetime64[D]")
        pos = {d: i for i, d in enumerate(cd)}
        fila = np.array([pos.get(d - desplazamiento * un, -1) for d in dias])
        if (fila < 0).any():
            raise RuntimeError(f"Faltan candidatos de texto para {int((fila < 0).sum())} días (desplazamiento {desplazamiento})")
        self.fila = fila
        self.q168 = q168

    def tramo(self, anios) -> np.ndarray:
        return np.flatnonzero(np.isin(self.anio, list(anios)))


def entrenar(datos: Datos, cand: dict, emb: torch.Tensor, variante: str, p: dict, semilla: int, tramos: dict, dispositivo: str,
             max_epocas: int = 60, paciencia: int = 8, observador=None) -> dict:
    """Ajusta el corrector con tramos['ajuste'], detiene con tramos['parada'] y predice tramos['evaluacion']."""
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    a, s, e = tramos["ajuste"], tramos["parada"], tramos["evaluacion"]
    mu, sd = datos.num[a].mean(0), datos.num[a].std(0) + 1e-6
    num = torch.from_numpy((datos.num - mu) / sd).to(dispositivo)
    cota = np.percentile(np.abs(datos.r[a]), 99, axis=0).astype("float32")
    masc, feat = mascara_variante(variante, cand["reservas"], cand["cand"], cand["feat"].astype("float32"))
    C = torch.from_numpy(np.maximum(cand["cand"], 0).astype("int64"))
    Mk, Ft, Fu = torch.from_numpy(masc), torch.from_numpy(feat), torch.from_numpy(cand["fuente"].astype("int64"))
    sin_vectores = variante == "metadatos"
    y, ycr = torch.from_numpy(datos.y).to(dispositivo), torch.from_numpy(datos.ycr).to(dispositivo)
    red = Corrector(num.shape[1], feat.shape[2], int(cand["n_fuentes"]), int(p["d"]), int(p["cabezas"]), float(p["dropout"]), cota).to(dispositivo)
    opt = torch.optim.AdamW(red.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    gen = torch.Generator().manual_seed(semilla)

    def lote(ix):
        f = torch.from_numpy(datos.fila[ix])
        c = C[f]
        ev = torch.zeros(len(ix), c.shape[1], emb.shape[1], device=dispositivo) if sin_vectores else emb[c.to(emb.device)].to(dispositivo).float()
        return num[ix], ev, Ft[f].to(dispositivo), Fu[f].to(dispositivo), Mk[f].to(dispositivo)

    def perdida(ix, entrenando):
        red.train(entrenando)
        d = red(*lote(ix))
        return (y[ix] - ycr[ix] * (1 + d)).abs().mean() / datos.q168, d

    @torch.no_grad()
    def evaluar(ix):
        out, tot = [], 0.0
        for k in range(0, len(ix), 256):
            l, d = perdida(ix[k:k + 256], False)
            tot += float(l) * len(ix[k:k + 256])
            out.append(d.cpu().numpy())
        return tot / max(len(ix), 1), np.concatenate(out) if out else np.zeros((0, 24))

    mejor, estado, sin, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, max_epocas + 1):
        orden = torch.from_numpy(a)[torch.randperm(len(a), generator=gen)].numpy()
        for k in range(0, len(orden), int(p["lote"])):
            ix = orden[k:k + int(p["lote"])]
            if len(ix) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            l, _ = perdida(ix, True)
            l.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            opt.step()
        vl, _ = evaluar(s)
        hist.append(dict(epoca=ep, mase_parada=vl))
        if observador is not None:
            observador(hist[-1])
        if vl < mejor - 1e-6:
            mejor, estado, sin = vl, {k: v.detach().clone() for k, v in red.state_dict().items()}, 0
        else:
            sin += 1
            if sin >= paciencia:
                break
    red.load_state_dict(estado)
    mase_e, delta = evaluar(e)
    return dict(delta=delta, mase_parada=mejor, mase_evaluacion=mase_e, epocas=len(hist), historia=hist, segundos=round(time.time() - t0, 1),
                cota=cota.tolist(), n_parametros=int(sum(v.numel() for v in red.parameters())),
                compuerta_media=float(np.mean(np.abs(delta)) / max(float(np.mean(cota)), 1e-9)))
