"""Constructores de subgrafos de regularidad controlada (v8 — EXP-5).

Reescriben la topología del subgrafo H_t construido por BFS **sobre los mismos
top-M nodos**, para probar el régimen donde la teoría (Cap. 4) predice ventaja
dispersiva de la DTQW:

  - ``d_regular``  : grafo d-regular por emparejamiento voraz de peso máximo
                     con topes de grado y reparación por intercambio de aristas.
                     Sub-variantes: pesos = afinidades retenidas renormalizadas
                     o pesos uniformes (separa topología de pesos).
  - ``cycle``      : ciclo C_M con orden por heurística vecino-más-cercano
                     sobre la afinidad, pesos uniformes.
  - ``bipartite``  : bipartito completo K_{a,b} con partición por sector
                     mayoritario vs resto (fallback: mediana del puntaje
                     semilla), pesos uniformes.

El ``TopologyWrapper`` implementa el protocolo ``LocalModule`` envolviendo un
selector existente: recablea el subgrafo y delega. La interfaz con la política
queda intacta (regla 2 del pre-registro v8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.graph.subgraph_selector import Subgraph


def _renorm(W: np.ndarray) -> np.ndarray:
    mx = W.max()
    return W / (mx + 1e-12) if mx > 0 else W


def d_regular(sub: Subgraph, d: int, rng: np.random.Generator,
              uniform: bool = False) -> Subgraph:
    """Grafo d-regular sobre los nodos de ``sub`` (M·d debe ser par)."""
    M = sub.size
    if d >= M or (M * d) % 2 != 0:
        raise ValueError(f"d-regular imposible: M={M}, d={d}")
    A = sub.W_local
    # Voraz: aristas por afinidad descendente con topes de grado.
    pares = [(i, j) for i in range(M) for j in range(i + 1, M)]
    pares.sort(key=lambda p: -A[p[0], p[1]])
    deg = np.zeros(M, dtype=int)
    edges: set[tuple[int, int]] = set()
    for i, j in pares:
        if deg[i] < d and deg[j] < d:
            edges.add((i, j))
            deg[i] += 1
            deg[j] += 1
    # Reparación: intercambio de aristas hasta regularidad exacta.
    guard = 0
    while deg.min() < d and guard < 10_000:
        guard += 1
        deficit = [v for v in range(M) if deg[v] < d]
        if len(deficit) >= 2:
            u, v = deficit[0], deficit[1] if deficit[1] != deficit[0] else deficit[-1]
            if u != v and (min(u, v), max(u, v)) not in edges:
                edges.add((min(u, v), max(u, v)))
                deg[u] += 1
                deg[v] += 1
                continue
        u = deficit[0]
        # Romper una arista (a,b) con a,b de grado d y no vecinos de u.
        cand = [e for e in edges if u not in e and deg[e[0]] >= d and deg[e[1]] >= d]
        if not cand:
            raise ValueError("Reparación d-regular estancada.")
        a, b = cand[int(rng.integers(0, len(cand)))]
        edges.discard((a, b))
        deg[a] -= 1
        deg[b] -= 1
        for x in (a, b):
            key = (min(u, x), max(u, x))
            if deg[u] < d and deg[x] < d and key not in edges:
                edges.add(key)
                deg[u] += 1
                deg[x] += 1
    if deg.min() < d or deg.max() > d:
        raise ValueError("No se alcanzó regularidad exacta.")
    W = np.zeros((M, M))
    for i, j in edges:
        w = 1.0 if uniform else max(A[i, j], 1e-6)
        W[i, j] = W[j, i] = w
    if not uniform:
        W = _renorm(W)
    return Subgraph(W_local=W, global_node_ids=sub.global_node_ids,
                    seed_idx_local=sub.seed_idx_local)


def cycle(sub: Subgraph) -> Subgraph:
    """Ciclo C_M: orden vecino-más-cercano por afinidad, pesos uniformes."""
    M = sub.size
    if M < 3:
        raise ValueError("Ciclo requiere M >= 3.")
    A = sub.W_local
    orden = [int(sub.seed_idx_local)]
    restantes = set(range(M)) - {orden[0]}
    while restantes:
        actual = orden[-1]
        siguiente = max(restantes, key=lambda j: A[actual, j])
        orden.append(siguiente)
        restantes.discard(siguiente)
    W = np.zeros((M, M))
    for a, b in zip(orden, orden[1:] + orden[:1], strict=True):
        W[a, b] = W[b, a] = 1.0
    return Subgraph(W_local=W, global_node_ids=sub.global_node_ids,
                    seed_idx_local=sub.seed_idx_local)


def bipartite(sub: Subgraph, sectors: list[str] | None,
              seed_scores: np.ndarray | None = None) -> Subgraph:
    """Bipartito completo K_{a,b}: sector mayoritario vs resto.

    Fallback (sin sectores o partición degenerada): mediana del puntaje
    semilla de los nodos del subgrafo.
    """
    M = sub.size
    grupo_a: np.ndarray | None = None
    if sectors is not None:
        secs = [sectors[g] for g in sub.global_node_ids]
        valores, cuentas = np.unique(secs, return_counts=True)
        mayoritario = valores[int(np.argmax(cuentas))]
        marca = np.array([s == mayoritario for s in secs])
        if 0 < marca.sum() < M:
            grupo_a = marca
    if grupo_a is None:
        if seed_scores is None:
            raise ValueError("bipartite requiere sectores o seed_scores.")
        vals = seed_scores[sub.global_node_ids]
        grupo_a = vals >= np.median(vals)
        if grupo_a.all() or not grupo_a.any():
            grupo_a = np.zeros(M, dtype=bool)
            grupo_a[: M // 2] = True
    W = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j and grupo_a[i] != grupo_a[j]:
                W[i, j] = 1.0
    return Subgraph(W_local=W, global_node_ids=sub.global_node_ids,
                    seed_idx_local=sub.seed_idx_local)


@dataclass(slots=True)
class TopologyWrapper:
    """LocalModule que recablea H_t a una topología controlada y delega."""

    inner: object
    kind: str                      # 'dreg_aff' | 'dreg_uni' | 'cycle' | 'bipartite'
    d: int = 3
    rng_seed: int = 0
    sectors: list[str] | None = None
    seed_scores: np.ndarray | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    @property
    def name(self) -> str:
        return f"{getattr(self.inner, 'name', 'x')}_{self.kind}"

    def rewire(self, sub: Subgraph) -> Subgraph:
        if self.kind == "dreg_aff":
            return d_regular(sub, self.d, self._rng, uniform=False)
        if self.kind == "dreg_uni":
            return d_regular(sub, self.d, self._rng, uniform=True)
        if self.kind == "cycle":
            return cycle(sub)
        if self.kind == "bipartite":
            return bipartite(sub, self.sectors, self.seed_scores)
        raise ValueError(self.kind)

    def candidate_set(self, subgraph: Subgraph, k: int, m: int) -> np.ndarray:
        try:
            rewired = self.rewire(subgraph)
        except ValueError:
            rewired = subgraph  # degenerado (p.ej. M<d+1): usar BFS original
        return self.inner.candidate_set(rewired, k, m)
