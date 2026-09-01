# Context prompt

Paste this at the start of a session with any AI that needs to work on this repo.
Kept deliberately short. Everything else is in `docs/HANDOFF.md`.

---

Trabajas en `facr_portrait`, un **Portrait Retouching Engine** en Python.

**Principio del proyecto:** la IA decide qué editar, herramientas deterministas ejecutan la
edición. Este repo es solo la mitad de las herramientas (Mundo 1). El agente que decide (Mundo 2)
es un proyecto separado y **no se construye aquí**.

**Qué hay:** seis herramientas, cada una una función pura sobre arrays, con un parámetro
`strength` de 0 a 1:

- `skin_cleanup` (manchas y acné), `skin_smoothing` (piel local), `equalize_skin_tone` (color
  regional), `reduce_dark_circles` (ojeras), `whiten_teeth` (dientes),
  `balance_face_lighting` (iluminación).

Percepción con MediaPipe Face Landmarker (478 puntos, Apache-2.0) más un modelo de color de piel
por imagen. Sin modelos generativos, sin APIs externas, sin base de datos, sin autenticación.

**Reglas que no se rompen:**

1. No se modifica la geometría facial. Nada de adelgazar, agrandar ojos, cambiar nariz o labios,
   ni alterar la edad aparente. Se trabaja sobre la fotografía original.
2. **Todo píxel escrito pasa por `core.masking.apply_masked`.** Ahí viven los guardrails. Un test
   falla si una herramienta siquiera menciona `warpAffine`, `remap`, `resize` o `inpaint`. Otro
   falla si lee algo del análisis que no sea `masks` o `geometry`: eso cierra el reshaping
   fotométrico (dodge and burn reforma una cara sin mover un píxel).
3. Se preserva la textura natural de la piel. Nada de efecto plástico a fuerza moderada.
4. Nada de copiar código GPL al proyecto. Las referencias GPL se estudian, no se copian. Las
   licencias están en `docs/THIRD_PARTY.md`.
5. `strength=1.0` significa "máximo natural" y ese contrato no se ensancha.

**Método de trabajo, que importa tanto como el código:** cada constante está respaldada por una
medición registrada, nunca elegida a ojo. Antes de cambiar algo se mide el estado actual; después
se vuelve a medir. Las versiones equivocadas que se probaron primero quedan documentadas en
`docs/ALGORITHMS.md`, que es el documento de mayor valor del repo. Si un cambio mejora un número,
hay que comprobar que no empeora otro.

**Estado:** piel y ojeras están a nivel profesional y medidas. Dientes, iluminación y limpieza
funcionan y están testeadas, sin el mismo pulido. `229 passed`, ruff limpio.

**Antes de tocar nada:** lee `docs/HANDOFF.md`. Tiene el estado, las decisiones abiertas, las
trampas de medición conocidas y los límites reales del motor. Luego `docs/ALGORITHMS.md` para las
fórmulas y sus calibraciones.

**Práctico:** el intérprete es `.venv/bin/python` (no el `python` del sistema). Tests con
`.venv/bin/python -m pytest`. No escribas entregables bajo `data/`, está en gitignore.
