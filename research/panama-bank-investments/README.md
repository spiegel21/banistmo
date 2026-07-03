# Radar de Inversiones — Banca de Panamá

Análisis competitivo del portafolio de **inversiones en valores** de los principales
bancos de Panamá, con **Banistmo** como banco base de comparación. Enfoque de tesorería:
cómo están invertidos los competidores frente a Banistmo (tamaño, peso sobre activo,
tipo de instrumento, clasificación contable, rentabilidad estimada y calificaciones),
para 2022–2026.

**Perímetro:** Banistmo · Banco General · BAC (incl. Multibank) · Global Bank · Davivienda · Scotiabank.

## Contenido de esta carpeta

| Archivo | Qué es |
|---|---|
| `inversiones-bancos-panama.html` | Reporte visual (artifact) en español. Autocontenido: abrir en cualquier navegador. |
| `datos-extraidos.md` | **Dataset completo** con cada cifra, su nivel de confianza (Auditado / Estimado / Pendiente) y su fuente. Es la fuente de verdad detrás del HTML. |
| `fuentes-pdfs.md` | URLs de los EE.FF. auditados por banco, para la extracción a fidelidad completa. |
| `allowlist-domains.txt` | Dominios a habilitar en la política de red del entorno (ver más abajo). |

El reporte publicado como artifact vive en:
`https://claude.ai/code/artifact/8214a9e7-a100-446c-a1a8-a1c6803793ad`
(redeploy sobre la misma URL editando el HTML y volviendo a publicar).

## Estado: FIDELIDAD COMPLETA

Segunda sesión (jul-2026): con los dominios financieros habilitados en la política de red,
se **abrieron y leyeron directamente** las notas "Inversiones en valores" de los EE.FF.
auditados de los seis bancos. Todo lo que en el corte preliminar figuraba como *Pendiente*
—split FVTPL/FVOCI/costo amortizado, composición por instrumento (incl. MBS y Tesoros de
EE.UU.), escalera de vencimientos, tablas de calificaciones de cartera e ingreso por
inversiones para el yield— está ahora leído de la fuente. `datos-extraidos.md` y el HTML se
actualizaron y el artifact se redeployó sobre la misma URL.

### Correcciones de cabecera respecto al corte preliminar

- **El gran tenedor de MBS es Banco General, no BAC.** BG mantiene ~US$3,056 M en MBS+CMO
  (~51% de su cartera), 100% de agencias de EE.UU. (GNMA/FNMA/FHLMC). **BAC no divulga MBS**
  en su nota auditada. Invierte la conclusión preliminar "MBS confirmado solo en BAC".
- **Tesoros de EE.UU. confirmados y cuantificados** en Banistmo (US$383 M, ~25%), Scotiabank
  (US$176 M, 47%), BAC (US$703 M gob-EEUU, 14%) y Banco General (US$130 M explícito). Global
  y Davivienda no los separan por país.
- **Peso sobre activo — ranking real:** BG 28.6% > Davivienda 23.4% (post-fusión) > sistema
  21.6% > Banistmo 14.8% > Global 13.4% > BAC 12.3% > Scotiabank 9.7%.
- **Banistmo corre el libro más "trading"** del grupo: 36.5% a FVTPL a Sep-2025.

### Datos que siguen [No divulgado] (límite del emisor, no de acceso)

- Global Bank y Davivienda no desglosan los Tesoros de EE.UU. por país (solo "gubernamental").
- Banistmo no publica tabla de ratings de cartera en su Nota 7 (la duración ~2.5 años viene
  del reporte de S&P anexo al IN-A 2024).

### Notas de fuente descubiertas

- La URL de "Global Bank FY2023 auditado" en `fuentes-pdfs.md` en realidad sirve los EE.FF.
  **FY2019** (mismatch en el CDN de la web del banco). Los FY2024/FY2025 sí se leyeron del
  PDF auditado jun-2025; los headline FY2022/FY2023 se conservan de la indexación previa.
- `globalbank.com.pa` está detrás de un WAF (Incapsula) que a veces devuelve una página de
  bloqueo en vez del PDF; reintentar suele resolverlo.

## Metodología

- Moneda: USD (balboa a la par).
- Rendimiento de cartera estimado como `ingreso por intereses de inversiones ÷ saldo promedio`
  (los bancos rara vez lo publican).
- **Comparabilidad:** los perímetros de reporte difieren y **no son 1:1**:
  - **BAC** = consolidado **Centroamérica** (~US$38 mil M activos), no solo Panamá. Entidad
    individual de Panamá ~US$13.3 mil M activos (Dic-2025), composición de cartera no divulgada aparte.
  - **Banco General, Banistmo, Global Bank** = consolidados con base en **Panamá**.
  - **Global Bank** cierra fiscal en **junio**, no en diciembre.
  - **Scotiabank** reporta a **octubre**; ya fue absorbido por Davivienda (dic-2025).
  - Por eso el análisis se apoya en **ratios** (% de activo, yield), robustos al perímetro.

## Contexto estructural clave (verificado)

- **BAC ◄ Multibank:** compra mayoritaria de Multi Financial Group cerrada mar-2026; fusión
  operativa completada **11-jun-2026**. Todos los EE.FF. 2022–2025 son BAC standalone. BAC
  pasa a ser el 2.º banco más grande de Panamá.
- **Davivienda ◄ Scotiabank Panamá:** acuerdo ene-2025 (Scotia recibe ~20% de Davivienda
  Group); SBP aprueba ago-2025 (Resolución SBP.BAN-R-2026-00506, ~75% de activos/pasivos);
  integración legal en Panamá **5-dic-2025** (cierre global 1-dic-2025). Panamá fue el primer
  país integrado: una sola entidad marca Davivienda. Absorbe ~US$3.8 mil M de Scotia → activos
  de Davivienda Panamá +~180%. FY2025 no comparable con años previos. **Tratar a Davivienda
  Panamá (no a Scotiabank) como el competidor a futuro.**
- **Banistmo cambia de dueño:** Grupo Cibest/Bancolombia lo vende a **Inversiones Cuscatlán**
  (Grupo Financiero BSC / Banco La Hipotecaria), cierre ~1-jul-2026; se mantiene la marca.
