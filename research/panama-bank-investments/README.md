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

## Estado: CORTE PRELIMINAR

Este es un primer corte. La política de red de este entorno **bloqueó el acceso directo
a los PDF** de las notas "Inversiones en valores" de los EE.FF. auditados (denegación
403 de la organización sobre todos los dominios financieros de Panamá + WAF de origen).
Las cifras provienen de la **indexación** de esos documentos vía búsqueda web y de fuentes
secundarias verificadas de forma cruzada — no de la lectura directa de las tablas.

Lo que quedó **sólido**: tamaños de cartera, % sobre activo, dirección de la mezcla de
instrumentos, hechos de fusión/adquisición, tendencias de sistema.

Lo que quedó **pendiente** (requiere abrir los PDF): distribución completa por instrumento,
tabla de calificaciones de la cartera, escalera de vencimientos / duración, y el desglose
FVTPL / FVOCI / costo amortizado de los competidores.

## Cómo llegar a fidelidad completa (para la próxima sesión)

1. **Habilitar los dominios** de `allowlist-domains.txt` en la política de red del entorno:
   claude.ai/code → icono de nube → editar entorno → **Network access** → **Custom** →
   pegar la lista en **Allowed domains** → marcar "Also include default list of common
   package managers" → guardar. (Si el selector está bloqueado en un plan Enterprise/Team,
   lo debe habilitar un admin del workspace.)
2. **Iniciar una NUEVA sesión** en ese entorno (la política de red se fija al arrancar la
   sesión; la sesión actual no toma el cambio).
3. **Prompt sugerido para la nueva sesión:**

   > Retoma el trabajo en `research/panama-bank-investments/`. Los dominios financieros ya
   > están habilitados en la política de red. Abre los PDF listados en `fuentes-pdfs.md`,
   > extrae de la nota "Inversiones en valores" de cada banco lo que quedó marcado como
   > *Pendiente* en `datos-extraidos.md` (composición por instrumento incl. MBS y US
   > Treasuries, desglose FVTPL/FVOCI/costo amortizado, escalera de vencimientos/duración,
   > tabla de calificaciones de la cartera, e ingreso por intereses de inversiones para el
   > yield estimado), actualiza `datos-extraidos.md` y el HTML, y redeploya el artifact sobre
   > la misma URL.

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
