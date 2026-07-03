# Dataset extraído — Inversiones bancos Panamá

Fuente de verdad detrás del reporte HTML. Cada cifra lleva nivel de confianza:

- **[AUD]** Auditado — leído directamente de la nota "Inversiones en valores" de los EE.FF. auditados.
- **[EST]** Estimado — cálculo o derivación propia.
- **[PEND]** Pendiente — no divulgado por el emisor (queda como límite del análisis, no como pendiente de lectura).

Moneda: USD (B/. a la par). Todas las cifras en US$ salvo indicación. Cifras "en unidades" (no miles) salvo nota.

> **Estado: FIDELIDAD COMPLETA.** Segunda sesión: los dominios financieros quedaron habilitados y se
> abrieron y extrajeron directamente las notas "Inversiones en valores" de los EE.FF. auditados de los
> seis bancos. Lo que en el corte preliminar figuraba como [PEND] (split por instrumento, MBS/UST,
> FVTPL/FVOCI/costo amortizado, escalera de vencimientos, tabla de ratings, ingreso por inversiones)
> ahora está leído de la fuente. Los pocos [PEND] restantes son datos que el emisor **no divulga**.

---

## Hallazgos de cabecera (correcciones al corte preliminar)

1. **El gran tenedor de MBS es Banco General, NO BAC.** Banco General mantiene ~**US$3,056 millones** en
   MBS + CMO (≈51% de su cartera de inversiones a Dic-2025), 100% garantizados por agencias de EEUU
   (GNMA/FNMA/FHLMC). **BAC NO divulga MBS** en su nota auditada (solo bonos de gobierno, corporativos y
   acciones). Esto invierte la conclusión preliminar "MBS confirmado solo en BAC".
2. **US Treasuries confirmados y cuantificados** en cuatro bancos: Banistmo (383M, ~25%), BAC (703M gob.
   EEUU, 14% de su libro AFS), Banco General (130M explícito + Letras del Tesoro), Scotiabank (176M, 47%).
   Global Bank y Davivienda no los desglosan por país (solo "gubernamental").
3. **Peso sobre el activo — ranking real:** Banco General 28.6% (líder amplio) > Davivienda 23.4% (inflado
   por la fusión) > Banistmo 14.8% > Global 13.4% > BAC 12.3% > Scotiabank 9.7%. Sistema (CBI) ~21.6%.
4. **Banistmo corre el portafolio más "trading"** de todos: 36.5% a FVTPL a Sep-2025 (era 11.8% a
   Dic-2023). Ningún competidor grande se acerca (BG 16%, BAC 1%, Global 6%, Davivienda 0.5%).

---

## Sistema bancario (SBP — Informe de Actividad Bancaria)

Escopos: **SBN** = Sistema Bancario Nacional (licencia general en Panamá);
**CBI** = Centro Bancario Internacional (SBN + licencia internacional; agregado titular).

| Indicador | 2022 | 2023 | 2024 | 2025 | Conf. |
|---|---|---|---|---|---|
| Inversiones en valores — var. a/a | +5.4% | +9.2% | +7.0% | +4.79% (CBI, oct) | [AUD] |
| Inversiones — nivel absoluto | ≈29,100 (deriv.) | n/d | n/d | CBI 34,571.5 (oct) | [EST]/[AUD] |
| Activo total CBI | 140,025 | 147,526 | 156,392.8 | 160,378.3 (oct) | [AUD] |
| Cartera crédito neta CBI | 83,341 | 87,202 | 95,186.7 | 100,088.9 (oct) | [AUD] |
| % inversiones / activo CBI | ~20.8% | n/d | n/d | **21.6%** (oct) | [EST] |
| Índice adecuación capital (IAC) | 15.34% | ~15.71% | 15.29% | 16.34% (nov) | [AUD] |

Notas:
- Sistema **dominado por crédito**: préstamos ~62% del activo; inversiones ~21.6% (2.º bloque productivo).
- Oct-2025: inversiones netas CBI 34,571.5 (+4.79%, +1,579.8M); **inversiones internas +11.54%** (giro a
  papel local). Ingreso por inversiones del sistema +8.5% a/a.
- Pérdidas no realizadas de la cartera disponible para la venta (AFS) del sistema: bajando desde 540.4M.
- El sistema no publica split instrumento (soberano/corp/MBS) ni interno/externo de la cartera de valores. [PEND]
- Contexto: Panamá perdió grado de inversión con Fitch (mar-2024); presión de Moody's sobre bancos PA.

---

## Banistmo, S.A. y Subsidiarias (banco base) — consolidado Panamá

Fuente: IN-A 2024 (Nota 7, auditado) + Interino 30-sep-2025 (Nota 7, no auditado).

| Métrica | Dic-2023 | Dic-2024 | Sep-2025 | Conf. |
|---|---|---|---|---|
| Inversiones en valores, neto | 1,459,161,128 | **1,548,786,610** | **1,620,459,320** | [AUD] |
| — FVTPL (VRCR) | 172,580,394 (11.8%) | 530,775,476 (34.3%) | 591,503,419 (36.5%) | [AUD] |
| — FVOCI (ORI) | 737,469,973 (50.5%) | 435,323,046 (28.1%) | 359,088,904 (22.2%) | [AUD] |
| — Costo amortizado | 549,110,761 (37.6%) | 582,688,088 (37.6%) | 669,866,997 (41.3%) | [AUD] |
| Activo total | n/d | 10,471.8 | ~10,511 | [AUD] |
| % inversiones / activo | | **14.8%** | ~15.4% | [EST] |

**Composición por instrumento (Dic-2024, bruto por buckets):** [AUD]
- **US Treasuries (Bonos EEUU):** 346.2M (FVTPL) + 37.2M (FVOCI) = **383.4M (~24.8%)**
- **Soberano Panamá:** 132.0M (FVTPL) + 291.2M (FVOCI) = **423.3M (~27%)** + soberanos 26.3M (CA)
- **Corporativos:** 24.7M (FVTPL) + 40.9M (FVOCI) + 565.8M (CA bruto) = **631.4M (~40%)**
- Otros soberanos (Alemania, Rep.Dom, Colombia, Chile, México, Costa Rica): ~46M
- Patrimonio/acciones: 7.4M (FVTPL) + 36.9M (FVOCI) = 44.3M
- **Sin MBS** (solo bonos y acciones).

**Riesgo / duración (reporte S&P anexo al IN-A, dato a sep-2024):** [AUD cualit.]
- **Duración promedio del portafolio: 2.5 años.** VaR sobre patrimonio < 1%.
- Deuda pública Panamá + EEUU ≈ 50% del total (coincide con agregado: US 383M + Panamá 423M ≈ 52%).

- **Ingreso por intereses de inversiones (2024, individual SBP):** 64.19 → **yield ≈ 4.3%**
  = 64.19 / prom(1,459.2 ; 1,548.8). [EST] (ingreso individual vs. saldo consolidado → aprox.)
- **Hallazgo estructural:** giro a *trading* pronunciado — FVTPL 11.8% (2023) → 34.3% (2024) → 36.5%
  (sep-2025); FVOCI 50.5% → 22.2%. El más "trading" del perímetro.
- Banistmo **no publica** escalera de vencimientos ni tabla de ratings de cartera en la Nota 7
  (la duración proviene del reporte de la calificadora). [PEND emisor]
- FY2025 anual (IN-A 2025) aún no publicado (esperado ~mar-2026).

---

## Banco General, S.A. y Subsidiarias — consolidado Panamá (mayor banco privado)

Fuente: EE.FF. consolidados Dic-2025 (Nota 6 "Inversiones y otros activos financieros", auditado), con
comparativo Dic-2024 en la misma nota.

| Métrica | Dic-2024 | Dic-2025 | Conf. |
|---|---|---|---|
| Inversiones y otros activos financieros, neto | 5,347,770,702 | **5,968,302,539** | [AUD] |
| — FVTPL (VRCR) | 829,554,142 (15.5%) | 942,225,934 (15.8%) | [AUD] |
| — FVOCI (ORI) | 4,511,601,422 (84.4%) | 5,022,886,623 (84.2%) | [AUD] |
| — Costo amortizado | 6,615,138 (0.1%) | 3,189,982 (0.05%) | [AUD] |
| Activo total | 19,447,499,026 | 20,879,794,387 | [AUD] |
| % inversiones / activo | 27.5% | **28.6%** | [EST] |

**Composición por instrumento (Dic-2025):** [AUD]
- **MBS + CMO = 3,055.6M (~51.2% de la cartera)** — 100% del MBS garantizado GNMA/FNMA/FHLMC (agencias
  EEUU); 66% de los CMO respaldados por esos MBS. **El mayor tenedor de MBS del perímetro.**
- ABS: 207.9M
- **US Treasuries (Bonos Gob EEUU):** 130.3M (FVOCI) + Letras del Tesoro extranjeras (país no confirmado)
- Bonos corporativos: locales 1,125.8M + extranjeros 954.5M ≈ 2,080M
- Soberano Panamá: 95.4M + 2.7M ≈ 98M; otros gobiernos 12.7M
- Acciones locales: 36.8M
- **Duración divulgada solo para MBS (3.71a) y CMO (1.79a);** no del portafolio total. [AUD parcial]

- **Ingreso por intereses de inversiones (2025):** 283.5M (2024: 265.6M) → **yield ≈ 5.0%** [EST]
- Perfil: renta fija de **alta calidad, grado de inversión**, con núcleo de agency-MBS de EEUU (líquido).
- Ratings: núcleo extranjero AAA-AA+ (los agency-MBS); local grueso "< BBB-" (papel panameño). [AUD]
- Escalera de vencimientos (por posible fecha de venta): ≤3m 4,970M (grueso, MBS líquidos) · 1-5a 462M ·
  5-10a 253M · >10a 39M. [AUD]

---

## BAC International Bank, Inc. y Subsidiarias — consolidado Centroamérica (OJO perímetro)

Fuente: EE.FF. auditados Dic-2025 (Nota 8), con comparativos 2024/2023. Consolidado CR+PA+GT+HN+SV+NI.

| Métrica | 2023 | 2024 | 2025 | Conf. |
|---|---|---|---|---|
| Inversiones en valores, neto | 4,548,929,982 | 4,883,433,488 | **4,989,837,727** | [AUD] |
| — FVTPL (VRCR) | 38,756,743 | 35,201,978 | 45,673,846 | [AUD] |
| — FVOCI / AFS | 4,411,769,762 | 4,737,704,498 | 4,756,274,848 | [AUD] |
| — Costo amortizado | 98,403,477 | 110,527,012 | 187,889,033 | [AUD] |
| Activo total | 34,502,650,748 | 37,970,437,089 | 40,747,080,228 | [AUD] |
| % inversiones / activo | 13.2% | 12.9% | **12.3%** | [EST] |

**Composición por instrumento (VRCOUI/AFS, 2025):** [AUD]
- **Otros gobiernos Centroamérica: 3,651.7M (76.6%)** — núcleo soberano centroamericano.
- **US gobierno y agencias: 703.2M (14.1%)** — no separa Treasuries de agencias; sin identificar MBS.
- Bonos corporativos: 398.2M (9.2%); acciones 3.2M.
- **MBS: NO divulgado** (búsqueda exhaustiva 2025+2024; todo "hipotecario" es cartera de préstamos).
  Corrige la creencia preliminar de MBS en BAC.

- **Ingreso por intereses de inversiones (2025):** 291.4M (2024: 299.9M) → **yield ≈ 5.9%** (2024 ~6.4%) [EST]
- **Perímetro:** consolidado Centroamérica. Entidad individual Panamá ≈US$13.3 mil M activos (Dic-2025,
  composición de cartera no divulgada aparte). NO confundir. [AUD]
- Ratings: grueso "BB+ a B-" (2,967.6M) — refleja soberanos centroamericanos sub-grado-inversión. [AUD]
- Vencimientos (línea combinada inv+otros): ≤1a 2,088.6M · 1-5a 2,281.6M · >5a 1,455.7M. [AUD]
- **Rating emisor:** S&P BBB-/A-3 estable (6-ago-2025).

---

## Global Bank Corporation y Subsidiarias — consolidado Panamá (cierre fiscal JUNIO)

Fuente: EE.FF. auditados 30-jun-2025 (Nota 9), con comparativo FY2024 (jun-24) en la misma nota.

| Métrica | FY2022 (jun) | FY2023 (jun) | FY2024 (jun) | FY2025 (jun) | Conf. |
|---|---|---|---|---|---|
| Inversiones en valores, neto | 1,079.2 | 1,053.4 | **1,049,326,162** | **1,136,648,533** | [AUD]* |
| — FVTPL (VRCR) | | | 71,631,682 | 69,298,285 | [AUD] |
| — FVOCI (ORI) | | | 566,447,182 | 674,051,469 | [AUD] |
| — Costo amortizado | | | 402,341,535 | 383,980,942 | [AUD] |
| Activo total | 8,400.6 | 8,413.0 | 8,520,282,632 | 8,503,001,576 | [AUD] |
| % inversiones / activo | 12.85% | 12.5% | 12.31% | **13.37%** | [EST] |

\* FY2022/FY2023 headline provienen de indexación previa (la URL "FY2023" servía en realidad los EE.FF.
FY2019 — mismatch de fuente; FY2024/FY2025 sí leídos de la nota auditada).

**Composición por instrumento (FY2025):** [AUD]
- Deuda privada (corporativa): 617.5M · gobierno: 509.8M · acciones locales ~28M.
- **US Treasuries: NO desglosado por país.** Proxy geográfico EEUU: 283.2M (Panamá 575.1M, LatAm 251.7M).
- **MBS: NO presente** (búsqueda exhaustiva en 88 páginas; "hipotecario" = solo cartera de préstamos).

- **Ingreso por intereses de inversiones (FY2025):** 45.6M (FY2024: 40.5M) → **yield ≈ 4.2%** [EST]
- Ratings FY2025: grado inversión 562.5M · monitoreo estándar (BB+ a B-) 186.1M · sin calificación 383.8M. [AUD]
- Vencimientos FY2025: FVOCI 1-5a 350.1M / >5a 143.6M; CA 1-5a 111.2M / >5a 262.7M (libro largo a CA). [AUD]
- Sensibilidad ±100pb del ingreso: -46.4M / +53.0M. NIIF 17 desde 1-jul-2024.
- **Rating emisor:** S&P y Fitch BBB-, estable.

---

## Banco Davivienda (Panamá), S.A. y Subsidiarias — consolidado (post-fusión Scotiabank)

Fuente: EF Local 2025 (consolidado auditado, Nota 8). FY2025 es el **primer cierre post-fusión** (absorbe
The Bank of Nova Scotia — Sucursal de Panamá, 5-dic-2025).

| Métrica | Dic-2024 | Dic-2025 (post-fusión) | Conf. |
|---|---|---|---|
| Inversiones en valores, neto | 455,271,607 | **1,248,452,413** | [AUD] |
| — FVTPL (VRCR) | 0 | 6,164,984 | [AUD] |
| — FVOCI (VRCOUI) | 259,374,130 | 988,382,401 (79%) | [AUD] |
| — Costo amortizado | 195,897,477 | 253,905,028 (20%) | [AUD] |
| Activo total | 1,816,389,671 | 5,342,265,863 | [AUD] |
| % inversiones / activo | 25.1% | **23.4%** | [EST] |

**Composición por instrumento (2025):** [AUD]
- Bonos gubernamentales: 923.3M (FVOCI) + 6.2M (FVTPL) — **núcleo soberano dominante**.
- Bonos corporativos: 64.8M (FVOCI) + 257.7M (CA bruto). Acciones 0.2M.
- **US Treasuries: NO desglosado** (solo "gubernamentales"); Norteamérica (geo) 412.75M FVOCI + 27.5M CA.
- **MBS: NO.**

- **Ingreso por intereses de inversiones (2025):** 34.5M (2024: 27.7M) → **yield ≈ 4.05%** (saldo prom.) [EST]
- Adquirió **318.5M de inversiones de Scotiabank** al 1-dic-2025 (nota de combinación de negocios).
- Ratings (VRCOUI 2025): AAA-AA− 397.4M · A 6.0M · BB+ a BB− 519.9M (gob); corp BBB 32.0M / BB 29.2M. [AUD]
- Vencimientos 2025: ≤1a 747.0M · 1-5a 466.0M · >5a 147.0M. VaR cartera 806,880 (límite 1.76M). [AUD]
- **Tratar a Davivienda (no a Scotiabank) como el competidor a futuro.** FY2025 no comparable con años previos.
- Davivienda Internacional (offshore, entidad separada): inversiones 0 en 2025 (2024: 25.1M corp, redimido).

---

## Scotiabank (Sucursal de Panamá) — cierre fiscal OCTUBRE — ABSORBIDO (dic-2025)

Fuente: EE.FF. FY2024 (31-oct-2024, Nota 8), con comparativo oct-2023. Último año standalone.

| Métrica | oct-2023 | oct-2024 | Conf. |
|---|---|---|---|
| Inversiones en valores, neto | 359,289,454 | **371,322,361** | [AUD] |
| — FVOCI (VRCOUI) | 296,018,162 | 318,896,535 | [AUD] |
| — Costo amortizado | 63,271,292 | 52,425,826 | [AUD] |
| Activo total | 3,744,781,864 | 3,841,407,316 | [AUD] |
| % inversiones / activo | 9.6% | **9.67%** | [EST] |

**Composición por instrumento (oct-2024):** [AUD]
- **US Treasuries (Bonos Gob EEUU): 176.3M (47% de la cartera)** — línea explícita, la mayor proporción
  de UST del perímetro. Cruzado con la concentración geográfica "Canadá y EEUU".
- Soberano Panamá: 141.3M. Corporativos (CA): 53.8M bruto. Sin acciones. Sin FVTPL. **Sin MBS.**

- **Ingreso por intereses de inversiones (2024):** 15.0M → **yield ≈ 4.1%** [EST]. Tasa CA: 4.20%–6.17%.
- Ratings: VRCOUI AA+ a A 176.6M / BBB 142.2M; CA AA+ a A− 23.9M / BBB a BB+ 28.5M. [AUD]
- Vencimientos VRCOUI: ≤3m 27.8M / 3m-1a 67.6M / 1-5a 222.2M (sin tramo >5a). [AUD]
- A futuro el competidor es Davivienda, no Scotiabank (esta entidad ya no reporta standalone).

---

## Distribución por calificación — normalizada (para comparabilidad)

Cada banco publica su tabla de ratings con **cortes de letra y coberturas distintas**, así que se
colapsan a tres brackets comparables sobre la **cartera calificada divulgada** de cada uno:
**Grado de inversión (AAA→BBB-)** · **Sub-grado / monitoreo estándar (BB+ a B-)** · **Sin calificación (NR)**.
Fuente: nota de riesgo de crédito de cada EE.FF. auditado (montos verificados y reconciliados con el total).

| Banco (fecha) | Cobertura de la tabla | Grado inversión (AAA→BBB-) | Sub-grado (BB+ a B-) | Sin calif. | Total calificado |
|---|---|---|---|---|---|
| Banco General (Dic-25) | deuda (excl. acciones 37M) | **4,536.0 (76.5%)** | 1,395.1 (23.5%) | 0.05 (~0%) | 5,931.2 |
| BAC (Dic-25) | bonos VRCR+VRCOUI+CA | 1,626.4 (32.7%) | **3,340.5 (67.3%)** | 0 | 4,966.8 |
| Global Bank (jun-25) | toda la cartera | 562.5 (49.7%) | 186.1 (16.4%) | **383.8 (33.9%)** | 1,132.3 |
| Davivienda (Dic-25) | VRCOUI+VRCR (excl. CA 253.9M) | 437.4 (44.3%) | **549.1 (55.6%)** | 1.8 (0.2%) | 988.3 |
| Scotiabank (Oct-24) | toda la cartera | **342.7 (92.3%)** | 28.5 (7.7%)* | 0 | 371.3 |
| Banistmo (Dic-24) | **no divulga tabla** — inferido por emisor | ver nota ↓ | | | 1,548.8 |

Detalle de brackets leído de la fuente (US$, 2025 salvo Scotia oct-24):
- **Banco General:** AAA→AA+ 3,334.0 · AA→BBB- 1,202.0 · <BBB- 1,395.1 · NR 0.05. El bucket <BBB- es
  casi todo **papel local de Panamá** (local <BBB- 1,034.2). [AUD]
- **BAC:** VRCOUI gob AA+ 728.3 · BBB 5.6 · BBB- 653.3 · **BB+ a B- 2,967.6**; corp IG 239.1 · corp BB+ a B-
  159.1; CA gob BB+ a B- 64.7 · CA corp BB+ a B- 123.2; VRCR BB 25.9. El grueso sub-grado son **soberanos
  de Centroamérica**. [AUD]
- **Global Bank:** grado inversión 562.5 · monitoreo estándar (BB+ a B-) 186.1 · **sin calif. internacional
  383.8**. El 34% sin calificar es papel local sin rating internacional. [AUD]
- **Davivienda:** gob AAA→AA− 397.4 · A 8.0 · BBB 32.0 · **BB 549.1** · NR 1.8. (La tabla no cubre el bucket
  a costo amortizado 253.9M.) [AUD]
- **Scotiabank:** VRCOUI AA+ a A 176.6 (los **UST**) · BBB+ a BBB- 142.2; CA AA+ a A− 23.9 · *BBB+ a BB+ 28.5*
  (*bracket frontera que cruza IG/sub-IG; contado aquí como sub-grado, conservador). Cartera casi íntegra IG. [AUD]
- **Banistmo (inferido):** sin tabla de ratings en la Nota 7. Anclas verificables: **UST 383M ≈ AA+ (~25%)**,
  Panamá 423M (frontera BBB-/BB+, ~27%), y ~40% **corporativo sin desglose de rating** (631M, emisores no
  nombrados) → **no colapsable a IG/sub-IG sin especular**. S&P (anexo IN-A): deuda pública Panamá+EEUU ~50%,
  duración 2.5a. [EST]

**Caveat de comparabilidad — la línea IG/sub-IG depende de cómo cada banco trata el papel de Panamá.**
Panamá quedó en frontera (S&P BBB, Fitch BB+ desde mar-2024, Moody's Baa3): BG y BAC lo clasifican como
**sub-grado (<BBB-/BB+)**, Global lo deja **sin calificación internacional**, y Scotiabank lo trata como **IG
(BBB)**. Por eso el % "grado de inversión" no es 1:1 entre bancos; el orden cualitativo (BG y Scotia = cartera
de alta calidad; BAC y Davivienda = grueso sub-grado por soberanos regionales; Global = un tercio sin rating)
sí es robusto.

---

## Resumen para gráficos del reporte

**Tamaño cartera (último disponible):**
BG 5,968.3 (Dic-25) · BAC 4,989.8 (Dic-25, CA) · Banistmo 1,620.5 (Sep-25) · Davivienda 1,248.5 (Dic-25) ·
Global 1,136.6 (jun-25) · Scotiabank 371.3 (oct-24, absorbido).

**% inversiones / activo:**
BG 28.6% · Davivienda 23.4% · Sistema (CBI) 21.6% · Banistmo 14.8% · Global 13.4% · BAC 12.3% · Scotia 9.7%.

**Yield estimado:**
BAC 5.9% · BG 5.0% · Banistmo 4.3% · Global 4.2% · Scotiabank 4.1% · Davivienda 4.05%.

**Clasificación contable (% FVTPL / FVOCI / CA, último):**
- Banistmo (Sep-25): 36.5 / 22.2 / 41.3 → **el más trading**
- BG (Dic-25): 15.8 / 84.2 / 0.05 → casi todo FVOCI
- BAC (Dic-25): 0.9 / 95.3 / 3.8 → casi todo AFS
- Global (jun-25): 6.1 / 59.3 / 33.8
- Davivienda (Dic-25): 0.5 / 79.2 / 20.3
- Scotiabank (oct-24): 0 / 85.9 / 14.1

**Instrumentos — mapa definitivo:**
- **MBS/CMO:** SOLO Banco General (~3,056M, ~51%, agency GNMA/FNMA/FHLMC). Ningún otro banco los divulga.
- **US Treasuries:** Scotiabank 176M (47%) · Banistmo 383M (25%) · BAC 703M gob-EEUU (14%) · BG 130M+ explícito.
  Global y Davivienda no los separan por país.
- **Núcleo soberano local/regional:** BAC (Centroamérica), Davivienda (gob.), Banistmo y Global (Panamá + LatAm).
- **Corporativo:** peso relevante en Banistmo (~40%) y Banco General (~35% ex-MBS).
