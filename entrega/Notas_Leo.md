# NOTES — Predicción de diseños "ganadores"

self_reported_validation_auc: 0.51

Autor: Leonardo González

Lo primero que hice fue ordenar el lago en una tabla por diseño (capa silver):
deduplicar `pipeline_designs` (2426 filas para 2400 diseños), parsear las fechas
que venían en dos formatos y normalizar las categóricas, porque el mismo valor
aparecía escrito de varias formas (`GUITAR`/`guitar`, `electric_guitar`/`electric
guitar`) e inflaba la cardinalidad.

Lo importante de esta prueba no era el modelo, era decidir qué entra y qué no.
Me topé con dos features que dan un AUC casi perfecto (~0.99) y a las dos las
descarté a propósito. Las ventas (`product_sales`) son fuga obvia: solo existen
después de lanzar, así que no las tengo cuando hay que decidir. El gate de
aprobación (`approval_agent_shadows`) es más sutil: parece legítimo porque ocurre
antes, pero es circular —un diseño rechazado nunca llega al mercado, así que no
puede ser ganador (0 de 150 rechazos en holdout lo son). Usarlo es repetir una
decisión que el negocio ya toma; no aporta nada accionable. También dejé fuera
`stage`, `products`, `ad_comments` y los metadatos de la etiqueta, por la misma
lógica.

Me quedé solo con lo intrínseco del diseño (atributos, brief, parámetros de
generación). Y aquí está lo interesante: en validación interna esto daba ~0.74,
pero al medirlo contra los diseños nuevos cae a ~0.49, es decir, azar. Lo
comprobé y no es un bug ni fuga de grupos: es que las relaciones atributo→ganador
del pasado no se sostienen en los diseños nuevos (las tasas cambian de signo
entre una muestra y otra). Dicho claro: con estos datos, acertar el ganador es
prácticamente imposible.

Por eso reporto 0.51 y no el 0.74. Prefiero un número honesto a uno inflado por
una validación que no ve ese cambio de distribución. El modelo es un gradient
boosting sencillo; con señal de azar, afinarlo no cambia la historia.

¿Siguiente paso? La señal real probablemente está en la imagen del diseño y en la
demanda de mercado, que este lago no captura. Ahí metería embeddings de la imagen
antes que cualquier otra cosa.

Para reproducir: `pip install -r requirements.txt` y luego
`python3 src/entrenar.py` (y `src/diagnostico_fuga.py` para ver la evidencia de
las fugas).
