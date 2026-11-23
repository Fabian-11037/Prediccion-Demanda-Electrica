# Estabilidad de la corrección con banco y vecinos fijos

Sean q una consulta, k_i claves fijas, r_i residuos normalizados, s_i=qᵀk_i/√32 y α=softmax(s) sobre los k vecinos admisibles. La corrección normalizada es c(q)=g(q)∑α_i r_i. La fecha, banco, base y parámetros se mantienen fijos. Supóngase ||r_i||∞≤R y ||k_i||₂/√32≤K. Si la separación entre el último vecino seleccionado y el primero descartado es m>0, cualquier perturbación δ con2K||δ||₂<m preserva el conjunto de vecinos. Esta condición no cubre empates ni cambios de disponibilidad.

En esa región, ||Δs||∞≤K||δ||₂. El jacobiano de softmax implica ||Δα||₁≤2||Δs||∞; por tanto ||Δr̄||∞≤2RK||δ||₂. Los residuos son fijos; la identidad Var(r)=E(r²)−E(r)² y el acotamiento anterior dan |ΔVar|≤3R²||Δα||₁ para cada componente y también para su promedio sobre24 horas. La operación √max(v,ε), con ε=10⁻¹², es Lipschitz con constante1/(2√ε), de modo que |Δdispersión|≤3R²||Δα||₁/(2√ε).

El apoyo es constante mientras los vecinos admisibles sean fijos. La antigüedad normalizada cambia a lo sumo ||Δα||₁ y el máximo de similitud a lo sumo K||δ||₂. En la norma euclídea, una cota conservadora para el cambio del vector de la compuerta es

L_z=1+K+2K(1+3R²/(2√ε)).

En evaluación, Dropout es identidad. Para la compuerta sigmoide de dos capas, L_g≤(1/4)·1,13·||W₂||₂||W₁||₂, usando la cota de derivada de GELU. Como0≤g≤1 y ||r̄||∞≤R,

||c(q+δ)−c(q)||∞≤[2RK+R·L_g·L_z]||δ||₂.

La predicción en unidades originales tiene la misma cota multiplicada por la escala MASE del dataset, con la base fija. Si también varía la base, se suma su propio cambio. La composición con el encoder numérico introduce su constante Lipschitz, finita con la epsilon positiva de LayerNorm y pesos finitos; la propiedad anterior se expresa directamente sobre consultas para evitar confundir ambas normas.

La cota puede ser muy holgada, especialmente por el piso ε. No es una garantía de precisión, de no degradación ni de estabilidad global. Una inserción del banco, un cruce del umbral top-k o un empate pueden cambiar discontinuamente los vecinos. El rechazo de equivalencia observado en empates no contradice esta propiedad, porque su hipótesis m>0 deja de cumplirse. Sin vecinos, la identidad con la base se verifica por separado.
