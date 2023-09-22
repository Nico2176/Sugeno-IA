# Importa las bibliotecas necesarias: numpy para operaciones numéricas, skfuzzy para lógica difusa y matplotlib.pyplot para la visualización de gráficos.
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define los rangos de las variables de universo:
# - x_qual: Rango de 0 a 10 que representa la calidad.
# - x_serv: Rango de 0 a 10 que representa el servicio.
# - x_tip: Rango de 0 a 25 que representa la propina en porcentaje.
x_qual = np.arange(0, 11, 1)
x_serv = np.arange(0, 11, 1)
x_tip  = np.arange(0, 26, 1)

# Genera funciones de membresía difusa para las variables de calidad (qual), servicio (serv), y propina (tip).

# Las siguientes líneas definen funciones de membresía tipo triangular (trimf) para las variables de calidad y servicio.
qual_lo = fuzz.trimf(x_qual, [0, 0, 5])
qual_md = fuzz.trimf(x_qual, [0, 5, 10])
qual_hi = fuzz.trimf(x_qual, [5, 10, 10])

serv_lo = fuzz.trimf(x_serv, [0, 0, 5])
serv_md = fuzz.trimf(x_serv, [0, 5, 10])
serv_hi = fuzz.trimf(x_serv, [5, 10, 10])

# Las siguientes líneas definen funciones de membresía difusa alternativas utilizando funciones tipo gaussiana (gaussmf).
# Puedes elegir entre las funciones de membresía triangular o gaussiana para modelar tus variables.
#qual_lo = fuzz.gaussmf(x_qual, 2, 2)
#qual_md = fuzz.gaussmf(x_qual, 5, 1)
#qual_hi = fuzz.gaussmf(x_qual, 8, 0.5)

# Estas líneas definen funciones de membresía tipo triangular para la variable de propina.
tip_lo = fuzz.trimf(x_tip, [0, 0, 13])
tip_md = fuzz.trimf(x_tip, [0, 13, 25])
tip_hi = fuzz.trimf(x_tip, [13, 25, 25])

"""# Crea una figura de Matplotlib con tres subgráficos (ax0, ax1 y ax2) para visualizar las funciones de membresía difusa.
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

# En las siguientes secciones, se grafican las funciones de membresía difusa en los subgráficos correspondientes.
# Cada subgráfico representa una variable (calidad, servicio y propina) con sus respectivas funciones de membresía.

# - ax0: Subgráfico para la calidad.
ax0.plot(x_qual, qual_lo, 'b', linewidth=1.5, label='Bad')
ax0.plot(x_qual, qual_md, 'g', linewidth=1.5, label='Decent')
ax0.plot(x_qual, qual_hi, 'r', linewidth=1.5, label='Great')
ax0.set_title('Food quality')  # Título del subgráfico.
ax0.legend()  # Agrega una leyenda al subgráfico.

# - ax1: Subgráfico para el servicio.
ax1.plot(x_serv, serv_lo, 'b', linewidth=1.5, label='Poor')
ax1.plot(x_serv, serv_md, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(x_serv, serv_hi, 'r', linewidth=1.5, label='Amazing')
ax1.set_title('Service quality')  # Título del subgráfico.
ax1.legend()  # Agrega una leyenda al subgráfico.

# - ax2: Subgráfico para la propina.
ax2.plot(x_tip, tip_lo, 'b', linewidth=1.5, label='Low')
ax2.plot(x_tip, tip_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_tip, tip_hi, 'r', linewidth=1.5, label='High')
ax2.set_title('Tip amount')  # Título del subgráfico.
ax2.legend()  # Agrega una leyenda al subgráfico.

# Desactiva los ejes superiores y derechos de cada subgráfico para hacer que las gráficas sean más limpias.
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Ajusta el diseño general de la figura.
plt.tight_layout()

#plt.show() """

x_nota_asignatura = np.arange(0,101,1)
x_nota_examen = np.arange(0,11,1)
x_nota_concepto = np.arange(0,11,1)

nota_examen_baja = fuzz.trimf(x_nota_examen,[0,0,4])
nota_examen_regular = fuzz.trimf(x_nota_examen,[0,4,10])
nota_examen_excelente = fuzz.trimf(x_nota_examen,[7,10,10])

nota_concepto_baja = fuzz.trimf(x_nota_concepto,[0,0,4])
nota_concepto_regular = fuzz.trimf(x_nota_concepto,[0,4,10])
nota_concepto_excelente = fuzz.trimf(x_nota_concepto,[7,10,10])

nota_asignatura_baja = fuzz.trimf(x_nota_asignatura,[0,0,40])
nota_asignatura_regular = fuzz.trimf(x_nota_asignatura,[0,40,100])
nota_asignatura_excelente = fuzz.trimf(x_nota_asignatura,[70,100,100])

pertenencia_examen_bajo = fuzz.interp_membership(x_nota_examen,nota_examen_baja,9.5)
pertenencia_examen_regular = fuzz.interp_membership(x_nota_examen,nota_examen_regular,9.5)
pertenencia_examen_excelente = fuzz.interp_membership(x_nota_examen,nota_examen_excelente,9.5)

pertenencia_concepto_baja = fuzz.interp_membership(x_nota_concepto,nota_concepto_baja,7.3)
pertenencia_concepto_regular = fuzz.interp_membership(x_nota_concepto,nota_concepto_regular,7.3)
pertenencia_concepto_excelente = fuzz.interp_membership(x_nota_concepto,nota_concepto_excelente,7.3)

# Crea una figura de Matplotlib con tres subgráficos (ax0, ax1 y ax2) para visualizar las funciones de membresía difusa.
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

# En las siguientes secciones, se grafican las funciones de membresía difusa en los subgráficos correspondientes.
# Cada subgráfico representa una variable (calidad, servicio y propina) con sus respectivas funciones de membresía.

# - ax0: Subgráfico para la calidad.
ax0.plot(x_nota_examen, nota_examen_baja, 'b', linewidth=1.5, label='Regular')
ax0.plot(x_nota_examen, nota_examen_regular, 'g', linewidth=1.5, label='Buena')
ax0.plot(x_nota_examen, nota_examen_excelente, 'r', linewidth=1.5, label='Excelente')
ax0.set_title('Nota del examen')  # Título del subgráfico.
ax0.legend()  # Agrega una leyenda al subgráfico.

# - ax1: Subgráfico para el servicio.
ax1.plot(x_nota_concepto, nota_concepto_baja, 'b', linewidth=1.5, label='Regular')
ax1.plot(x_nota_concepto, nota_concepto_regular, 'g', linewidth=1.5, label='Buena')
ax1.plot(x_nota_concepto, nota_concepto_excelente, 'r', linewidth=1.5, label='Excelente')
ax1.set_title('Nota conceptual')  # Título del subgráfico.
ax1.legend()  # Agrega una leyenda al subgráfico.

# - ax2: Subgráfico para la propina.
ax2.plot(x_nota_asignatura, nota_asignatura_baja, 'b', linewidth=1.5, label='Regular')
ax2.plot(x_nota_asignatura, nota_asignatura_regular, 'g', linewidth=1.5, label='Buena')
ax2.plot(x_nota_asignatura, nota_asignatura_excelente, 'r', linewidth=1.5, label='Excelente')
ax2.set_title('Nota asignatura')  # Título del subgráfico.
ax2.legend()  # Agrega una leyenda al subgráfico.

# Desactiva los ejes superiores y derechos de cada subgráfico para hacer que las gráficas sean más limpias.
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Ajusta el diseño general de la figura.
plt.tight_layout()

plt.show()


print("pertenencia examen bajo: ", pertenencia_examen_bajo )
print("pertenencia examen regular: ", pertenencia_examen_regular)
print("pertenencia examen exc: ", pertenencia_examen_excelente)

print("pertenencia concepto bajo: ", pertenencia_concepto_baja)
print("pertenencia concepto regular: ", pertenencia_concepto_regular)
print("pertenencia concepto exc: ", pertenencia_concepto_excelente)

#Regla 1: Si la nota del examen es baja, la nota de asignatura es baja
activacionNotaBaja = np.fmin(pertenencia_examen_bajo,nota_asignatura_baja)

#Regla 2: Si la nota del examen es regular, la nota de asignatura será regular
activacionNotaRegular = np.fmin(pertenencia_examen_regular,nota_asignatura_regular)

#Regla 3: si la nota del examen es alta y la nota de concepto es alta, la nota de asignatura será alta
regla3 = np.fmin(pertenencia_examen_excelente,pertenencia_concepto_excelente)
activacionNotaExcelente = np.fmin(regla3,nota_asignatura_excelente)

print("activacion nota baja regular y excelente en ese orden:", activacionNotaBaja,activacionNotaRegular,activacionNotaExcelente)
fig, ax0 = plt.subplots(figsize=(8, 3))
asignatura0 = np.zeros_like(x_nota_asignatura)

ax0.fill_between(x_nota_asignatura, asignatura0, activacionNotaBaja, facecolor='b', alpha=0.7)
ax0.plot(x_nota_asignatura, nota_asignatura_baja, 'b', linewidth=0.5, linestyle='--')
ax0.fill_between(x_nota_asignatura, asignatura0, activacionNotaRegular, facecolor='g', alpha=0.7)
ax0.plot(x_nota_asignatura, nota_asignatura_regular, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_nota_asignatura,asignatura0, activacionNotaExcelente, facecolor='r', alpha=0.7)
ax0.plot(x_nota_asignatura, nota_asignatura_excelente, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Desactiva ejes superiores y derechos del subgráfico
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left() 

# Ajusta el diseño del subgráfico
plt.tight_layout()

# Muestra los gráficos
plt.show()

############################################# hasta aca anda bien, ver lo de abajo !


# Agrega todas las tres funciones de membresía de salida juntas
aggregated = np.fmax(activacionNotaBaja,np.fmax(activacionNotaRegular,activacionNotaExcelente))




# Calcula el resultado desfusificado
nota_asignatura = fuzz.defuzz(x_nota_asignatura, aggregated, 'centroide')
notaasignatura_activation = fuzz.interp_membership(x_nota_asignatura, aggregated, nota_asignatura)  # para la visualización

# Visualiza esto
fig, ax0 = plt.subplots(figsize=(8, 3))

# Grafica las funciones de membresía de salida y la función de membresía agregada

# Funciones de membresía de salida
ax0.plot(x_nota_asignatura, nota_asignatura_baja, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_nota_asignatura, nota_asignatura_regular, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_nota_asignatura, nota_asignatura_excelente, 'r', linewidth=0.5, linestyle='--')

# Área sombreada que representa la función de membresía agregada
ax0.fill_between(x_nota_asignatura, asignatura0, aggregated, facecolor='Orange', alpha=0.7)

# Línea vertical que representa el valor desfusificado de la propina
ax0.plot([nota_asignatura, nota_asignatura], [0, notaasignatura_activation], 'k', linewidth=1.5, alpha=0.9)

ax0.set_title('Resultado (línea)')

# Desactiva los ejes superiores y derechos del gráfico
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Ajusta el diseño del gráfico
plt.tight_layout()

# Genera una función de membresía trapezoidal en el rango [0, 1]
x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])

# Desfusifica esta función de membresía de cinco formas diferentes
defuzz_centroid = fuzz.defuzz(x, mfx, 'Centroide')  # Igual que skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x, mfx, 'Bisector')
defuzz_mom = fuzz.defuzz(x, mfx, 'mom')
defuzz_som = fuzz.defuzz(x, mfx, 'som')
defuzz_lom = fuzz.defuzz(x, mfx, 'lom')

# Recopila información para líneas verticales
labels = ['Centroide', 'Bisector', 'Media de máximo', 'Mínimo de máximo',
          'Máximo de máximo']
xvals = [defuzz_centroid,
         defuzz_bisector,
         defuzz_mom,
         defuzz_som,
         defuzz_lom]
colors = ['r', 'b', 'g', 'c', 'm']
ymax = [fuzz.interp_membership(x, mfx, i) for i in xvals]

# Muestra y compara los resultados de desfusificación frente a la función de membresía
plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)
plt.ylabel('Membresía difusa')
plt.xlabel('Variable del universo (arb)')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()

 
""" Reglas TP:
        Regla 1: Si la nota del examen es baja, la nota de la asignatura es baja
        Regla 2: Si la nota del examen es regular,  la nota de la asignatura será regular
        Regla 3: Si la nota del examen es Alta Y la nota de concepto es alta, la nota de la asignatura será alta

        probemos con estas, si no da algo muy coherente vamos agregando alguna otra regla o modificando
"""

"""# Calcula la activación de las funciones de membresía para ciertos valores
# que no coinciden exactamente con los valores del universo.
qual_level_lo = fuzz.interp_membership(x_qual, qual_lo, 6.5) #valor de pertenencia de calidad mala para 6.5
qual_level_md = fuzz.interp_membership(x_qual, qual_md, 6.5) #valor de pertenencia de calidad media para 6.5
qual_level_hi = fuzz.interp_membership(x_qual, qual_hi, 6.5) #valor de pertenencia de calidad buena para 6.5

serv_level_lo = fuzz.interp_membership(x_serv, serv_lo, 9.8)
serv_level_md = fuzz.interp_membership(x_serv, serv_md, 9.8)
serv_level_hi = fuzz.interp_membership(x_serv, serv_hi, 9.8)

# Aplica las reglas difusas

# Regla 1: Comida mala O Servicio deficiente con propina baja
active_rule1 = np.fmax(qual_level_lo, serv_level_lo)
tip_activation_lo = np.fmin(active_rule1, tip_lo)

# Regla 2: Conectar Servicio aceptable con Propina media
tip_activation_md = np.fmin(serv_level_md, tip_md)

# Regla 3: Conectar Servicio excelente O Comida excelente con Propina alta
active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, tip_hi)

# Visualiza la activación de las funciones de membresía de salida
fig, ax0 = plt.subplots(figsize=(8, 3))

tip0 = np.zeros_like(x_tip)

ax0.fill_between(x_tip, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip, tip0, tip_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip,tip0, tip_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Desactiva ejes superiores y derechos del subgráfico
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left() """

exit()
# Ajusta el diseño del subgráfico
plt.tight_layout()

# Muestra los gráficos
plt.show()

aggregated = np.fmax(activacionNotaBaja,np.fmax(activacionNotaRegular,activacionNotaExcelente))

# Agrega todas las tres funciones de membresía de salida juntas
aggregated = np.fmax(tip_activation_lo,
                     np.fmax(tip_activation_md, tip_activation_hi))

# Calcula el resultado desfusificado
tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)  # para la visualización

# Visualiza esto
fig, ax0 = plt.subplots(figsize=(8, 3))

# Grafica las funciones de membresía de salida y la función de membresía agregada

# Funciones de membresía de salida
ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')

# Área sombreada que representa la función de membresía agregada
ax0.fill_between(x_tip, tip0, aggregated, facecolor='Orange', alpha=0.7)

# Línea vertical que representa el valor desfusificado de la propina
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)

ax0.set_title('Aggregated membership and result (line)')

# Desactiva los ejes superiores y derechos del gráfico
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Ajusta el diseño del gráfico
plt.tight_layout()

# Genera una función de membresía trapezoidal en el rango [0, 1]
x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])

# Desfusifica esta función de membresía de cinco formas diferentes
defuzz_centroid = fuzz.defuzz(x, mfx, 'centroid')  # Igual que skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x, mfx, 'bisector')
defuzz_mom = fuzz.defuzz(x, mfx, 'mom')
defuzz_som = fuzz.defuzz(x, mfx, 'som')
defuzz_lom = fuzz.defuzz(x, mfx, 'lom')

# Recopila información para líneas verticales
labels = ['centroide', 'bisector', 'media de máximo', 'mínimo de máximo',
          'máximo de máximo']
xvals = [defuzz_centroid,
         defuzz_bisector,
         defuzz_mom,
         defuzz_som,
         defuzz_lom]
colors = ['r', 'b', 'g', 'c', 'm']
ymax = [fuzz.interp_membership(x, mfx, i) for i in xvals]

# Muestra y compara los resultados de desfusificación frente a la función de membresía
plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)
plt.ylabel('Membresía difusa')
plt.xlabel('Variable del universo (arb)')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()