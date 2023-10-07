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


x_nota_asignatura = np.arange(-40,141,1)
x_nota_examen = np.arange(0,11,1)
x_nota_concepto = np.arange(0,11,1)


#aca defino los valores de las variables difusas
nota_examen_baja = fuzz.trimf(x_nota_examen,[0,0,4])
nota_examen_regular = fuzz.trimf(x_nota_examen,[2,4,10])
nota_examen_excelente = fuzz.trimf(x_nota_examen,[7,10,10])

nota_concepto_baja = fuzz.trimf(x_nota_concepto,[0,0,4])
nota_concepto_regular = fuzz.trimf(x_nota_concepto,[2,4,10])
nota_concepto_excelente = fuzz.trimf(x_nota_concepto,[7,10,10])

nota_asignatura_baja = fuzz.trimf(x_nota_asignatura,[-40,0,41])
nota_asignatura_regular = fuzz.trimf(x_nota_asignatura,[20,50,100])
nota_asignatura_excelente = fuzz.trimf(x_nota_asignatura,[60,100,140])


notaexamen=7
notaconcepto=5

#aca cambias los valores de entrada
pertenencia_examen_bajo = fuzz.interp_membership(x_nota_examen,nota_examen_baja,notaexamen)
pertenencia_examen_regular = fuzz.interp_membership(x_nota_examen,nota_examen_regular,notaexamen)
pertenencia_examen_excelente = fuzz.interp_membership(x_nota_examen,nota_examen_excelente,notaexamen)

pertenencia_concepto_baja = fuzz.interp_membership(x_nota_concepto,nota_concepto_baja,notaconcepto)
pertenencia_concepto_regular = fuzz.interp_membership(x_nota_concepto,nota_concepto_regular,notaconcepto)
pertenencia_concepto_excelente = fuzz.interp_membership(x_nota_concepto,nota_concepto_excelente,notaconcepto)




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


#Regla 1: Si la nota del examen es baja y la nota de concepto es baja, regular o excelente, la nota de asignatura es baja
Regla1 = np.fmax(pertenencia_concepto_baja,pertenencia_concepto_regular) #si es baja o regular
Regla1 = np.fmax(Regla1,pertenencia_concepto_excelente) #o excelente
Regla1 = np.fmin(Regla1,pertenencia_examen_bajo) #y la nota del examen baja
activacionRegla1 = np.fmin(Regla1,nota_asignatura_baja) #la nota de la asignatura sera baja

#Regla 2: Si la nota del examen es excelente y la nota de concepto es excelente, la nota de asignatura es excelente
regla2 = np.fmin(pertenencia_concepto_excelente,pertenencia_examen_excelente)
activacionRegla2 = np.fmin(regla2,nota_asignatura_excelente)

#Regla 3: Si la nota del examen es excelente y la nota de concepto es baja o regular, la nota de la asignatura es regular
regla3 = np.fmax(pertenencia_concepto_regular,pertenencia_concepto_baja)
regla3 = np.fmin(regla3,pertenencia_examen_excelente)
activacionRegla3 = np.fmin(regla3,nota_asignatura_regular)

#Regla 4: Si la nota del examen es regular y la nota de concepto es baja, la nota de la asignatura es baja
regla4 = np.fmin(pertenencia_examen_regular,pertenencia_concepto_baja)
activacionRegla4 = np.fmin(regla4,nota_asignatura_baja)

#Regla 5: Si la nota del examen es regular y la nota de concepto es excelente o regular, la nota de la asignatura es regular
regla5 = np.fmax(pertenencia_concepto_excelente,pertenencia_concepto_regular)
regla5 = np.fmin(regla5,pertenencia_examen_regular)
activacionRegla5 = np.fmin(regla5,nota_asignatura_regular)

print("activacion nota baja regular y excelente en ese orden:", activacionRegla1,activacionRegla3,activacionRegla2)
fig, ax0 = plt.subplots(figsize=(8, 3))
asignatura0 = np.zeros_like(x_nota_asignatura)

ax0.fill_between(x_nota_asignatura, asignatura0, activacionRegla1, facecolor='b', alpha=0.7)
ax0.plot(x_nota_asignatura, nota_asignatura_baja, 'b', linewidth=0.5, linestyle='--')
ax0.fill_between(x_nota_asignatura, asignatura0, activacionRegla3, facecolor='g', alpha=0.7)
ax0.plot(x_nota_asignatura, nota_asignatura_regular, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_nota_asignatura,asignatura0, activacionRegla2, facecolor='r', alpha=0.7)
ax0.plot(x_nota_asignatura, nota_asignatura_excelente, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_nota_asignatura,asignatura0,activacionRegla4,facecolor='m',alpha=0.7)
ax0.plot(x_nota_asignatura,nota_asignatura_baja, 'm',linewidth=0.5, linestyle='--')
ax0.fill_between(x_nota_asignatura,asignatura0,activacionRegla5,facecolor='y',alpha=0.7)
ax0.plot(x_nota_asignatura,nota_asignatura_regular, 'y',linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')


aggregated = np.fmax(activacionRegla1,np.fmax(np.fmax(np.fmax(activacionRegla4,activacionRegla5),activacionRegla3),activacionRegla2))

"""
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
ax0.set_title('Output membership activity')"""

# Desactiva ejes superiores y derechos del subgráfico
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left() 

# Ajusta el diseño del subgráfico
plt.tight_layout()

# Muestra los gráficos
plt.show()



# Agrega todas las tres funciones de membresía de salida juntas
#aggregated = np.fmax(activacionNotaBaja,np.fmax(activacionNotaRegular,activacionNotaExcelente))




# Calcula el resultado desfusificado
nota_asignatura = fuzz.defuzz(x_nota_asignatura, aggregated, 'centroid')
print("La nota final de asignatura es de ", nota_asignatura)
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


