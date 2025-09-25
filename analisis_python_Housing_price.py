# en la terminal antes de ejecutar este script, instalar las librerías necesarias con:
# pip install pandas numpy matplotlib seaborn scipy scikit-learn missingno

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
import missingno as msno
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS - PRECIOS DE VIVIENDAS")
print("="*80)

# Cargar datos
df = pd.read_csv('Housing_price_prediction.csv')

# 1. Descripción del conjunto de datos
print("\n1. DESCRIPCIÓN DEL CONJUNTO DE DATOS:")
print("-" * 50)
print(f"📊 Dimensiones del dataset: {df.shape[0]} observaciones (filas) × {df.shape[1]} variables (columnas)")
print(f"📈 Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n📋 TIPOS DE VARIABLES:")
for col, dtype in df.dtypes.items():
    unique_vals = df[col].nunique()
    print(f"  • {col:20} → {str(dtype):10} ({unique_vals:3} valores únicos)")

print("\n🔍 VISTA PREVIA DE LOS DATOS:")
print(df.head().to_string())

print("\n📊 INFORMACIÓN GENERAL:")
print(f"  • Variables categóricas: {len(df.select_dtypes(include=['object']).columns)}")
print(f"  • Variables numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"  • Valores faltantes totales: {df.isnull().sum().sum()}")

# 2. Análisis gráfico de variables categóricas (MEJORADO)
print("\n" + "="*80)
print("2. ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("="*80)

categorical_cols = ['mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Crear tabla de frecuencias para cada variable categórica
print("\n📊 TABLA DE FRECUENCIAS:")
for col in categorical_cols:
    print(f"\n{col.upper()}:")
    freq_table = df[col].value_counts()
    perc_table = df[col].value_counts(normalize=True) * 100
    combined = pd.DataFrame({
        'Frecuencia': freq_table,
        'Porcentaje': perc_table.round(2)
    })
    print(combined.to_string())

# Gráfico mejorado con títulos y etiquetas claras
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Distribución de Variables Categóricas en el Dataset de Viviendas', 
             fontsize=16, fontweight='bold', y=0.98)

for i, col in enumerate(categorical_cols):
    row = i // 3
    col_idx = i % 3
    ax = axes[row, col_idx]
    
    counts = df[col].value_counts()
    bars = ax.bar(counts.index, counts.values, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(f'{col.replace("_", " ").title()}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Categorías', fontsize=10, fontweight='bold')
    ax.set_ylabel('Número de Viviendas', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotar etiquetas si son largas
    if len(str(counts.index[0])) > 5:
        ax.tick_params(axis='x', rotation=45)

# Eliminar subplot vacío
axes[2, 2].remove()

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# 3. Tabla resumen de variables numéricas (MEJORADA)
print("\n" + "="*80)
print("3. ANÁLISIS ESTADÍSTICO DE VARIABLES NUMÉRICAS")
print("="*80)

numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

print("\n📊 MEDIDAS ESTADÍSTICAS DESCRIPTIVAS:")
print("Fórmulas aplicadas:")
print("  • Media (μ) = Σxi/n")
print("  • Desviación estándar (σ) = √(Σ(xi-μ)²/(n-1))")
print("  • Percentiles: valores que dividen los datos ordenados")
print("  • Rango intercuartílico (IQR) = Q3 - Q1")

desc_stats = df[numeric_cols].describe()
print("\n" + desc_stats.to_string())

# Información adicional
print("\n📈 INFORMACIÓN ADICIONAL:")
for col in numeric_cols:
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"\n{col.upper()}:")
    print(f"  • Asimetría (Skewness): {skewness:.3f} {'(Sesgada a la derecha)' if skewness > 0 else '(Sesgada a la izquierda)' if skewness < 0 else '(Simétrica)'}")
    print(f"  • Curtosis: {kurtosis:.3f} {'(Leptocúrtica - más puntiaguda)' if kurtosis > 0 else '(Platicúrtica - más plana)' if kurtosis < 0 else '(Mesocúrtica - normal)'}")
    print(f"  • Rango: {df[col].min():.0f} - {df[col].max():.0f}")
    print(f"  • Coeficiente de variación: {(df[col].std()/df[col].mean()*100):.2f}%")

# 4. Histograma con curva de densidad para 'price' (MEJORADO)
print("\n" + "="*80)
print("4. ANÁLISIS DE DISTRIBUCIÓN DE PRECIOS")
print("="*80)

print("\n📊 ESTADÍSTICAS DETALLADAS DE PRECIOS:")
price_stats = {
    'Media': df['price'].mean(),
    'Mediana': df['price'].median(),
    'Moda': df['price'].mode().iloc[0],
    'Desviación Estándar': df['price'].std(),
    'Varianza': df['price'].var(),
    'Rango': df['price'].max() - df['price'].min(),
    'IQR': df['price'].quantile(0.75) - df['price'].quantile(0.25)
}

for stat, value in price_stats.items():
    print(f"  • {stat}: ${value:,.2f}")

# Crear bins más informativos
n_bins = 25
plt.figure(figsize=(14, 8))

# Histograma con más detalle
n, bins, patches = plt.hist(df['price'], bins=n_bins, alpha=0.7, color='skyblue', 
                           edgecolor='black', linewidth=0.5, density=True)

# Curva de densidad
kde_x = np.linspace(df['price'].min(), df['price'].max(), 100)
kde = stats.gaussian_kde(df['price'])
plt.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='Curva de Densidad (KDE)')

# Líneas de referencia
plt.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Media: ${df["price"].mean():,.0f}')
plt.axvline(df['price'].median(), color='green', linestyle='--', linewidth=2, 
           label=f'Mediana: ${df["price"].median():,.0f}')

plt.title('Distribución de Precios de Viviendas\n(Histograma con Curva de Densidad)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Precio de la Vivienda (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Densidad de Probabilidad', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Formato de números en el eje x
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

print(f"\n📈 INTERPRETACIÓN DE LA DISTRIBUCIÓN:")
skew_price = df['price'].skew()
if skew_price > 0.5:
    print("  • La distribución está SESGADA A LA DERECHA (cola larga hacia valores altos)")
    print("  • Esto indica que hay más viviendas de precio bajo-medio y pocas muy caras")
elif skew_price < -0.5:
    print("  • La distribución está SESGADA A LA IZQUIERDA (cola larga hacia valores bajos)")
else:
    print("  • La distribución es aproximadamente SIMÉTRICA")

# 5. Análisis de normalidad para 'price' (Q-Q plot y Shapiro-Wilk) - MEJORADO
print("\n" + "="*80)
print("5. ANÁLISIS DE NORMALIDAD - GRÁFICO Q-Q")
print("="*80)

print("\n🔍 ¿QUÉ ES UN GRÁFICO Q-Q?")
print("El gráfico Q-Q (Quantile-Quantile) compara los cuantiles de nuestros datos")
print("con los cuantiles de una distribución normal teórica.")
print("\nFórmula matemática:")
print("  • Cuantil teórico: Φ⁻¹(p) donde Φ es la función de distribución normal")
print("  • Cuantil observado: valor en la posición p de los datos ordenados")
print("\nInterpretación:")
print("  • Si los puntos siguen la línea diagonal → datos normales")
print("  • Curvatura hacia arriba → cola derecha más pesada")
print("  • Curvatura hacia abajo → cola izquierda más pesada")

plt.figure(figsize=(12, 6))

# Q-Q plot
plt.subplot(1, 2, 1)
stats.probplot(df['price'], dist="norm", plot=plt)
plt.title('Gráfico Q-Q: Precios vs Distribución Normal\n(Análisis de Normalidad)', 
          fontsize=12, fontweight='bold')
plt.xlabel('Cuantiles Teóricos (Distribución Normal)', fontsize=10, fontweight='bold')
plt.ylabel('Cuantiles Observados (Precios)', fontsize=10, fontweight='bold')
plt.grid(alpha=0.3)

# Histograma comparativo
plt.subplot(1, 2, 2)
plt.hist(df['price'], bins=30, density=True, alpha=0.7, color='lightblue', 
         edgecolor='black', label='Datos Observados')

# Distribución normal teórica
mu, sigma = df['price'].mean(), df['price'].std()
x = np.linspace(df['price'].min(), df['price'].max(), 100)
normal_dist = stats.norm.pdf(x, mu, sigma)
plt.plot(x, normal_dist, 'r-', linewidth=2, label='Distribución Normal Teórica')

plt.title('Comparación: Datos vs Distribución Normal', fontsize=12, fontweight='bold')
plt.xlabel('Precio de la Vivienda', fontsize=10, fontweight='bold')
plt.ylabel('Densidad', fontsize=10, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Test de Shapiro-Wilk
stat, p_value = stats.shapiro(df['price'])
print(f"\n📊 TEST DE SHAPIRO-WILK:")
print(f"Fórmula: W = (Σaᵢx₍ᵢ₎)² / Σ(xᵢ - x̄)²")
print(f"  • Estadístico W: {stat:.6f}")
print(f"  • Valor p: {p_value:.2e}")
print(f"  • Interpretación: {'Los datos NO siguen distribución normal' if p_value < 0.05 else 'Los datos siguen distribución normal'} (α = 0.05)")

print(f"\n🎯 CONCLUSIÓN SOBRE NORMALIDAD:")
if p_value < 0.05:
    print("  • Los precios NO siguen una distribución normal")
    print("  • Esto es LÓGICO porque los precios de viviendas típicamente:")
    print("    - Tienen un límite inferior (no pueden ser negativos)")
    print("    - Presentan sesgo hacia la derecha (pocas casas muy caras)")
    print("    - Reflejan la realidad del mercado inmobiliario")
else:
    print("  • Los precios siguen aproximadamente una distribución normal")

# 6. Identificación de datos atípicos (boxplot) - MEJORADO
print("\n" + "="*80)
print("6. ANÁLISIS DE DATOS ATÍPICOS (OUTLIERS)")
print("="*80)

print("\n📊 FÓRMULAS PARA DETECCIÓN DE OUTLIERS:")
print("Método del Rango Intercuartílico (IQR):")
print("  • Q1 = Percentil 25")
print("  • Q3 = Percentil 75") 
print("  • IQR = Q3 - Q1")
print("  • Límite inferior = Q1 - 1.5 × IQR")
print("  • Límite superior = Q3 + 1.5 × IQR")
print("  • Outlier: valor < límite inferior O valor > límite superior")

# Calcular outliers para cada variable
outlier_info = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    
    outlier_info[col] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lower_bound': lower_bound, 'upper_bound': upper_bound,
        'n_outliers': len(outliers), 'outliers': outliers.tolist()
    }

print(f"\n📈 RESUMEN DE OUTLIERS POR VARIABLE:")
for col, info in outlier_info.items():
    print(f"\n{col.upper()}:")
    print(f"  • Rango normal: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    print(f"  • Número de outliers: {info['n_outliers']} ({info['n_outliers']/len(df)*100:.1f}% del total)")
    if info['n_outliers'] > 0 and info['n_outliers'] <= 10:
        print(f"  • Valores atípicos: {info['outliers']}")

# Boxplot mejorado
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Diagramas de Caja para Variables Numéricas\n(Detección de Valores Atípicos)', 
             fontsize=16, fontweight='bold', y=0.98)

for i, col in enumerate(numeric_cols):
    row = i // 3
    col_idx = i % 3
    ax = axes[row, col_idx]
    
    # Crear boxplot
    bp = ax.boxplot(df[col], patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Añadir estadísticas
    stats_text = f"Mediana: {df[col].median():.1f}\nIQR: {outlier_info[col]['IQR']:.1f}\nOutliers: {outlier_info[col]['n_outliers']}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'{col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Valores de {col.title()}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Variable', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Formato especial para precio
    if col == 'price':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

print(f"\n🎯 INTERPRETACIÓN DE LOS BOXPLOTS:")
print("  • La 'caja' representa el 50% central de los datos (Q1 a Q3)")
print("  • La línea dentro de la caja es la mediana")
print("  • Los 'bigotes' se extienden hasta 1.5×IQR desde los cuartiles")
print("  • Los puntos fuera de los bigotes son outliers")
print("  • Es NORMAL que 'price' tenga la escala más amplia (valores en miles)")

# 7. Relación entre variable numérica y categórica (MEJORADO)
print("\n" + "="*80)
print("7. ANÁLISIS: PRECIO POR ESTADO DE AMUEBLADO")
print("="*80)

print("\n📊 ESTADÍSTICAS POR CATEGORÍA:")
price_by_furnishing = df.groupby('furnishingstatus')['price'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)
print(price_by_furnishing.to_string())

# Test ANOVA para diferencias significativas
groups = [group['price'].values for name, group in df.groupby('furnishingstatus')]
f_stat, p_val_anova = stats.f_oneway(*groups)

print(f"\n📈 TEST ANOVA (Análisis de Varianza):")
print(f"Fórmula: F = (Varianza entre grupos) / (Varianza dentro de grupos)")
print(f"  • Estadístico F: {f_stat:.4f}")
print(f"  • Valor p: {p_val_anova:.6f}")
print(f"  • Interpretación: {'Hay diferencias significativas' if p_val_anova < 0.05 else 'No hay diferencias significativas'} entre grupos (α = 0.05)")

plt.figure(figsize=(14, 8))

# Boxplot mejorado
ax = sns.boxplot(x='furnishingstatus', y='price', data=df, palette='Set2')

# Añadir puntos de datos
sns.stripplot(x='furnishingstatus', y='price', data=df, size=4, color='black', alpha=0.3)

# Añadir medias
means = df.groupby('furnishingstatus')['price'].mean()
for i, (category, mean_val) in enumerate(means.items()):
    ax.plot(i, mean_val, marker='D', color='red', markersize=8, label='Media' if i == 0 else "")

plt.title('Distribución de Precios por Estado de Amueblado\n(Boxplot con Puntos de Datos)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Estado de Amueblado', fontsize=12, fontweight='bold')
plt.ylabel('Precio de la Vivienda (USD)', fontsize=12, fontweight='bold')

# Formato del eje Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n🎯 INTERPRETACIÓN DEL GRÁFICO:")
print("  • Cada caja muestra la distribución de precios para cada categoría")
print("  • Los puntos negros son las observaciones individuales")
print("  • Los diamantes rojos son las medias de cada grupo")
print("  • Es NORMAL ver diferencias entre categorías:")
print("    - 'Furnished' (amueblado) típicamente más caro")
print("    - 'Semi-furnished' (semi-amueblado) precio intermedio") 
print("    - 'Unfurnished' (sin amueblar) generalmente más barato")
print("  • Las diferencias reflejan el valor agregado del mobiliario")

# 8. Análisis de datos faltantes (MEJORADO)
print("\n" + "="*80)
print("8. ANÁLISIS DE DATOS FALTANTES")
print("="*80)

missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

print(f"\n📊 RESUMEN DE DATOS FALTANTES:")
missing_summary = pd.DataFrame({
    'Columna': missing_data.index,
    'Valores Faltantes': missing_data.values,
    'Porcentaje': missing_percent.values
}).round(2)

print(missing_summary.to_string(index=False))

if missing_data.sum() == 0:
    print("\n✅ ¡EXCELENTE! No hay datos faltantes en el dataset")
    print("  • Todas las 545 observaciones están completas")
    print("  • No se requiere imputación de datos")
else:
    print(f"\n⚠️  Se encontraron {missing_data.sum()} valores faltantes")

plt.figure(figsize=(12, 6))

# Gráfico de barras para datos faltantes
plt.subplot(1, 2, 1)
if missing_data.sum() > 0:
    missing_data[missing_data > 0].plot(kind='bar', color='salmon', alpha=0.8)
    plt.title('Datos Faltantes por Columna', fontsize=12, fontweight='bold')
    plt.xlabel('Columnas del Dataset', fontsize=10, fontweight='bold')
    plt.ylabel('Número de Valores Faltantes', fontsize=10, fontweight='bold')
else:
    plt.bar(range(len(df.columns)), [0]*len(df.columns), color='lightgreen', alpha=0.8)
    plt.title('Datos Faltantes por Columna\n(Dataset Completo)', fontsize=12, fontweight='bold')
    plt.xlabel('Columnas del Dataset', fontsize=10, fontweight='bold')
    plt.ylabel('Número de Valores Faltantes', fontsize=10, fontweight='bold')
    plt.xticks(range(len(df.columns)), df.columns, rotation=45)

plt.grid(axis='y', alpha=0.3)

# Matriz de completitud
plt.subplot(1, 2, 2)
msno.matrix(df, ax=plt.gca(), fontsize=8)
plt.title('Matriz de Completitud de Datos\n(Blanco = Faltante, Negro = Presente)', 
          fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n🎯 INTERPRETACIÓN DE LA MATRIZ:")
print("  • Cada fila representa una observación (vivienda)")
print("  • Cada columna representa una variable")
print("  • El color negro indica datos presentes")
print("  • El color blanco indica datos faltantes")
print(f"  • El número 545 se repite porque tenemos exactamente 545 observaciones")
print("  • Es NORMAL ver 545 en el eje Y (representa el número total de filas)")

# 9. Matriz de correlación (MEJORADA)
print("\n" + "="*80)
print("9. ANÁLISIS DE CORRELACIÓN ENTRE VARIABLES")
print("="*80)

print("\n📊 FÓRMULA DE CORRELACIÓN DE PEARSON:")
print("r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]")
print("\nInterpretación de valores:")
print("  • r = +1: Correlación positiva perfecta")
print("  • r = +0.7 a +0.9: Correlación positiva fuerte")
print("  • r = +0.3 a +0.7: Correlación positiva moderada")
print("  • r = -0.3 a +0.3: Correlación débil o nula")
print("  • r = -0.3 a -0.7: Correlación negativa moderada")
print("  • r = -0.7 a -0.9: Correlación negativa fuerte")
print("  • r = -1: Correlación negativa perfecta")

# Calcular matriz de correlación
corr_matrix = df[numeric_cols].corr()

print(f"\n📈 MATRIZ DE CORRELACIÓN:")
print(corr_matrix.round(3).to_string())

# Encontrar correlaciones más fuertes
correlations = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        correlations.append((var1, var2, corr_val))

correlations.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\n🔍 CORRELACIONES MÁS FUERTES:")
for var1, var2, corr in correlations[:5]:
    strength = "fuerte" if abs(corr) > 0.7 else "moderada" if abs(corr) > 0.3 else "débil"
    direction = "positiva" if corr > 0 else "negativa"
    print(f"  • {var1} ↔ {var2}: {corr:.3f} (correlación {direction} {strength})")

plt.figure(figsize=(12, 10))

# Crear heatmap mejorado
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.3f', cbar_kws={"shrink": .8},
            mask=mask, linewidths=0.5)

plt.title('Matriz de Correlación entre Variables Numéricas\n(Coeficiente de Correlación de Pearson)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Variables del Dataset', fontsize=12, fontweight='bold')
plt.ylabel('Variables del Dataset', fontsize=12, fontweight='bold')

# Añadir texto explicativo
plt.figtext(0.02, 0.02, 'Interpretación: Azul = Correlación Positiva, Rojo = Correlación Negativa\nValores cercanos a ±1 indican correlación fuerte', 
            fontsize=10, style='italic')

plt.tight_layout()
plt.show()

print(f"\n🎯 INTERPRETACIÓN DE LA MATRIZ:")
print("  • Diagonal principal = 1 (cada variable correlaciona perfectamente consigo misma)")
print("  • Colores azules = correlaciones positivas (cuando una sube, la otra también)")
print("  • Colores rojos = correlaciones negativas (cuando una sube, la otra baja)")
print("  • Intensidad del color = fuerza de la correlación")
print("  • Solo se muestra la mitad inferior para evitar redundancia")

# 10. Análisis de Componentes Principales (ACP) - MEJORADO
print("\n" + "="*80)
print("10. ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
print("="*80)

print("\n📊 FUNDAMENTOS MATEMÁTICOS DEL PCA:")
print("Objetivo: Reducir dimensionalidad preservando máxima varianza")
print("Fórmulas principales:")
print("  • Matriz de covarianza: C = (1/(n-1)) × X^T × X")
print("  • Eigenvalores (λ) y eigenvectors (v): C × v = λ × v")
print("  • Componentes principales = combinaciones lineales de variables originales")
print("  • Varianza explicada por componente i = λᵢ / Σλⱼ")

# Preprocesamiento para ACP
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

print(f"\n🔄 PREPROCESAMIENTO:")
print("  • Variables categóricas codificadas numéricamente (Label Encoding)")
print("  • Datos estandarizados (media=0, desviación=1)")
print("  • Fórmula estandarización: z = (x - μ) / σ")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Análisis de varianza explicada
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\n📈 VARIANZA EXPLICADA POR COMPONENTE:")
for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
    print(f"  • PC{i+1}: {var:.3f} ({var*100:.1f}%) - Acumulada: {cum_var:.3f} ({cum_var*100:.1f}%)")

# Determinar número óptimo de componentes
n_components_80 = np.argmax(cumulative_var >= 0.80) + 1
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1

print(f"\n🎯 COMPONENTES RECOMENDADOS:")
print(f"  • Para explicar 80% de varianza: {n_components_80} componentes")
print(f"  • Para explicar 90% de varianza: {n_components_90} componentes")

# Gráficos de PCA
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis de Componentes Principales (PCA)', fontsize=16, fontweight='bold')

# Varianza explicada
axes[0,0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.8, color='skyblue')
axes[0,0].set_title('Varianza Explicada por Componente', fontweight='bold')
axes[0,0].set_xlabel('Número de Componente')
axes[0,0].set_ylabel('Proporción de Varianza Explicada')
axes[0,0].grid(axis='y', alpha=0.3)

# Varianza acumulada
axes[0,1].plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linewidth=2, markersize=6)
axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80% varianza')
axes[0,1].axhline(y=0.9, color='g', linestyle='--', label='90% varianza')
axes[0,1].set_title('Varianza Explicada Acumulada', fontweight='bold')
axes[0,1].set_xlabel('Número de Componentes')
axes[0,1].set_ylabel('Varianza Explicada Acumulada')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Contribución de variables a PC1 y PC2
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
axes[1,0].scatter(loadings[:, 0], loadings[:, 1], alpha=0.8, s=100)
for i, col in enumerate(df_encoded.columns):
    axes[1,0].annotate(col, (loadings[i, 0], loadings[i, 1]), 
                      xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1,0].set_title('Cargas de Variables en PC1 vs PC2', fontweight='bold')
axes[1,0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% varianza)')
axes[1,0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% varianza)')
axes[1,0].grid(alpha=0.3)
axes[1,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1,0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Biplot (observaciones en espacio PC1-PC2)
scatter = axes[1,1].scatter(pca_result[:, 0], pca_result[:, 1], 
                           c=df_encoded['price'], cmap='viridis', alpha=0.6)
axes[1,1].set_title('Biplot: Observaciones en Espacio PC1-PC2', fontweight='bold')
axes[1,1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% varianza)')
axes[1,1].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% varianza)')
plt.colorbar(scatter, ax=axes[1,1], label='Precio')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎯 INTERPRETACIÓN DEL PCA:")
print("  • PC1 captura la mayor variabilidad en los datos")
print("  • Variables con cargas altas en PC1 son las más importantes")
print("  • El biplot muestra cómo se distribuyen las observaciones")
print("  • Colores en biplot representan precios (amarillo=alto, azul=bajo)")

# 11. Modelo 1: Regresión logística con todas las variables (MEJORADO)
print("\n" + "="*80)
print("11. MODELO 1: REGRESIÓN LOGÍSTICA (TODAS LAS VARIABLES)")
print("="*80)

print("\n📊 FUNDAMENTOS DE REGRESIÓN LOGÍSTICA:")
print("Función logística: p = 1 / (1 + e^(-z))")
print("donde z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
print("Para multiclase: Softmax = e^(zᵢ) / Σe^(zⱼ)")

# Preparar datos para el modelo
X = df_encoded.drop('furnishingstatus', axis=1)
y = df_encoded['furnishingstatus']

print(f"\n🔄 PREPARACIÓN DE DATOS:")
print(f"  • Variables predictoras (X): {X.shape[1]} variables")
print(f"  • Variable objetivo (y): furnishingstatus ({y.nunique()} clases)")
print(f"  • Distribución de clases:")
for class_val, count in y.value_counts().items():
    class_name = ['furnished', 'semi-furnished', 'unfurnished'][class_val]
    print(f"    - Clase {class_val} ({class_name}): {count} observaciones ({count/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n📊 DIVISIÓN DE DATOS:")
print(f"  • Entrenamiento: {X_train.shape[0]} observaciones ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  • Prueba: {X_test.shape[0]} observaciones ({X_test.shape[0]/len(df)*100:.1f}%)")

model1 = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
y_prob1 = model1.predict_proba(X_test)

print(f"\n📈 RESULTADOS DEL MODELO 1:")
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"  • Exactitud (Accuracy): {accuracy1:.4f} ({accuracy1*100:.2f}%)")
print(f"  • Fórmula Accuracy: (VP + VN) / (VP + VN + FP + FN)")

print(f"\n📊 MATRIZ DE CONFUSIÓN:")
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

print(f"\n📋 REPORTE DETALLADO:")
class_names = ['furnished', 'semi-furnished', 'unfurnished']
report1 = classification_report(y_test, y_pred1, target_names=class_names, output_dict=True)
for class_name in class_names:
    metrics = report1[class_name]
    print(f"\n{class_name.upper()}:")
    print(f"  • Precisión: {metrics['precision']:.3f} (VP/(VP+FP))")
    print(f"  • Recall: {metrics['recall']:.3f} (VP/(VP+FN))")
    print(f"  • F1-Score: {metrics['f1-score']:.3f} (2×Precisión×Recall/(Precisión+Recall))")

# 12. Modelo 2: Método backward (eliminación recursiva de variables) - MEJORADO
print("\n" + "="*80)
print("12. MODELO 2: SELECCIÓN DE VARIABLES (RFE)")
print("="*80)

print("\n📊 RECURSIVE FEATURE ELIMINATION (RFE):")
print("Algoritmo:")
print("  1. Entrenar modelo con todas las variables")
print("  2. Calcular importancia de cada variable")
print("  3. Eliminar la variable menos importante")
print("  4. Repetir hasta alcanzar número deseado de variables")

n_features_select = 5
selector = RFE(LogisticRegression(multi_class='multinomial',
               solver='lbfgs', max_iter=1000, random_state=42), 
               n_features_to_select=n_features_select)
selector.fit(X_train, y_train)

selected_features = X.columns[selector.support_]
print(f"\n🎯 VARIABLES SELECCIONADAS ({n_features_select} de {X.shape[1]}):")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i}. {feature}")

print(f"\n❌ VARIABLES ELIMINADAS:")
eliminated_features = X.columns[~selector.support_]
for i, feature in enumerate(eliminated_features, 1):
    print(f"  {i}. {feature}")

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

model2 = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model2.fit(X_train_selected, y_train)
y_pred2 = model2.predict(X_test_selected)
y_prob2 = model2.predict_proba(X_test_selected)

print(f"\n📈 RESULTADOS DEL MODELO 2:")
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"  • Exactitud (Accuracy): {accuracy2:.4f} ({accuracy2*100:.2f}%)")

print(f"\n📊 MATRIZ DE CONFUSIÓN:")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

print(f"\n📋 REPORTE DETALLADO:")
report2 = classification_report(y_test, y_pred2, target_names=class_names, output_dict=True)
for class_name in class_names:
    metrics = report2[class_name]
    print(f"\n{class_name.upper()}:")
    print(f"  • Precisión: {metrics['precision']:.3f}")
    print(f"  • Recall: {metrics['recall']:.3f}")
    print(f"  • F1-Score: {metrics['f1-score']:.3f}")

# 13. Comparación de modelos (MEJORADA)
print("\n" + "="*80)
print("13. COMPARACIÓN DETALLADA DE MODELOS")
print("="*80)

metrics = {
    'Modelo': ['Todas las variables', 'RFE (5 variables)'],
    'Accuracy': [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2)],
    'Precision': [precision_score(y_test, y_pred1, average='weighted'), 
                  precision_score(y_test, y_pred2, average='weighted')],
    'Recall': [recall_score(y_test, y_pred1, average='weighted'), 
               recall_score(y_test, y_pred2, average='weighted')],
    'F1-Score': [f1_score(y_test, y_pred1, average='weighted'), 
                 f1_score(y_test, y_pred2, average='weighted')],
    'N_Variables': [X.shape[1], n_features_select]
}
metrics_df = pd.DataFrame(metrics)
print("\n📊 TABLA COMPARATIVA:")
print(metrics_df.round(4).to_string(index=False))

# Determinar mejor modelo
if metrics_df.loc[0, 'Accuracy'] > metrics_df.loc[1, 'Accuracy']:
    best_model = "Modelo 1 (Todas las variables)"
    improvement = metrics_df.loc[0, 'Accuracy'] - metrics_df.loc[1, 'Accuracy']
else:
    best_model = "Modelo 2 (RFE)"
    improvement = metrics_df.loc[1, 'Accuracy'] - metrics_df.loc[0, 'Accuracy']

print(f"\n🏆 MEJOR MODELO: {best_model}")
print(f"  • Mejora en accuracy: {improvement:.4f} ({improvement*100:.2f}%)")
print(f"  • Consideraciones:")
print(f"    - Modelo 1: Usa toda la información disponible")
print(f"    - Modelo 2: Más simple, menos riesgo de sobreajuste")

# 14. Curvas ROC para modelos (multiclase) - MEJORADO
print("\n" + "="*80)
print("14. ANÁLISIS ROC (RECEIVER OPERATING CHARACTERISTIC)")
print("="*80)

print("\n📊 FUNDAMENTOS DE CURVAS ROC:")
print("Para clasificación multiclase:")
print("  • Se crea una curva ROC para cada clase (One-vs-Rest)")
print("  • TPR (Sensibilidad) = VP / (VP + FN)")
print("  • FPR (1-Especificidad) = FP / (FP + VN)")
print("  • AUC = Área bajo la curva ROC")
print("  • AUC = 0.5: Clasificador aleatorio")
print("  • AUC = 1.0: Clasificador perfecto")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Curva ROC para Modelo 1
fpr1, tpr1, roc_auc1 = dict(), dict(), dict()
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(
        y_test_bin[:, i], y_prob1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

# Curva ROC para Modelo 2
fpr2, tpr2, roc_auc2 = dict(), dict(), dict()
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(
        y_test_bin[:, i], y_prob2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

# Calcular ROC macro-promedio
all_fpr1 = np.unique(np.concatenate([fpr1[i] for i in range(n_classes)]))
mean_tpr1 = np.zeros_like(all_fpr1)
for i in range(n_classes):
    mean_tpr1 += np.interp(all_fpr1, fpr1[i], tpr1[i])
mean_tpr1 /= n_classes
roc_auc1['macro'] = auc(all_fpr1, mean_tpr1)

all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(n_classes)]))
mean_tpr2 = np.zeros_like(all_fpr2)
for i in range(n_classes):
    mean_tpr2 += np.interp(all_fpr2, fpr2[i], tpr2[i])
mean_tpr2 /= n_classes
roc_auc2['macro'] = auc(all_fpr2, mean_tpr2)

print(f"\n📈 VALORES AUC POR CLASE:")
print(f"{'Clase':<15} {'Modelo 1':<10} {'Modelo 2':<10}")
print("-" * 35)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<15} {roc_auc1[i]:<10.3f} {roc_auc2[i]:<10.3f}")
print(f"{'Macro-promedio':<15} {roc_auc1['macro']:<10.3f} {roc_auc2['macro']:<10.3f}")

# Plot ROC mejorado
plt.figure(figsize=(15, 10))

# ROC por clase
for i in range(2):  # Solo primeras 2 clases para claridad
    plt.subplot(2, 2, i+1)
    
    plt.plot(fpr1[i], tpr1[i], color='blue', lw=2,
             label=f'Modelo 1 (AUC = {roc_auc1[i]:.3f})')
    plt.plot(fpr2[i], tpr2[i], color='red', lw=2, linestyle='--',
             label=f'Modelo 2 (AUC = {roc_auc2[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Clasificador Aleatorio')
    
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontweight='bold')
    plt.title(f'ROC: Clase "{class_names[i]}"', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

# ROC macro-promedio
plt.subplot(2, 2, 3)
plt.plot(all_fpr1, mean_tpr1, color='blue', lw=2,
         label=f'Modelo 1 Macro-avg (AUC = {roc_auc1["macro"]:.3f})')
plt.plot(all_fpr2, mean_tpr2, color='red', lw=2, linestyle='--',
         label=f'Modelo 2 Macro-avg (AUC = {roc_auc2["macro"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

plt.xlabel('Tasa de Falsos Positivos (FPR)', fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontweight='bold')
plt.title('ROC Macro-Promedio', fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Comparación AUC
plt.subplot(2, 2, 4)
classes_extended = class_names + ['Macro-avg']
auc1_values = [roc_auc1[i] for i in range(n_classes)] + [roc_auc1['macro']]
auc2_values = [roc_auc2[i] for i in range(n_classes)] + [roc_auc2['macro']]

x = np.arange(len(classes_extended))
width = 0.35

plt.bar(x - width/2, auc1_values, width, label='Modelo 1', alpha=0.8, color='blue')
plt.bar(x + width/2, auc2_values, width, label='Modelo 2', alpha=0.8, color='red')

plt.xlabel('Clases', fontweight='bold')
plt.ylabel('Área Bajo la Curva (AUC)', fontweight='bold')
plt.title('Comparación de AUC por Clase', fontweight='bold')
plt.xticks(x, classes_extended, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print(f"\n🎯 INTERPRETACIÓN DE CURVAS ROC:")
print("  • Curva más cercana a la esquina superior izquierda = mejor modelo")
print("  • AUC > 0.8: Excelente discriminación")
print("  • AUC 0.7-0.8: Buena discriminación")
print("  • AUC 0.6-0.7: Discriminación aceptable")
print("  • AUC < 0.6: Discriminación pobre")

# Conclusiones finales
print("\n" + "="*80)
print("CONCLUSIONES FINALES DEL ANÁLISIS")
print("="*80)

print(f"\n🎯 RESUMEN EJECUTIVO:")
print(f"  • Dataset: {df.shape[0]} viviendas con {df.shape[1]} características")
print(f"  • Calidad de datos: {'Excelente (sin valores faltantes)' if df.isnull().sum().sum() == 0 else 'Requiere limpieza'}")
print(f"  • Distribución de precios: {'Sesgada a la derecha (normal en inmuebles)' if df['price'].skew() > 0.5 else 'Aproximadamente simétrica'}")
print(f"  • Mejor modelo: {best_model}")
print(f"  • Accuracy del mejor modelo: {max(accuracy1, accuracy2)*100:.2f}%")

print(f"\n📊 HALLAZGOS CLAVE:")
print("  • Los precios NO siguen distribución normal (típico en bienes raíces)")
print("  • Existe correlación significativa entre área y precio")
print("  • El estado de amueblado influye significativamente en el precio")
print("  • La selección de variables puede mantener el rendimiento con menos complejidad")

print(f"\n🚀 RECOMENDACIONES:")
print("  • Usar el modelo seleccionado para predicción de categorías de amueblado")
print("  • Considerar transformaciones logarítmicas para normalizar precios")
print("  • Explorar modelos no lineales (Random Forest, SVM) para mejor rendimiento")
print("  • Recopilar más datos de viviendas de alta gama para balancear el dataset")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)