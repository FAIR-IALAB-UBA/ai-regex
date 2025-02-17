import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import time
from openai import OpenAI
import json

client = OpenAI()

cat_id = {
    "Actividad Crítica": 1,
    "Fonaindo con Bruno": 2,
    "Fonaindo sin Bruno": 3,
    "Franqueros con reso 499": 4,
    "Franqueros sin reso 499": 5,
    "Material Didáctico": 6
}

# Read the Excel file with the category and content (Ground Truth)
df_content = pd.read_excel('categorias_documentos.xlsx')

# Read the Excel file with the categories and regex
df_regex = pd.read_excel('categoria_regexs.xlsx')

# Create a dictionary of regular expressions by category
regex_dict = {}
for _, row in df_regex.iterrows():
    categoria = row['EJE TEMÁTICO']
    regex_text = row['CONJUNTO DE REGEX']
    regex_list = regex_text.strip().split('\n')
    if categoria not in regex_dict:
        regex_dict[categoria] = []
    regex_dict[categoria].extend(regex_list)


def evaluate_regex(method):
    # Check the regular expressions in the content
    resultados = []
    tiempos = []
    for _, row in df_content.iterrows():
        contenido = row['CONTENIDO']
        mejor_categoria = None
        mejor_porcentaje = 0
        start_time = time.time()
        for categoria, regex_list in regex_dict.items():
            total_regex = len(regex_list)
            match_count = sum(1 for regex in regex_list if re.search(regex, contenido))
            porcentaje = match_count / total_regex
            if porcentaje >= 0.10 and porcentaje > mejor_porcentaje:
                mejor_categoria = categoria
                mejor_porcentaje = porcentaje
        end_time = time.time()
        tiempos.append(end_time - start_time)
        resultados.append((contenido, mejor_categoria, mejor_porcentaje))
    create_dataframe_results(resultados, tiempos)


def get_response(system_content, user_content_query):
    completion = client.chat.completions.create(
        model = "gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content_query + "\nRespond in JSON format, only one key and one value: {'category':category}"}
        ]
    )
    return completion.to_dict()


def evaluate_llm():
    with open('prompt.txt', 'r', encoding='utf-8') as file:
        prompt = file.read()
        resultados = []
        tiempos = []
        for _, row in df_content.iterrows():
            contenido = row['CONTENIDO']
            start_time = time.time()
            completion = get_response(prompt, contenido)
            end_time = time.time()
            output_info = json.loads(completion["choices"][0]["message"]["content"])
            mejor_categoria = output_info['category']
            tiempos.append(end_time - start_time)
            resultados.append((contenido, cat_id[mejor_categoria], 0))
        create_dataframe_results(resultados, tiempos)


def create_dataframe_results(results, times):
    df_resultados = pd.DataFrame(results, columns=['contenido', 'categoria', 'porcentaje'])
    df_resultados['categoria_real'] = df_content['EJE TEMÁTICO']
    
    # Ensure that the columns are of type object
    df_resultados = df_resultados.dropna(subset=['categoria', 'categoria_real'])


    # Calcular las métricas
    precision = precision_score(df_resultados['categoria_real'], df_resultados['categoria'], average='weighted', zero_division=0)
    recall = recall_score(df_resultados['categoria_real'], df_resultados['categoria'], average='weighted', zero_division=0)
    f1 = f1_score(df_resultados['categoria_real'], df_resultados['categoria'], average='weighted', zero_division=0)
    accuracy = accuracy_score(df_resultados['categoria_real'], df_resultados['categoria'])

    # Mostrar las métricas en un gráfico
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')

    # Agregar etiquetas a cada barra
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{value:.2f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')

    plt.show()

    # Calcular y mostrar el tiempo promedio
    tiempo_promedio = sum(times) / len(times)
    print(f"Tiempo promedio requerido para computar la categoría de un documento: {tiempo_promedio:.4f} segundos")