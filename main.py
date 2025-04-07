import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import time
from openai import OpenAI
import json

client = OpenAI()

cat_id_v1 = {
    "Actividad Crítica": 1,
    "Fonaindo con Bruno": 2,
    "Fonaindo sin Bruno": 3,
    "Franqueros con reso 499": 4,
    "Franqueros sin reso 499": 5,
    "Material Didáctico": 6
}

cat_id_v2 = {
    "Material_didactico": 1,
    "Franqueros": 2,
    "Fonaindo": 3,
    "Actividad_Critica": 4,
}

# Read the Excel file with the category and content (Ground Truth)
df_content = pd.read_excel('v_2/categorias_documentos.xlsx')

# Read the Excel file with the categories and regex
df_regex = pd.read_excel('v_2/categoria_regexs.xlsx')

# Create a dictionary of regular expressions by category
regex_dict = {}
for _, row in df_regex.iterrows():
    categoria = row['EJE TEMÁTICO']
    regex_text = row['CONJUNTO DE REGEX']
    leng_natural = row['LENGUAJE NATURAL']
    regex_list = [r.strip() for r in regex_text.strip().split('\n')]
    natural = [n.strip() for n in leng_natural.strip().split('\n')]
    if categoria not in regex_dict:
        regex_dict[categoria] = {'regex':[], 'natural':[]}
    regex_dict[categoria]['regex'].extend(regex_list)
    regex_dict[categoria]['natural'].extend(natural)


def evaluate_regex():
    # Check the regular expressions in the content
    resultados = []
    tiempos = []
    for _, row in df_content.iterrows():
        contenido = row['CONTENIDO']
        mejor_categoria = None
        mejor_porcentaje = 0
        start_time = time.time()
        for categoria, regex_and_lang_list in regex_dict.items():
            regex_list = regex_and_lang_list['regex']
            total_regex = len(regex_list)
            match_count = sum(1 for regex in regex_list if re.search(regex, contenido))
            porcentaje = match_count / total_regex
            if porcentaje >= 0.60 and porcentaje > mejor_porcentaje:
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
            {"role": "user", "content": user_content_query + "\nRespond in JSON format with the following keys: 'category' and 'response'. The 'response' should be your response to the user prompt, and 'category' should containt the id of the category detected."}
        ]
    )
    return completion.to_dict()


def evaluate_llm():
    # Load system prompt and user prompt from separate files
    with open('v_2/system_prompt.txt', 'r', encoding='utf-8') as file:
        system_prompt = file.read()
    
    with open('v_2/user_prompt.txt', 'r', encoding='utf-8') as file:
        user_prompt = file.read()
    
    resultados = []
    tiempos = []
    # Only process the first 10 documents
    for i, (_, row) in enumerate(df_content.iterrows()):
        contenido = row['CONTENIDO']
        start_time = time.time()
        # Use both prompts when making the API call
        completion = get_response(system_prompt, user_prompt + "\n" + contenido)
        end_time = time.time()
        output_info = json.loads(completion["choices"][0]["message"]["content"])
        mejor_categoria = output_info['category']
        tiempos.append(end_time - start_time)
        resultados.append((contenido, mejor_categoria, output_info, 0))
    create_dataframe_results(resultados, tiempos)


def create_dataframe_results(results, times):
    df_resultados = pd.DataFrame(results, columns=['contenido', 'categoria', 'explicacion', 'porcentaje'])
    df_resultados['categoria_real'] = df_content['EJE TEMÁTICO']
    
    # Ensure that the columns are of type object
    df_resultados = df_resultados.dropna(subset=['categoria', 'categoria_real'])
    df_resultados.to_excel('v_2/resultados.xlsx', index=False)
    
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

    # # Calcular y mostrar el tiempo promedio
    #tiempo_promedio = sum(times) / len(times)
    #print(f"Tiempo promedio requerido para computar la categoría de un documento: {tiempo_promedio:.4f} segundos")

if __name__ == "__main__":
    evaluate_llm()