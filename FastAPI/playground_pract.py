from fastapi import FastAPI , Query, HTTPException
import pandas as pd
import torch
from transformers import pipeline, set_seed #Importamos el paqUETE 

app = FastAPI()

#Aqui creamos los metodos que consideremos 


@app.get('/Saluda') #
def saluda(name: str): 
    return {'message': f'Hola Soy {name}'}

@app.get('/Despedida')
def despedida(name: str):
    return {'message': f'Hasta pronto, {name} 👋'}


@app.get('/read_dataframe')#Para leer un dataframe de prueba 
def read_dataframe(): 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    return {'message': df}

@app.get('/get-sepal-length')
def read_sepal_length(position: int):#Importante hay que indicarle el tipado
    print('Este es el type', type(position)) 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    value = df['sepal_length'][position]
    return {'message': value}

@app.get('/text-classification') #Textos positivos o negativos de INGLES 
def text_classification(query: str):
    pipe = pipeline("text-classification")
    classified = pipe(query)
    return {'message': classified}


@app.get('/parafrasear') 
def parafrasear(query: str):
    pipe = pipeline("text2text-generation", model="milyiyo/paraphraser-spanish-t5-small")
    prompt = f"Parafrasea el texto: {query}"
    resultado = pipe(prompt, max_length=80, do_sample=False)
    return {'message': resultado[0]['generated_text']}


@app.get('/transforma-creativo')
def transforma_creativo(query: str):
    pipe = pipeline("text-generation", model="PlanTL-GOB-ES/gpt2-base-bne")
    prompt = f"Escribe sobre el tema: {query}"
    resultado = pipe(prompt, max_new_tokens=80, do_sample=True, top_k=50)
    return {'message': resultado[0]['generated_text']}


@app.get('/rima-nueva')  # Genera poema o rima breve
def rima_nueva(query: str = Query(..., description="Palabra o tema para rimar")):
    pipe = pipeline(
        "text2text-generation",
        model="DrishtiSharma/poem-gen-spanish-t5-small-v6",
        device=-1  
    )
    prompt = f"poema: estilo: libre && sentimiento: alegre && palabras: {query} && texto: {query}"
    resultado = pipe(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        repetition_penalty=1.2
    )
    return {'message': resultado[0]['generated_text']}


@app.get('/rima-creativa2')
def rima_creativa(query: str = Query(..., description="Palabra o inicio del verso")):
    pipe = pipeline(
        "text-generation",
        model="PlanTL-GOB-ES/gpt2-base-bne",
        device=-1
    )
    set_seed(42)

    # Prompt más simple que imita un poema en marcha
    prompt = f"{query},\n"

    resultado = pipe(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=1.0,
        top_k=40,
        repetition_penalty=1.2
    )
    return {'message': resultado[0]['generated_text']}

