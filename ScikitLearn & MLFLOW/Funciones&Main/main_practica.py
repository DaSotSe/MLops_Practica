
import funciones_practica as func
import torch
import mlflow


def main():

    # Carga y preprocesamiento
    df = func.load_dataset(
        "/Users/davidsoteloseguin/Library/Mobile Documents/com~apple~CloudDocs/Personal/Formacion /Bootcamp/Bootcamp KC/mlops/Practiva/poi_dataset.csv"
    )
    df = func.preprocess_dataframe(df)

    #Split 
    train_df, val_df, test_df = func.split_data(df)

    #Definir columnas de metadatos (excluyendo ruta de imagen y target)
    metadata_cols = [
        col for col in train_df.columns
        if col not in ('pop_class', 'main_image_path')
    ]

    #Crear DataLoaders
    train_loader, val_loader, test_loader = func.create_dataloaders(
        train_df,
        val_df,
        test_df,
        metadata_cols,
        batch_size=64,
        num_workers=4
    )

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Definir experimento en MLflow
    mlflow.set_experiment("POI_OHE")

    #Entrenamiento y registro en MLflow
    func.train_network_decay_fn_act(
        model_class=func.MultiModalNet,
        activation_function=torch.nn.LeakyReLU,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=1e-3,
        num_epochs=10,
        l1_lambda=1e-5,
        l2_lambda=1e-4
    )

if __name__ == '__main__':
    main()