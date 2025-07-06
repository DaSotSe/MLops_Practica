
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MultiLabelBinarizer
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets_OHE import POIDataset
from model import MultiModalNet  

# Funciones de entrenamiento y evaluación 

def train_epoch(model, device, dataloader, criterion, optimizer, scheduler=None, l1_lambda=0.0, l2_lambda=0.0):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for features, images, labels in dataloader:
        features, images, labels = features.to(device), images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features, images)
        loss = criterion(outputs, labels)
        # L1
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    return avg_loss, accuracy, current_lr


def eval_epoch(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for features, images, labels in dataloader:
            features, images, labels = features.to(device), images.to(device), labels.to(device)
            outputs = model(features, images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, images, labels in dataloader:
            features, images = features.to(device), images.to(device)
            outputs = model(features, images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels

# Carga y preprocesamiento 
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Convertir strings a listas reales
    df['tags'] = df['tags'].apply(eval)
    df['categories'] = df['categories'].apply(eval)
    # One-hot encoding 
    mlb = MultiLabelBinarizer()
    one_hot = mlb.fit_transform(df['categories'])
    df = pd.concat([df, pd.DataFrame(one_hot, columns=mlb.classes_)], axis=1)
    df.drop(columns=['categories', 'tags'], inplace=True)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Log-transform 
    for col in ['Likes', 'Bookmarks', 'Dislikes']:
        df[f"{col}_log"] = np.log1p(df[col])
    # Escalado 
    df['Visits_scaled'] = StandardScaler().fit_transform(df[['Visits']])
    for col in ['Likes_log', 'Bookmarks_log', 'Dislikes_log']:
        df[f"{col.replace('_log','')}_scaled"] = RobustScaler().fit_transform(df[[col]])
    # Target 
    df['target_cont'] = (
        df['Visits_scaled'] + df['Likes_scaled'] + df['Bookmarks_scaled'] - df['Dislikes_scaled']
    )
    df['pop_class'] = pd.qcut(df['target_cont'], q=3, labels=[0,1,2]).astype(int)
    # Eliminar columnas 
    drop_cols = ['shortDescription','id','name','Visits','Likes','Bookmarks','Dislikes',
                 'Visits_scaled','Likes_log','Bookmarks_log','Dislikes_log','Likes_scaled',
                 'Bookmarks_scaled','Dislikes_scaled','target_cont']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    # Convertir ints a float64 
    exclude = ['main_image_path','pop_class']
    int_cols = [c for c,d in df.dtypes.items() if d=='int64' and c not in exclude]
    df[int_cols] = df[int_cols].astype('float64')
    return df


def split_data(df: pd.DataFrame,
               test_frac: float = 0.1,
               val_frac_of_train: float = 0.1333,
               random_state: int = 42):
   
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df['pop_class'], random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=val_frac_of_train,
        stratify=train_val['pop_class'], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_transforms(split: str = 'train', img_size=(224,224)) -> transforms.Compose:
    if split=='train':
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(0.3,0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])


def create_dataloaders(train_df, val_df, test_df, metadata_cols: list,
                       batch_size: int = 64, num_workers: int = 4):
    train_ds = POIDataset(
        target=train_df['pop_class'].values,
        image_paths=train_df['main_image_path'].values,
        features=train_df[metadata_cols].values,
        transform=get_transforms('train')
    )
    val_ds = POIDataset(
        target=val_df['pop_class'].values,
        image_paths=val_df['main_image_path'].values,
        features=val_df[metadata_cols].values,
        transform=get_transforms('val')
    )
    test_ds = POIDataset(
        target=test_df['pop_class'].values,
        image_paths=test_df['main_image_path'].values,
        features=test_df[metadata_cols].values,
        transform=get_transforms('val')
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_network_decay_fn_act(model_class, activation_function,
                               train_loader, val_loader, test_loader,
                               device, learning_rate, num_epochs,
                               l1_lambda, l2_lambda):

    mlflow.start_run()
    # Log parametris
    mlflow.log_param("model_class", model_class.__name__)
    mlflow.log_param("activation_function", activation_function.__name__)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("l1_lambda", l1_lambda)
    mlflow.log_param("l2_lambda", l2_lambda)
    mlflow.log_param("device", str(device))


    num_meta = train_loader.dataset.features.shape[1]

 
    model = model_class(
    n_features=num_meta,
    activation_function=activation_function,
    ).to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lrs = []

    start_train = time.time()
    for epoch in range(num_epochs):
        train_loss, train_acc, lr = train_epoch(model, device, train_loader,
                                                criterion, optimizer, None,
                                                l1_lambda, l2_lambda)
        val_loss, val_acc = eval_epoch(model, device, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lrs.append(lr)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("lr", lr, step=epoch)

    elapsed_train = time.time() - start_train
    mlflow.log_metric("train_duration_secs", elapsed_train)

    start_eval = time.time()
    all_preds, all_targets = evaluate_model(model, test_loader, device)
    elapsed_eval = time.time() - start_eval

    test_acc = 100 * sum(p==t for p,t in zip(all_preds, all_targets)) / len(all_targets)
    mlflow.log_metric("test_acc", test_acc)
    mlflow.log_metric("eval_duration_secs", elapsed_eval)

    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()
    return model

