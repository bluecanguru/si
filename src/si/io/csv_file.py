from si.data.dataset import Dataset
import pandas as pd
import numpy as np
 
def read_csv(filename: str, sep: str, features: bool, label: bool) -> Dataset:
    
    dataframe = pd.read_csv(filepath_or_buffer=filename, sep=sep)
    if features and label:
        X = dataframe.iloc[:, :-1].to_numpy() # convertemos porque x e y np
        y = dataframe.iloc[:, -1].to_numpy()
        feature_names = dataframe.columns[:-1].tolist()
        label_name = dataframe.columns[-1]
        return Dataset(X=X, y=y, features=feature_names, label=label_name)
    elif features and not label:
        X = dataframe.to_numpy()
        feature_names = dataframe.columns
        return Dataset(X=X, features=feature_names)
    elif not features and label:
        X = np.array() 
        y = dataframe.iloc[:, -1].to_numpy()
        label_name = dataframe.columns[-1]
        return Dataset(X=X, y=y, label=label_name) #chama o construtor (funçao init que corre) e faz a correspondencia entre os parametros e as instancias
    else:
        return None
    
def write_csv(filename:str, sep:str = ";", dataset = Dataset, features: bool = False, label: bool =False) -> None:
    df = pd.Dataframe(dataset.x)
    if features:
        df.columns = dataset.features
        
    if label:
        y = dataset.y
        label_name = dataset.label
        df[label_name] = y
   
    else:
        y = None
        label_name = None
    
    df.to_csv(filename, sep=sep, index=False)import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str,
             sep: str = ',',
             features: bool = False,
             label: bool = False) -> Dataset:
    """
    Reads a csv file (data file) into a Dataset object

    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False

    Returns
    -------
    Dataset
        The dataset object
    """
    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = data.columns[-1]

    else:
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X=X, y=y, features=features, label=label)


def write_csv(filename: str,
              dataset: Dataset,
              sep: str = ',',
              features: bool = False,
              label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False
    """
    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False)