import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import  SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import SGDRegressor


SEED=42

def read_file(filepath):
        filepath = filepath
        return pd.read_csv(filepath)

def clean_data( dataset, columns: list):
        return dataset.drop(columns=columns)
    
def encode( dataset, columns: list):
    # Encode the labels
    ohe = OneHotEncoder()
    encoded = ohe.fit_transform(dataset[columns]).toarray()
    _columns = ohe.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded, columns=_columns)
    dataset = pd.concat([dataset, encoded_df], axis=1)
    return dataset.drop(columns=columns)

def normalize(dataset, columns: list):
    # Normalize the dataset
    norm = StandardScaler()
    norm_ = norm.fit_transform(dataset[columns])
    norm_df = pd.DataFrame(norm_, columns=[col + "_norm" for col in columns])
    dataset = pd.concat([dataset, norm_df], axis=1)
    return dataset.drop(columns=columns)


def split_dataset(dataset, columns_X, column_y):
    X = dataset.drop(columns=columns_X, axis=1)
    y = dataset[column_y]
    return X, y

def print_r2_score(y_test, y_pred):
    acc = r2_score(y_test, y_pred)
    print(f"R2 Score: {acc} ")
    return acc

def print_mse(y_test, y_pred):
    acc = mean_squared_error(y_test, y_pred)
    print(f"MSE Score: {acc} ")
    return acc

def cross(model, X, y, cv=5, scoring="neg_mean_squared_error"):
    score = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    lin_r2_scores = np.sqrt(-score)
    print(f"Scores: {lin_r2_scores}")
    print(f"Mean: {lin_r2_scores.mean()}")
    print(f"Standard Deviation: {lin_r2_scores.std()}")


def reduce(X_train, X_test,y_train, k=50):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector = selector.fit(X_train, y_train)
    X_train_selector = selector.transform(X_train)
    X_test_selector = selector.transform(X_test)
    return X_train_selector, X_test_selector

def param_search(model, param_dist, X_train, y_train):
    search = RandomizedSearchCV(model, random_state=SEED,param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    print(search.best_params_)

def fetch_data():
    dataset = read_file("../../FM 2023.csv")

    dataset = clean_data(dataset, columns=["Rental club",
                                    "Age",
                                    "Position",
                                    "Salary",
                                    "Values",
                                    "Race",
                                    "UID", 
                                    "Date of birth",
                                    "Colour of skin",
                                    "RCA",
                                    "Race",
                                    "Club",
                                    "Nationality",
                                    "Name",
                                    "Current reputation", 
                                    "Domestic reputation",
                                    "World reputation"
                                    ]
                    )
    return dataset


if __name__ == "__main__":
    np.random.seed(SEED)
    dataset = fetch_data()
    dataset = normalize(dataset, columns=dataset.columns)
    #dataset = encode(dataset, columns=['Position'])

    X = dataset.drop(columns=["ca_norm", "pa_norm"], axis=1)
    y = dataset["ca_norm"]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=SEED
    )

    param_dist= {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'fit_intercept': [True, False],
        'learning_rate': ['optimal', 'invscaling']
    }
    sgd = SGDRegressor(penalty='l2', learning_rate='invscaling', fit_intercept=True, alpha=0.0001)

    #param_search(sgd, param_dist, X_train, y_train)

    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    mse = print_mse(y_test, y_pred)
    r2 = print_r2_score(y_test, y_pred)
    cross(sgd, X, y, cv=5)
    
    filename = 'current_ability_regression_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(sgd, f)
        print("Model Succssefully Saved As {}".format(filename))
    
    #pickle.dump(sgd, open('current_ability_regression_model.pkl', 'wb'))