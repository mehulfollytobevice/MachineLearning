#import libraries
from sklearn import datasets, manifold, model_selection
import config


def folds(df):
    # creating a new column kfold and filling it with -1
    df["kfold"]=-1
    
    # shuffling the dataset
    df=df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y= df.target.values

    # initiate the kfold class from model_selection module
    kf= model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,"kfold"]= f

    # save the new csv with the kfold column
    df.to_csv(config.TRAINING_FILE,index=False)


if __name__=="__main__":
    data= datasets.fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True
    )
    pixel_values, targets = data
    targets= targets.astype(int)

    pixel_values['target']=targets
    # print(pixel_values)
    folds(pixel_values)