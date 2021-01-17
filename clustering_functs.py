import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display
    display.max_columns = 100000
    display.max_rows = 200000
    display.max_colwidth = 200000
    display.width = None
    # display.precision = 2  # set as needed
    pass

def pre1():
    set_pandas_display_options()
    filename_1 = 'beacons_dataset.csv'
    df = pd.read_csv(filename_1, sep=';')
    df = df.dropna()
    a = sorted(df.room.unique())
    #print(df)
    #print(a)
    #print(len(a))
    ###

    list_kicken = ['Kichen', 'Kitcen', 'Kitch', 'Kitcheb', 'Kitchen',
                   'Kitchen2', 'Kithen', 'Kitvhen', 'Kiychen', 'kitchen', 'K']

    list_bathroom = ['Bqthroom', 'Bsthroom', 'Baghroom', 'Barhroom', 'Bathroim',
                     'Bathroom', 'Bathroom-1', 'Bathroom1', 'Bathroon', 'Laundry', 'LaundryRoom', 'Washroom']

    list_living = ['LeavingRoom', 'Leavingroom', 'Leavivinroom', 'Liningroom',
                   'LivibgRoom', 'Living', 'LivingRoom', 'LivingRoom2', 'Livingroom',
                   'Livingroom1', 'Livingroom2', 'Livingroon', 'Livroom', 'LuvingRoom',
                   'Luvingroom1', 'livingroom', 'SeatingRoom', 'Sitingroom', 'Sittigroom',
                   'SittingOver', 'SittingRoom', 'Sittingroom', 'Sittinroom']

    list_bedroom = ['Bed', 'Bedroom', 'Bedroom-1', 'Bedroom1', 'Bedroom1st',
                    'Bedroom2', 'bedroom', '2ndRoom']
    list_dinner = ['Dinerroom', 'DiningRoom', 'DinnerRoom', 'DinningRoom', 'DinerRoom']

    list_office = ['Office', 'Office-2', 'Office1', 'Office1st', 'Office2', 'Workroom']

    random = ['Three', 'Two', 'Left', 'One', 'Four', 'Right', 'Box', 'three', 'Box-1', 'T',
              'TV', 'Chambre', 'Desk', 'Entrance', 'Entry', 'ExitHall', 'Garage',
              'Garden', 'Guard', 'Hall', 'Library', 'Outdoor', 'Pantry', 'Storage', 'Veranda']

    df['room'] = df['room'].replace(list_kicken, 'kitchen')
    df['room'] = df['room'].replace(list_bedroom, 'bedroom')
    df['room'] = df['room'].replace(list_living, 'living')
    df['room'] = df['room'].replace(list_bathroom, 'bathroom')
    df['room'] = df['room'].replace(list_dinner, 'kitchen')  # dinner

    df['room'] = df['room'].replace(list_office, 'office')
    df['room'] = df['room'].replace(random, 'lol')
    #print(df.dtypes)
    df = df[pd.to_numeric(df['part_id'], errors='coerce').notnull()]
    #print(df)

    #  filename_1 = '1.csv'
    #  df = pd.read_csv(filename_1)
    #print(df)
    df['ts_date'] = df['ts_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    #print(df)
    #  df['ts_time'] = pd.to_timedelta(df['ts_time'].astype(str))
    df = df.sort_values(['part_id', 'ts_date', 'ts_time'])
    #print(df)

    df['ts_time'] = pd.to_timedelta(df['ts_time'].astype(str))
    print(df.dtypes)

    df['Time_diff'] = df.groupby(['part_id', 'ts_date'])['ts_time'].diff().dt.total_seconds()
    print(df.head(50))
    df.Time_diff = df.Time_diff.shift(-1)
    print(df.head(50))

    df = df.dropna()
    print(df.head(50))
    df = df.drop(columns=['ts_time', 'ts_date'])
    print(df.head(50))
    df = df.drop(df[df.Time_diff < 60].index)
    print(df.head(50))
    df['sum_m'] = df['Time_diff'].groupby(df['part_id']).transform('sum')
    print(df.head(50))

    #  df = df[df.room != 'lol']
    df['sum_l'] = df.groupby(['part_id', 'room'])['Time_diff'].transform('sum')
    print(df.head(50))

    df = df.drop_duplicates(['part_id', 'room', 'sum_m', 'sum_l'])
    print(df.head(50))

    df['Time_diff'] = (df['sum_l'] / df['sum_m']) * 100
    print(df.head(50))

    df = df.drop(['sum_m', 'sum_l'], axis=1)
    print(df.head(50))

    #  print(df.loc[df['part_id'] == '2113'])
    df.loc[df['room'] == 'kitchen', 'kitchen'] = df['Time_diff']
    df.loc[df['room'] == 'bedroom', 'bedroom'] = df['Time_diff']
    df.loc[df['room'] == 'living', 'living'] = df['Time_diff']
    df.loc[df['room'] == 'bathroom', 'bathroom'] = df['Time_diff']
    df.loc[df['room'] == 'office', 'office'] = df['Time_diff']
    df.loc[df['room'] == 'lol', 'lol'] = df['Time_diff']
    df = df.drop(['Time_diff', 'room'], axis=1)
    print(df.head(50))

    def get_notnull(x):
        if x.notnull().any():
            return x[x.notnull()]
        else:
            return np.nan

    df = df.groupby('part_id').agg(get_notnull)


    df = df.fillna(0)
    df = df.drop(['lol'], axis=1)
    print(df)
    # df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    df.to_csv('11.csv', index=True)
pass